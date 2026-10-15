#!/usr/bin/env python3
"""Segmentation prepare."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import platform
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[3]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def pixel_sha(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str((a.shape, a.dtype.str)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
            handle.flush();os.fsync(handle.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def write_rows(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as handle:
            for row in data:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            handle.flush();os.fsync(handle.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def csv_write(path: Path, data: list[dict]) -> None:
    if data:
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)


def batch_schedule(arm: str, steps: int, batch: int, size: int, seed: int) -> list[list[tuple[str, int]]]:
    """Exact update budget; mixture has exact 1:1 quota at every update."""
    if arm not in ('real', 'synthetic', 'mixed') or batch % 2 or size < 1:
        raise ValueError('invalid arm, batch or pool size')
    rng = random.Random(seed)
    schedule = []
    for _ in range(steps):
        sources = ['real'] * batch if arm == 'real' else ['synthetic'] * batch
        if arm == 'mixed':
            sources = ['real'] * (batch // 2) + ['synthetic'] * (batch // 2)
            rng.shuffle(sources)
        if arm == 'mixed':
            choices = {'real': [i for i in range(size) if i % 4 < 2],
                       'synthetic': [i for i in range(size) if i % 4 >= 2]}
            if not all(choices.values()):
                raise ValueError('mixed corpus requires at least four candidate slots')
            schedule.append([(s, rng.choice(choices[s])) for s in sources])
        else:
            schedule.append([(s, rng.randrange(size)) for s in sources])
    return schedule


def audit_sources(exp5: Path, config: dict, out: Path) -> tuple[list[dict], dict]:
    data = exp5 / 'data/ctc_segmentation_st_v1'
    all_rows = read_rows(data / 'manifest.jsonl')
    rows = [r for r in all_rows if r['dataset'] in config['datasets']]
    checked, summaries, old_mapping = [], [], []
    for dataset in config['datasets']:
        archive_path = exp5 / 'rawdata/CTC' / f'{dataset}.zip'
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            dataset_rows = [r for r in rows if r['dataset'] == dataset]
            seen = set()
            for r in dataset_rows:
                seq, t = r['sequence'], int(r['time_index'])
                assert r['annotation_source'] == 'st' and int(r['z_index']) == -1
                assert (seq, t) not in seen
                seen.add((seq, t))
                image_member = f'{dataset}/{seq}/t{t:03d}.tif'
                mask_member = r['source_annotation']
                assert f'/{seq}_ST/SEG/' in mask_member
                assert int(re.search(r'(\d+)\.tif$', mask_member)[1]) == t
                # ZipFile.read checks each member's CRC before returning.
                raw = np.asarray(tifffile.imread(io.BytesIO(archive.read(image_member))))
                mask = np.asarray(tifffile.imread(io.BytesIO(archive.read(mask_member))))
                prepared_image = np.asarray(tifffile.imread(data / r['image_path']))
                prepared_mask = np.asarray(tifffile.imread(data / r['mask_path']))
                assert raw.shape == mask.shape == (r['height'], r['width'])
                assert np.issubdtype(mask.dtype, np.integer) and mask.min() >= 0
                assert raw.dtype == prepared_image.dtype and np.array_equal(raw, prepared_image)
                assert np.array_equal(mask, prepared_mask), f'Label mismatch: {r["sample_id"]}'
                count = int(np.count_nonzero(np.unique(mask) > 0))
                assert count == r['num_instances_on_plane']
                checked.append({
                    **r, 'source_archive': str(archive_path),
                    'image_path': str((data / r['image_path']).resolve()),
                    'mask_path': str((data / r['mask_path']).resolve()),
                    'source_image_member': image_member, 'source_image_crc32': f'{archive.getinfo(image_member).CRC:08x}',
                    'source_mask_crc32': f'{archive.getinfo(mask_member).CRC:08x}',
                    'zip_crc_verified': True, 'image_pixels_match_zip': True, 'mask_pixels_match_zip': True,
                    'image_pixel_sha256': pixel_sha(raw), 'mask_pixel_sha256': pixel_sha(mask),
                    'native_dtype': str(raw.dtype), 'mask_dtype': str(mask.dtype),
                })
            for seq in ('01', '02'):
                seq_rows = [r for r in checked if r['dataset'] == dataset and r['sequence'] == seq]
                assert sorted(r['time_index'] for r in seq_rows) == list(range(92))
                gt = [n for n in names if f'/{seq}_GT/SEG/' in n and n.endswith('.tif')]
                summaries.append({'dataset': dataset, 'sequence': seq, 'frames': len(seq_rows),
                                  'shape': [seq_rows[0]['height'], seq_rows[0]['width']],
                                  'instances_over_frames': sum(r['num_instances_on_plane'] for r in seq_rows),
                                  'ST_frames': len(seq_rows), 'sparse_GT_files': len(gt)})
            slug = dataset.lower().rstrip('+')
            old_root = exp5 / 'data/raw_morphology_v1' / f'ctc_{slug}'
            for old in read_rows(old_root / 'annotations/train.jsonl'):
                m = old['metadata']
                member = m['source_seg_member']
                seq = Path(member).parts[1].replace('_GT', '').replace('_ST', '')
                time_index = int(re.search(r'(\d+)\.tif$', member)[1])
                correct = f'{dataset}/{seq}/t{time_index:03d}.tif'
                original = m.get('source_image_member')
                old_mapping.append({'dataset': dataset, 'image_id': old['image_id'],
                                    'annotation_kind': 'ST' if '_ST/SEG/' in member else 'GT',
                                    'source_mask_member': member, 'reported_image_member': original,
                                    'reported_member_exists': original in names, 'corrected_image_member': correct,
                                    'corrected_member_exists': correct in names,
                                    'uses_correct_ST_prepared_source': False})
        print(f'audited ZIP/raw/ST: {dataset} {len(dataset_rows)} pairs', flush=True)
    crossing = {}
    for d in config['datasets']:
        a = {r['image_pixel_sha256'] for r in checked if r['dataset'] == d and r['sequence'] == '01'}
        b = {r['image_pixel_sha256'] for r in checked if r['dataset'] == d and r['sequence'] == '02'}
        crossing[d] = len(a & b)
    write_rows(out / 'verified_pairs.jsonl', checked)
    csv_write(out / 'raw_morphology_member_corrections.csv', old_mapping)
    summary = {'source_read_only': True, 'verified_pairs': len(checked), 'sequences': summaries,
               'all_prepared_image_and_mask_pixels_match_zip': True,
               'raw_morphology_mapping': dict(Counter(f'{r["dataset"]}:{r["annotation_kind"]}:valid={r["reported_member_exists"]}' for r in old_mapping)),
               'exact_cross_sequence_image_hash_collisions': crossing,
               'ST_is_manual_Gold': False, 'exp5_prior_use_allowed_for_current_task': True,
               'workspace_never_seen_test_claim': False,
               'full_ST_inventory_frame_counts': dict(Counter(f'{r["dataset"]}:{r["sequence"]}' for r in all_rows if int(r['z_index']) in (-1, 0)))}
    write_json(out / 'data_audit.json', summary)
    return checked, summary


def audit_assets(exp5: Path, config: dict, out: Path) -> dict:
    spec = config['initialization']['cellpose_sam']
    cpsam = exp5 / spec['path_relative_to_exp5']
    digest = sha_file(cpsam) if cpsam.is_file() else None
    generic = {'path': str(cpsam), 'exists': cpsam.is_file(),
               'bytes': cpsam.stat().st_size if cpsam.is_file() else None,
               'sha256': digest, 'verified': digest == spec['expected_sha256'] and cpsam.stat().st_size == spec['expected_bytes'] if cpsam.is_file() else False}
    historical = []
    for d in config['datasets']:
        for model in config['models']:
            for fold in ('train01_test02', 'train02_test01'):
                root = exp5 / 'outputs/ctc_segmentation_st_three_models_v1' / d / model / fold
                p = root / 'training/summary.json'
                hist = json.loads(p.read_text()) if p.is_file() else {}
                cp = root / 'training/best.pt' if model != 'cellpose_sam' else root / f'cellpose_data/train/models/ctc_{d}_train{fold[5:7]}_cpsam_v2'
                historical.append({'dataset': d, 'model': model, 'fold': fold,
                                   'local_checkpoint': str(cp), 'exists_locally': cp.is_file(),
                                   'server_path_from_summary': hist.get('best_checkpoint'),
                                   'server_checkpoint_verified_in_this_task': False,
                                   'eligible_common_four_arm_initialization': False,
                                   'reason': 'CTC sequence-specific fine-tune, not a target-free common generic pretrain'})
    state = json.loads((ROOT / 'configs/evaluation/v4_foreground_renderer_stage26_v1/manifest_audit.json').read_text())
    generator_assets = {k: {'path': str(ROOT / config['synthesis'][k]), 'exists_locally': (ROOT / config['synthesis'][k]).is_file()}
                        for k in ('checkpoint_relative_to_exp4', 'vae_checkpoint_relative_to_exp4')}
    assets = {'implementations': [str(exp5 / 'scripts/ctc_segmentation' / p) for p in ('models.py', 'train.py', 'predict.py', 'run_crossval.py')],
              'generic_cpsam_v2': generic,
              'classical_pretraining': {'unet': 'not found locally; old train.py creates random model',
                                       'scherr_distance': 'not found locally; old train.py creates random model'},
              'historical_ctc_weights': historical, 'generator_assets': generator_assets,
              'generator_state_vocabulary': state['state_to_id'],
              'GOWT1_trained_state_available': any('gowt' in s.lower() for s in state['state_to_id']),
              'cellpose_installed_in_preparation_python': __import__('importlib.util', fromlist=['find_spec']).find_spec('cellpose') is not None,
              'python': sys.executable, 'python_version': platform.python_version(),
              'numpy': np.__version__, 'tifffile': tifffile.__version__}
    write_json(out / 'model_asset_audit.json', assets)
    print(f'cpsam_v2 verified={generic["verified"]}; classical common pretrain absent', flush=True)
    return assets


def prepare_requests(pairs: list[dict], config: dict, out: Path) -> list[dict]:
    all_runs = []
    train = config['training']
    for d in config['datasets']:
        for direction in config['directions']:
            train_seq, test_seq = direction['train'], direction['test']
            fold = f'{d}/train{train_seq}_test{test_seq}'
            selected = sorted((r for r in pairs if r['dataset'] == d and r['sequence'] == train_seq), key=lambda r: r['time_index'])
            test = [r for r in pairs if r['dataset'] == d and r['sequence'] == test_seq]
            requests = []
            for r in selected:
                for j in range(train['crops_per_training_frame']):
                    ident = f'{r["sample_id"]}__crop{j:02d}'
                    stable = int(hashlib.sha256(f'{config["synthesis"]["condition_seed"]}:{ident}'.encode()).hexdigest()[:16], 16)
                    rng = random.Random(stable)
                    size = train['condition_crop_size']
                    top, left = rng.randrange(r['height'] - size + 1), rng.randrange(r['width'] - size + 1)
                    rel = f'conditions/{fold}/{ident}'
                    requests.append({'condition_id': ident, 'dataset': d, 'train_sequence': train_seq,
                                     'test_sequence': test_seq, 'parent_frame_id': r['frame_id'],
                                     'parent_image_path': r['image_path'], 'parent_mask_path': r['mask_path'],
                                     'parent_image_sha256': r['image_pixel_sha256'], 'parent_mask_sha256': r['mask_pixel_sha256'],
                                     'crop_top': top, 'crop_left': left, 'crop_size': size,
                                     'crop_policy': 'uniform coordinates fixed before model outcomes; all IDs retained',
                                     'real_image_path': f'{rel}/real.tif', 'instance_mask_path': f'{rel}/instances.tif',
                                     'support_path': f'{rel}/support.png', 'background_path': f'{rel}/background.png',
                                     'synthetic_image_path': f'generated/{fold}/{ident}.png',
                                     'generator_seed': stable % (2**63 - 1),
                                     'generator_source': 'dynamicnuclearnet' if d == 'Fluo-N2DL-HeLa' else None,
                                     'generator_state_key': 'dynamicnuclearnet::hela' if d == 'Fluo-N2DL-HeLa' else None,
                                     'source_role': 'training_side_only', 'target_sequence_input_used': False,
                                     'pair_status': 'planned_not_generated', 'instance_label_policy': 'exact crop of ST map; do not relabel by connected components'})
            folder = out / 'folds' / fold
            write_rows(folder / 'generation_requests.jsonl', requests)
            write_rows(folder / 'test_pairs.jsonl', test)
            write_rows(folder / 'train_parent_pairs.jsonl', selected)
            for arm in ('real', 'synthetic', 'mixed'):
                pool = []
                for index, request in enumerate(requests):
                    source = arm if arm != 'mixed' else ('real' if index % 4 < 2 else 'synthetic')
                    pool.append({'condition_id': request['condition_id'], 'parent_frame_id': request['parent_frame_id'],
                                 'dataset': d, 'sequence': train_seq, 'source': source,
                                 'image_path': request['real_image_path'] if source == 'real' else request['synthetic_image_path'],
                                 'mask_path': request['instance_mask_path'], 'request_index': index})
                # Mixed candidate corpus has N total pairs, never 2N.
                assert len(pool) == len(requests)
                write_rows(folder / f'{arm}_pool.jsonl', pool)
            write_json(folder / 'pool_contract.json', {'train_frames': len(selected), 'test_frames': len(test),
                        'real_pool': len(requests), 'synthetic_pool': len(requests), 'mixed_total': len(requests),
                        'mixed_real': len(requests) // 2, 'mixed_synthetic': len(requests) // 2,
                        'total_draws_per_run': train['optimizer_updates'] * train['effective_batch'],
                        'mixed_draws_per_run': {'real': train['optimizer_updates'] * 4, 'synthetic': train['optimizer_updates'] * 4},
                        'real_and_synthetic_pool_are_plans_until_materialized': True})
            for model in config['models']:
                all_runs.append({'run_id': f'{fold}/{model}/pretrained_only', 'dataset': d, 'model': model,
                                 'direction': f'{train_seq}->{test_seq}', 'arm': 'pretrained_only', 'seed': None,
                                 'optimizer_updates': 0, 'effective_batch': 0,
                                 'status': 'baseline_weights_and_native_prediction_not_executed'})
                for arm in ('real', 'synthetic', 'mixed'):
                    for seed in config['seeds']:
                        all_runs.append({'run_id': f'{fold}/{model}/{arm}/seed{seed}', 'dataset': d, 'model': model,
                                         'direction': f'{train_seq}->{test_seq}', 'arm': arm, 'seed': seed,
                                         'optimizer_updates': train['optimizer_updates'], 'effective_batch': train['effective_batch'],
                                         'status': 'planned_not_trained'})
            print(f'prepared {fold}: real={len(requests)} synthetic={len(requests)} mixed={len(requests)}', flush=True)
    write_json(out / 'run_matrix.json', all_runs)
    csv_write(out / 'run_matrix.csv', all_runs)
    return all_runs


def prepare_from_manifest(data_dir: Path, config: dict, out: Path) -> list[dict]:
    """Prepare metadata-only requests from the provider-converted CTC manifest."""
    if out.exists():
        raise FileExistsError(out)
    rows = [r for r in read_rows(data_dir / 'manifest.jsonl') if r['dataset'] in config['datasets']]
    pairs = []
    expected = config['training'].get('selected_training_frames_per_direction', 92)
    for dataset in config['datasets']:
        for sequence in ('01', '02'):
            selected = sorted((r for r in rows if r['dataset'] == dataset and r['sequence'] == sequence), key=lambda r: r['time_index'])
            if len(selected) != expected:
                raise ValueError(f'{dataset}/{sequence}: expected {expected} frames, got {len(selected)}')
            if len({r['time_index'] for r in selected}) != len(selected):
                raise ValueError('Duplicate parent frame')
            for r in selected:
                if r['annotation_source'] != 'st' or int(r['z_index']) != -1:
                    raise ValueError('Paper protocol requires two-dimensional ST labels')
                ip, mp = data_dir / r['image_path'], data_dir / r['mask_path']
                image, mask = np.asarray(tifffile.imread(ip)), np.asarray(tifffile.imread(mp))
                if image.shape != mask.shape or image.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
                    raise ValueError('Invalid native image/instance map')
                pairs.append({**r, 'image_path': str(ip.resolve()), 'mask_path': str(mp.resolve()),
                              'image_pixel_sha256': pixel_sha(image), 'mask_pixel_sha256': pixel_sha(mask)})
    out.mkdir(parents=True)
    prepare_requests(pairs, config, out)
    write_json(out / 'protocol_snapshot.json', config)
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description='Prepare fixed foreground-resynthesis requests from converted CTC data.')
    p.add_argument('--ctc-data-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/segmentation_protocol.json')
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    pairs = prepare_from_manifest(args.ctc_data_dir.resolve(), json.loads(args.config.read_text()), args.out_dir.resolve())
    print(json.dumps({'parent_frames': len(pairs), 'output': str(args.out_dir)}))


if __name__ == '__main__':
    main()
