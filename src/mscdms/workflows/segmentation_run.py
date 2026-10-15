#!/usr/bin/env python3
"""Segmentation run."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from mscdms.workflows.segmentation_prepare import ROOT, batch_schedule, read_rows, sha_file, write_json
from mscdms.workflows.segmentation_metrics import evaluate_manifest


def transformed_patch(image: np.ndarray, labels: np.ndarray, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    y, x = rng.randrange(image.shape[0] - size + 1), rng.randrange(image.shape[1] - size + 1)
    im, lab = image[y:y+size, x:x+size], labels[y:y+size, x:x+size]
    if rng.random() < .5:
        im, lab = np.flip(im, 0), np.flip(lab, 0)
    if rng.random() < .5:
        im, lab = np.flip(im, 1), np.flip(lab, 1)
    k = rng.randrange(4)
    return np.ascontiguousarray(np.rot90(im, k)), np.ascontiguousarray(np.rot90(lab, k))


def budget_train(net, optimizer, steps: int, batch: int, micro: int, loss_for_positions, callback, start: int = 0) -> int:
    """Small reusable loop, tested on CPU independently of Cellpose availability."""
    if batch % micro:
        raise ValueError('microbatch must divide the effective batch')
    accumulation = batch // micro
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    for step in range(start, steps):
        # Schedule is a function of optimizer update, independent of pool size.
        factor = 0.5 * (1 + math.cos(math.pi * step / max(steps - 1, 1)))
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = base_lr * factor
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for begin in range(0, batch, micro):
            loss = loss_for_positions(step, list(range(begin, begin + micro)))
            if not bool(__import__('torch').isfinite(loss)):
                raise FloatingPointError(f'nonfinite training loss at update {step+1}')
            (loss / accumulation).backward()
            loss_sum += float(loss.detach()) / accumulation
        optimizer.step()
        callback(step + 1, loss_sum)
    return steps


def validate_resume_metadata(state, *, initial_sha, arm, seed, budget, pool_sha, protocol_sha):
    expected={'initial_sha256':initial_sha,'arm':arm,'seed':seed,'budget':budget,
              'pool_sha256':pool_sha,'protocol_sha256':protocol_sha}
    if any(state.get(k)!=v for k,v in expected.items()):
        raise ValueError('resume initialization/arm/seed/budget/pool/protocol mismatch')
    if type(state.get('optimizer_updates')) is not int or not 0<=state['optimizer_updates']<=budget:
        raise ValueError('invalid resume update counter')


def preflight(package: Path, fold: Path, model: str, arm: str, checkpoint: Path, provenance: Path | None) -> dict:
    missing, issues = [], []
    config = json.loads((package / 'protocol_snapshot.json').read_text())
    if config['synthesis'].get('construction') != 'single_instance_normalize_512_then_inverse_project_all_instances_to_native_scene':
        issues.append('old native-union pool protocol superseded; reprepare v2 with shared real/synthetic native-scale preprocessing')
    config = json.loads((package / 'protocol_snapshot.json').read_text())
    if config['runtime'].get('formal_data_root_relative_to_package'):
        data = package / config['runtime']['formal_data_root_relative_to_package']
        if not (data / 'manifest.jsonl').is_file():
            missing.append(str(data / 'manifest.jsonl'))
        else:
            test_sequence = fold.name[-2:]
            test = [r for r in read_rows(data / 'manifest.jsonl') if r['dataset'] == fold.parent.name and r['sequence'] == test_sequence]
            if len(test) != 92:
                issues.append('staged held-out manifest must contain all92 native frames')
            for r in test:
                for key in ('image_path', 'mask_path'):
                    if not (data / r[key]).is_file():
                        missing.append(str(data / r[key]))
    if not checkpoint.is_file():
        missing.append(str(checkpoint))
    elif model == 'cellpose_sam':
        if sha_file(checkpoint) != config['initialization']['cellpose_sam']['expected_sha256']:
            issues.append('initial CPSAM checkpoint differs from the fixed generic cpsam_v2 SHA')
    else:
        if provenance is None or not provenance.is_file():
            missing.append('classical pretrained provenance JSON: checkpoint_sha256, training_datasets')
        else:
            p = json.loads(provenance.read_text())
            if checkpoint.is_file() and p.get('checkpoint_sha256') != sha_file(checkpoint):
                issues.append('pretrained checkpoint SHA differs from provenance')
            if not p.get('training_datasets') or set(p['training_datasets']) & set(config['datasets']):
                issues.append('common classical initialization needs documented source outside the two target CTC datasets')
    if arm != 'pretrained_only':
        pool = read_rows(fold / f'{arm}_pool.jsonl')
        for r in pool:
            for key in ('image_path', 'mask_path'):
                if not (package / r[key]).is_file():
                    missing.append(r[key])
        if arm in ('synthetic', 'mixed'):
            receipt = fold / 'generation_validation/generated_pair_receipt.json'
            if not receipt.is_file():
                missing.append(str(receipt))
            else:
                generated = json.loads(receipt.read_text())
                if generated.get('status') != 'generated_pair_structure_verified':
                    issues.append('new P14 generation-pair receipt is not valid')
                if config['synthesis'].get('full_scene_assembly_required'):
                    expected = read_rows(fold / 'generation_requests.jsonl')
                    records = {r['condition_id']: r for r in generated.get('records', [])}
                    if (generated.get('all_positive_ST_IDs_generated') is not True
                            or generated.get('recipe') != config['synthesis']['formal_cache_recipe']
                            or set(records) != {r['condition_id'] for r in expected}):
                        issues.append('full native synthetic scene coverage is incomplete')
                    else:
                        for req in expected:
                            row = records[req['condition_id']]
                            for key, digest in (('real_image_path','real_sha256'), ('synthetic_image_path','synthetic_sha256'), ('instance_mask_path','mask_sha256')):
                                path = package / req[key]
                                if not path.is_file() or sha_file(path) != row[digest]:
                                    issues.append(f'assembled scene checksum mismatch: {req["condition_id"]}/{key}')
    if model == 'cellpose_sam':
        if importlib.util.find_spec('cellpose') is None:
            missing.append('Python dependency cellpose>=4.2,<4.3 in the execution interpreter')
        else:
            version = importlib.metadata.version('cellpose')
            if tuple(int(x) for x in version.split('.')[:2]) != (4, 2):
                issues.append(f'unsupported Cellpose API version {version}')
    return {'ready': not missing and not issues, 'missing_count': len(missing),
            'missing_examples': missing[:12], 'issues': issues, 'formal_work_started': False}


def execute(a, package: Path, fold: Path, config: dict, checkpoint: Path) -> dict:
    import torch
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    if a.model != 'cellpose_sam':
        raise ValueError('This release supports the paper Cellpose-SAM experiment only.')
    exp5 = a.exp5_root.resolve()
    code_root = exp5 / 'scripts/ctc_segmentation'
    if config['runtime'].get('ctc_training_code_root_relative_to_package'):
        code_root = package / config['runtime']['ctc_training_code_root_relative_to_package']
    direction = fold.name
    dataset = fold.parent.name
    test_sequence = direction[-2:]
    data_dir = exp5 / 'data/ctc_segmentation_st_v1'
    if config['runtime'].get('formal_data_root_relative_to_package'):
        data_dir = package / config['runtime']['formal_data_root_relative_to_package']
    output = a.out_dir.resolve()
    if output.exists() and not a.resume:
        raise FileExistsError(f'preserve existing run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    train = config['training']
    initial_sha = sha_file(checkpoint)
    selected_checkpoint = checkpoint
    updates_done = 0
    if a.arm != 'pretrained_only':
        pool = read_rows(fold / f'{a.arm}_pool.jsonl')
        budget = train['optimizer_updates']
        batch = train['effective_batch']
        micro = train[a.model]['micro_batch']
        if a.model == 'cellpose_sam':
            from cellpose import models as cp_models, dynamics, train as cp_train
            cp = cp_models.CellposeModel(pretrained_model=str(checkpoint), device=device, use_bfloat16=False)
            net = cp.net.float().train()
            optimizer = torch.optim.AdamW(net.parameters(), lr=train[a.model]['learning_rate'], weight_decay=train[a.model]['weight_decay'])
        else:
            loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if loaded['model'] != a.model:
                raise ValueError('pretrained architecture mismatch')
            net = create_model(a.model, train[a.model]['base_channels'], train[a.model]['depth']).to(device).train()
            net.load_state_dict(loaded['state_dict'], strict=True)
            optimizer = torch.optim.Adam(net.parameters(), lr=train[a.model]['learning_rate'])
        start = 0
        last_path = output / 'training_state.pt'
        pool_sha = sha_file(fold / f'{a.arm}_pool.jsonl')
        protocol_sha = sha_file(package / 'protocol_snapshot.json')
        if a.resume:
            state = torch.load(last_path, map_location='cpu', weights_only=False)
            validate_resume_metadata(state,initial_sha=initial_sha,arm=a.arm,seed=a.seed,budget=budget,
                                     pool_sha=pool_sha,protocol_sha=protocol_sha)
            net.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            start = state['optimizer_updates']
            torch.set_rng_state(state['torch_rng_state'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state(state['cuda_rng_state'], device=device)
            # budget_train takes base learning rates; do not restart cosine from
            # the decayed rate loaded from the last checkpoint.
            for group in optimizer.param_groups:
                group['lr'] = train[a.model]['learning_rate']
        schedule = batch_schedule(a.arm, budget, batch, len(pool), a.seed)
        cache = {}
        def load_pair(index):
            if index not in cache:
                row = pool[index]
                p = package / row['image_path']
                raw = np.asarray(tifffile.imread(p) if p.suffix.lower() in ('.tif', '.tiff') else Image.open(p))
                if raw.ndim == 3:
                    raw = raw.astype(np.float32).mean(-1) / 255.0
                else:
                    raw = raw.astype(np.float32) / 65535.0
                label = np.asarray(tifffile.imread(package / row['mask_path'])).copy()
                cache[index] = raw, label
            return cache[index]
        def loss_for_positions(step, positions):
            images, labels = [], []
            for position in positions:
                source, index = schedule[step][position]
                assert pool[index]['source'] == source
                im, lab = load_pair(index)
                im, lab = transformed_patch(im, lab, train['patch_size'], a.seed * 1_000_000_007 + step * batch + position)
                images.append(im)
                labels.append(lab)
            if a.model == 'cellpose_sam':
                targets = []
                for lab in labels:
                    if (lab > 0).any():
                        flow = dynamics.labels_to_flows([lab.copy()], device=device)[0][1:]
                    else:
                        flow = np.zeros((3, *lab.shape), np.float32)
                    targets.append(flow)
                # Match native Cellpose's grayscale -> [gray,0,0] mapping.
                x = torch.from_numpy(np.stack([np.stack((im, im * 0, im * 0)) for im in images])).to(device)
                target = torch.from_numpy(np.stack(targets)).to(device)
                y = net(x)[0]
                if y.shape[1] != 3:
                    raise ValueError('this adapter requires the three-head generic cpsam_v2 segmentation checkpoint')
                return cp_train._loss_fn_seg(target, y, device)
            x = torch.from_numpy(np.stack(images)[:, None]).to(device)
            if a.model == 'unet':
                targets = (torch.from_numpy(np.stack([(lab > 0).astype(np.float32) for lab in labels])[:, None]).to(device),)
            else:
                targets = tuple(torch.from_numpy(np.stack([fn(lab) for lab in labels])[:, None]).to(device) for fn in (cell_distance_target, neighbor_distance_target))
            return model_loss(a.model, net(x), targets)
        def callback(step, loss):
            print(json.dumps({'optimizer_update': step, 'budget': budget, 'effective_batch': batch,
                              'arm': a.arm, 'seed': a.seed, 'loss': loss}), flush=True)
            with (output / 'train_log.jsonl').open('a') as f:
                f.write(json.dumps({'optimizer_update': step, 'loss': loss}) + '\n')
            if step % a.save_every == 0 or step == budget:
                state_tmp = output / '.training_state.tmp.pt'
                torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                            'optimizer_updates': step, 'initial_sha256': initial_sha,
                            'arm': a.arm, 'seed': a.seed, 'budget': budget,
                            'pool_sha256': pool_sha, 'protocol_sha256': protocol_sha,
                            'torch_rng_state': torch.get_rng_state(),
                            'cuda_rng_state': torch.cuda.get_rng_state(device=device) if device.type == 'cuda' else None}, state_tmp)
                state_tmp.replace(last_path)
        updates_done = budget_train(net, optimizer, budget, batch, micro, loss_for_positions, callback, start)
        if a.model == 'cellpose_sam':
            selected_checkpoint = output / 'final_cpsam_v2'
            net.save_model(str(selected_checkpoint))
        else:
            selected_checkpoint = output / 'final.pt'
            torch.save({'model': a.model, 'base_channels': train[a.model]['base_channels'],
                        'depth': train[a.model]['depth'], 'state_dict': net.state_dict()}, selected_checkpoint)
        del net, optimizer
        if a.model == 'cellpose_sam':
            del cp
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    pred_dir = output / 'prediction'
    if not (pred_dir / 'prediction_manifest.jsonl').is_file():
        if config['synthesis'].get('full_scene_assembly_required'):
            command = [sys.executable,'-u',str(ROOT/'scripts/segmentation_native_predict.py'),
                       '--data-dir',str(data_dir),'--test-pairs',str(fold/'test_pairs.jsonl'),
                       '--checkpoint',str(selected_checkpoint),'--out-dir',str(pred_dir),'--device',a.device]
        else:
            command = [sys.executable, '-u', str(exp5 / 'scripts/ctc_segmentation/predict.py'),
                       '--data_dir', str(data_dir), '--dataset', dataset,
                       '--test_sequence', test_sequence, '--model', a.model,
                       '--checkpoint', str(selected_checkpoint), '--out_dir', str(pred_dir), '--device', a.device]
        print('native full-sequence prediction', json.dumps(command), flush=True)
        subprocess.run(command, check=True)
    metrics_path = output / 'p14_ap_dice/metrics.json'
    if not metrics_path.exists():
        truth_path = fold / 'test_pairs.jsonl'
        if config['runtime'].get('formal_data_root_relative_to_package'):
            truth = read_rows(truth_path)
            by_id = {r['frame_id']: r for r in read_rows(data_dir / 'manifest.jsonl')}
            for r in truth:
                actual = by_id[r['frame_id']]
                if actual['sequence'] != test_sequence:
                    raise ValueError('held-out test sequence relocation mismatch')
                r['mask_path'] = str(data_dir / actual['mask_path'])
            truth_path = output / 'test_pairs_runtime_paths.jsonl'
            truth_path.write_text(''.join(json.dumps(r) + '\n' for r in truth))
        evaluate_manifest(pred_dir / 'prediction_manifest.jsonl', truth_path, metrics_path.parent)
    result = {'status': 'completed', 'model': a.model, 'arm': a.arm, 'seed': a.seed,
              'initial_sha256': initial_sha, 'optimizer_updates': updates_done,
              'final_checkpoint': str(selected_checkpoint), 'metrics': str(metrics_path),
              'protocol_sha256': sha_file(package / 'protocol_snapshot.json'),
              'pool_sha256': None if a.arm == 'pretrained_only' else sha_file(fold / f'{a.arm}_pool.jsonl'),
              'effective_batch': 0 if a.arm == 'pretrained_only' else train['effective_batch'],
              'prediction_image_eval_calls_actual': json.loads((pred_dir/'gpu_receipt.json').read_text()).get('actual_image_eval_calls') if (pred_dir/'gpu_receipt.json').is_file() else None,
              'test_used_for_selection': False, 'P10_called': False, 'GPU_idle_probe': False}
    write_json(output / 'run_receipt.json', result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--fold', required=True)
    p.add_argument('--model', choices=('cellpose_sam',), default='cellpose_sam')
    p.add_argument('--arm', choices=('pretrained_only', 'real', 'synthetic', 'mixed'), required=True)
    p.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    p.add_argument('--parent-root', '--exp5-root', dest='exp5_root', type=Path, default=ROOT)
    p.add_argument('--initial-checkpoint', type=Path)
    p.add_argument('--initial-provenance-json', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--execute', action='store_true', help='explicit user foreground execution; absent=dependency-only dry run')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--save-every', type=int, default=250)
    a = p.parse_args()
    package = a.package.resolve()
    fold = (package / 'folds' / a.fold).resolve()
    if not fold.is_relative_to((package / 'folds').resolve()):
        raise ValueError('fold escapes package')
    config = json.loads((package / 'protocol_snapshot.json').read_text())
    if a.model != 'cellpose_sam' and a.initial_checkpoint is None:
        p.error('U-Net/Scherr require an explicitly bound common pretrained checkpoint')
    checkpoint = (a.initial_checkpoint or a.exp5_root / config['initialization']['cellpose_sam']['path_relative_to_exp5']).resolve()
    report = preflight(package, fold, a.model, a.arm, checkpoint, a.initial_provenance_json)
    report.update({'fold': a.fold, 'model': a.model, 'arm': a.arm, 'seed': a.seed,
                   'optimizer_updates': 0 if a.arm == 'pretrained_only' else config['training']['optimizer_updates'],
                   'effective_batch': 0 if a.arm == 'pretrained_only' else config['training']['effective_batch']})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if a.execute:
        if not report['ready']:
            raise SystemExit('missing explicit prerequisites; no training/inference started')
        if a.save_every <= 0:
            raise ValueError('save-every must be positive')
        print(json.dumps(execute(a, package, fold, config, checkpoint), indent=2), flush=True)


if __name__ == '__main__':
    main()
