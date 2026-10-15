#!/usr/bin/env python3
"""Segmentation pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from mscdms.workflows.segmentation_prepare import pixel_sha, read_rows, sha_file, write_rows, write_json


def validate_request(r: dict, package: Path, parents: dict[str, dict]) -> None:
    parent = parents.get(r['parent_frame_id'])
    if parent is None:
        raise ValueError('parent not in the frozen training-side manifest')
    if parent['dataset'] != r['dataset'] or parent['sequence'] != r['train_sequence']:
        raise ValueError('wrong dataset or target sequence in generation condition')
    if r['train_sequence'] == r['test_sequence'] or r.get('target_sequence_input_used') is not False:
        raise ValueError('target-sequence input forbidden')
    if r.get('source_role') != 'training_side_only':
        raise ValueError('wrong source role')
    for key in ('image', 'mask'):
        if r[f'parent_{key}_path'] != parent[f'{key}_path'] or r[f'parent_{key}_sha256'] != parent[f'{key}_pixel_sha256']:
            raise ValueError(f'parent {key} lineage mismatch')
    y, x, n = (r[k] for k in ('crop_top', 'crop_left', 'crop_size'))
    if any(type(v) is not int for v in (y, x, n)) or n <= 0:
        raise ValueError('crop coordinates/size must be integers with positive size')
    if not (0 <= y <= parent['height']-n and 0 <= x <= parent['width']-n):
        raise ValueError('crop escapes native parent geometry')
    for key in ('real_image_path', 'instance_mask_path', 'support_path', 'background_path', 'synthetic_image_path'):
        p = package / r[key]
        if not p.resolve().is_relative_to(package.resolve()):
            raise ValueError('artifact path escapes the preparation package')
    if not r['synthetic_image_path'].startswith('generated/'):
        raise ValueError('synthetic image must belong to the new P14 generation pool')


def materialize(package: Path, fold: Path, output: Path, limit: int, resume: bool = False) -> dict:
    requests = read_rows(fold / 'generation_requests.jsonl')
    parents = {r['frame_id']: r for r in read_rows(fold / 'train_parent_pairs.jsonl')}
    cache, records = {}, []
    for r in requests[:limit or None]:
        validate_request(r, package, parents)
        pid = r['parent_frame_id']
        if pid not in cache:
            image = np.asarray(tifffile.imread(r['parent_image_path']))
            labels = np.asarray(tifffile.imread(r['parent_mask_path']))
            if pixel_sha(image) != r['parent_image_sha256'] or pixel_sha(labels) != r['parent_mask_sha256']:
                raise ValueError('native parent pixels changed since audit')
            image = image.astype(np.float32)
            lo, hi = float(image.min()), float(image.max())
            image = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)
            cache[pid] = image, labels
        image, labels = cache[pid]
        y, x, n = r['crop_top'], r['crop_left'], r['crop_size']
        crop = image[y:y+n, x:x+n]
        instances = labels[y:y+n, x:x+n]
        if crop.shape != (n, n) or instances.shape != (n, n):
            raise ValueError('invalid native crop, no resize/padding is permitted here')
        support = instances > 0
        gray = np.rint(crop * 65535).astype(np.uint16)
        rgb = np.repeat(np.rint(crop * 255).astype(np.uint8)[..., None], 3, axis=2)
        rgb[support] = 0
        files = {k: package / r[k] for k in ('real_image_path', 'instance_mask_path', 'support_path', 'background_path')}
        if any(p.exists() for p in files.values()):
            if not resume or not all(p.exists() for p in files.values()):
                raise FileExistsError(f'preserve materialized condition: {r["condition_id"]}')
            existing = (np.asarray(tifffile.imread(files['real_image_path'])),
                        np.asarray(tifffile.imread(files['instance_mask_path'])),
                        np.asarray(Image.open(files['support_path'])),
                        np.asarray(Image.open(files['background_path'])))
            if not all(np.array_equal(a, b) for a, b in zip(existing, (gray, instances, support.astype(np.uint8)*255, rgb))):
                raise ValueError('resume condition pixels differ from frozen training-side crop')
        else:
            for p in files.values():
                p.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(files['real_image_path'], gray, compression='zlib')
            tifffile.imwrite(files['instance_mask_path'], instances, compression='zlib')
            Image.fromarray(support.astype(np.uint8) * 255).save(files['support_path'])
            Image.fromarray(rgb).save(files['background_path'])
        record = {**r, 'pair_status': 'real_condition_materialized_synthetic_not_generated',
                  'instance_pixel_sha256': pixel_sha(instances),
                  'request_sha256': __import__('hashlib').sha256(json.dumps(r, sort_keys=True).encode()).hexdigest(),
                  'artifact_sha256': {k: sha_file(p) for k, p in files.items()},
                  'cropped_instance_count': int(np.count_nonzero(np.unique(instances) > 0)),
                  'native_instance_crop_retained': True, 'background_zero_inside_support': True}
        records.append(record)
        if len(records) % 32 == 0:
            print(f'CPU condition materialization {len(records)}/{len(requests[:limit or None])}', flush=True)
    write_rows(output / 'materialized_conditions.jsonl', records)
    result = {'status': 'CPU_real_conditions_materialized', 'records': len(records),
              'synthetic_images_created': 0, 'integer_labels_preserved': True,
              'target_sequence_inputs_used': False, 'real_uint16_is_normalized_derived_crop': True}
    write_json(output / 'materialization_receipt.json', result)
    return result


def verify_generated(package: Path, fold: Path, manifest: Path, output: Path) -> dict:
    """The generator adapter must emit input and checkpoint hashes in each row."""
    requests = {r['condition_id']: r for r in read_rows(fold / 'generation_requests.jsonl')}
    parents = {r['frame_id']: r for r in read_rows(fold / 'train_parent_pairs.jsonl')}
    records = read_rows(manifest)
    if len(records) != len(requests) or {r['condition_id'] for r in records} != set(requests):
        raise ValueError('generated pool must cover every frozen request exactly once')
    verified, checkpoint_hashes = [], set()
    for r in records:
        req = requests[r['condition_id']]
        validate_request(req, package, parents)
        image_path = (package / req['synthetic_image_path']).resolve()
        mask_path = (package / req['instance_mask_path']).resolve()
        if Path(r['image_path']).resolve() != image_path or Path(r['mask_path']).resolve() != mask_path:
            raise ValueError('generated paths do not match this new P14 condition pool')
        if r.get('parent_frame_id') != req['parent_frame_id'] or r.get('train_sequence') != req['train_sequence']:
            raise ValueError('generated pair lacks correct training-side lineage')
        checkpoint_sha = r.get('generator_checkpoint_sha256', '')
        if len(checkpoint_sha) != 64 or r.get('target_sequence_input_used') is not False:
            raise ValueError('missing checkpoint hash or forbidden target input')
        checkpoint_hashes.add(checkpoint_sha)
        if r.get('generator_state_key') != req['generator_state_key'] or req['generator_state_key'] is None:
            raise ValueError('generator state mapping is absent or differs from the frozen request')
        request_sha = __import__('hashlib').sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
        if r.get('request_sha256') != request_sha:
            raise ValueError('generation request was changed or unbound')
        labels = np.asarray(tifffile.imread(mask_path))
        image = np.asarray(Image.open(image_path))
        n = req['crop_size']
        if image.shape[:2] != labels.shape or labels.shape != (n, n) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('invalid generated image/label dimensions or label dtype')
        parent_mask = np.asarray(tifffile.imread(req['parent_mask_path']))
        if pixel_sha(parent_mask) != req['parent_mask_sha256']:
            raise ValueError('parent mask changed since audit')
        y, x = req['crop_top'], req['crop_left']
        if not np.array_equal(labels, parent_mask[y:y+n, x:x+n]):
            raise ValueError('instance labels differ from the exact training-side ST crop')
        if r.get('input_instance_pixel_sha256') != pixel_sha(labels):
            raise ValueError('generated instance label hash mismatch')
        if r.get('image_sha256') != sha_file(image_path):
            raise ValueError('generated image checksum mismatch')
        verified.append({**r, 'pair_status': 'verified_file_dimensions_hashes_and_lineage',
                         'appearance_label_correspondence_manually_verified': False})
    if len(checkpoint_hashes) != 1:
        raise ValueError('one shared frozen generator is required per pool')
    write_rows(output / 'verified_generated_pairs.jsonl', verified)
    result = {'status': 'generated_pair_structure_verified', 'pairs': len(verified),
              'appearance_label_correspondence_verified': False,
              'note': 'Hard support and lineage do not establish that every labeled instance is visible; inspect fixed train-side examples before freezing the adapter.'}
    write_json(output / 'generated_pair_receipt.json', result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('materialize', 'verify-generated'))
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--fold', required=True, help='e.g. Fluo-N2DL-HeLa/train01_test02')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='materialization only, 0=all; never marks full pool complete')
    p.add_argument('--generated-manifest', type=Path)
    p.add_argument('--resume', action='store_true', help='verify existing materialized pixels and fill missing conditions; output receipt must still be fresh')
    a = p.parse_args()
    package = a.package.resolve()
    fold = (package / 'folds' / a.fold).resolve()
    if not fold.is_relative_to((package / 'folds').resolve()):
        raise ValueError('invalid fold')
    output = a.out_dir.resolve()
    if not output.is_relative_to(package):
        raise ValueError('pair receipts must stay inside the new P14 preparation package')
    if output.exists() and not (a.mode == 'materialize' and a.resume):
        raise FileExistsError(f'preserve existing output: {output}')
    if a.limit < 0:
        raise ValueError('limit must be nonnegative')
    if a.mode == 'verify-generated' and a.generated_manifest is None:
        p.error('--generated-manifest required')
    output.mkdir(parents=True, exist_ok=a.mode == 'materialize' and a.resume)
    result = materialize(package, fold, output, a.limit, a.resume) if a.mode == 'materialize' else verify_generated(package, fold, a.generated_manifest, output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


