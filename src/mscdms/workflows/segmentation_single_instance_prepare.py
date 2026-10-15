#!/usr/bin/env python3
"""Segmentation single instance prepare."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from mscdms.workflows.segmentation_prepare import ROOT, pixel_sha, read_rows, write_json, write_rows
sys.path.insert(0, str(ROOT))
from mscdms.instance_isolation import (
    IsolationConfig, square_bounds, isolate_instance, _safe_background_mask,
    _dark_background_pool, _matched_background,
)


def rng_seed(key):
    return int(hashlib.sha256(f'P14-single-instance-v2:20261009:{key}'.encode()).hexdigest()[:16], 16) % (2**63-1)


def normalized_rgb(image):
    im = image.astype(np.float32)
    lo, hi = float(im.min()), float(im.max())
    im = (im-lo)/(hi-lo) if hi > lo else np.zeros_like(im)
    return np.repeat(np.rint(im*255).astype(np.uint8)[..., None], 3, axis=2)


def geometry(labels, instance_id):
    ys, xs = np.nonzero(labels == instance_id)
    if not len(ys):
        raise ValueError('instance absent from its training parent')
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
    x0, y0, x1, y1 = square_bounds(bbox, labels.shape[1], labels.shape[0], .30)
    h, w = y1-y0, x1-x0
    side = max(h, w)
    return {'bbox_xywh': bbox, 'bounds_xyxy': [x0, y0, x1, y1], 'square_side_native': side,
            'canvas_offset_xy': [(side-w)//2, (side-h)//2], 'renderer_scale': 512/side,
            'target_area_native': len(ys)}


def background_pool(rgb, labels):
    safe, fallback = _safe_background_mask(labels, 2, 128)
    if not (labels == 0).any():
        raise ValueError('no unlabeled training-side background pixels')
    pixels, dark_fallback = _dark_background_pool(rgb[safe], intensity_quantile=.5, min_pixels=128)
    return pixels, {'safe_background_fallback': fallback, 'dark_background_fallback': dark_fallback}


def normalized_support(labels, instance_id, g):
    x0, y0, x1, y1 = g['bounds_xyxy']
    ox, oy = g['canvas_offset_xy']
    canvas = np.zeros((g['square_side_native'], g['square_side_native']), np.uint8)
    canvas[oy:oy+y1-y0, ox:ox+x1-x0] = (labels[y0:y1, x0:x1] == instance_id).astype(np.uint8)*255
    return np.asarray(Image.fromarray(canvas).resize((512, 512), Image.Resampling.NEAREST))


def normalize_instance(rgb, labels, instance_id, seed, pool):
    """Same pixel construction as existing isolate_instance, cached parent pool."""
    g = geometry(labels, instance_id)
    x0, y0, x1, y1 = g['bounds_xyxy']
    ox, oy = g['canvas_offset_xy']
    side, h, w = g['square_side_native'], y1-y0, x1-x0
    mask = np.zeros((side, side), bool)
    mask[oy:oy+h, ox:ox+w] = labels[y0:y1, x0:x1] == instance_id
    image = _matched_background(side, side, pool, np.random.default_rng(seed), .65, .15)
    window = image[oy:oy+h, ox:ox+w]
    window[mask[oy:oy+h, ox:ox+w]] = rgb[y0:y1, x0:x1][labels[y0:y1, x0:x1] == instance_id]
    resized = np.asarray(Image.fromarray(image).resize((512, 512), Image.Resampling.BICUBIC))
    support = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((512, 512), Image.Resampling.NEAREST))
    return resized, support, g


def inverse_project(image512, g):
    if image512.shape != (512, 512, 3) or image512.dtype != np.uint8:
        raise ValueError('renderer output must be 512 RGB uint8')
    side = g['square_side_native']
    square = np.asarray(Image.fromarray(image512).resize((side, side), Image.Resampling.BICUBIC))
    ox, oy = g['canvas_offset_xy']
    x0, y0, x1, y1 = g['bounds_xyxy']
    return square[oy:oy+y1-y0, ox:ox+x1-x0]


def paste_instance(destination, parent_labels, native_request, instance_id, projected, g):
    """Native integer IDs, not inverse-interpolated binary masks, own alpha."""
    y, x, n = (native_request[k] for k in ('crop_top', 'crop_left', 'crop_size'))
    gx0, gy0, gx1, gy1 = g['bounds_xyxy']
    if projected.shape != (gy1-gy0, gx1-gx0, 3):
        raise ValueError('inverse projection bounds/axes mismatch')
    x0, y0, x1, y1 = max(x, gx0), max(y, gy0), min(x+n, gx1), min(y+n, gy1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError('instance does not overlap its native crop')
    alpha = parent_labels[y0:y1, x0:x1] == instance_id
    patch = destination[y0-y:y1-y, x0-x:x1-x]
    patch[alpha] = projected[y0-gy0:y1-gy0, x0-gx0:x1-gx0][alpha]


def regression_checks():
    labels = np.zeros((27, 43), np.uint16)
    labels[1:7, 0:4] = 9  # frame edge, asymmetric padding
    labels[15:20, 21:30] = 103
    labels[17, 25] = 0  # hole
    image = normalized_rgb(np.arange(labels.size, dtype=np.uint16).reshape(labels.shape))
    pool, _ = background_pool(image, labels)
    for ident in (9, 103):
        seed = rng_seed(str(ident))
        normal, support, g = normalize_instance(image, labels, ident, seed, pool)
        existing, existing_mask, _ = isolate_instance(image, labels, ident, g['bbox_xywh'], config=IsolationConfig(), seed=seed)
        assert np.array_equal(normal, np.asarray(existing)) and np.array_equal(support, np.asarray(existing_mask))
        req = {'crop_top': 0, 'crop_left': 0, 'crop_size': 27}
        if ident == 103:
            req['crop_left'] = 16
        original = image[req['crop_top']:req['crop_top']+27, req['crop_left']:req['crop_left']+27].copy()
        changed = original.copy()
        paint = np.full((512, 512, 3), [255, 0, 17], np.uint8)
        projected = inverse_project(paint, g)
        paste_instance(changed, labels, req, ident, projected, g)
        cropped = labels[req['crop_top']:req['crop_top']+27, req['crop_left']:req['crop_left']+27]
        assert np.array_equal(changed[cropped != ident], original[cropped != ident])
        assert np.all(changed[cropped == ident] == [255, 0, 17])
        bad_projected = projected.transpose(1, 0, 2) if projected.shape[0] != projected.shape[1] else projected[:-1]
        try:
            paste_instance(changed, labels, req, ident, bad_projected, g)
        except ValueError:
            pass
        else:
            raise AssertionError('transposed projection not rejected')
    return {'pass': True, 'matches_existing_isolate_instance_pixels_and_support': True,
            'frame_edge_and_asymmetric_padding': True, 'noncontiguous_integer_IDs_hole_and_neighbor_preservation': True,
            'inverse_coordinates_and_transpose_rejection': True, 'fixture_pixels_are_not_generated_images': True}


def color_label(labels, target):
    ids = np.unique(labels)
    rgb = np.zeros((*labels.shape, 3), np.uint8)
    for ident in ids[ids > 0]:
        color = [255, 75, 75] if ident == target else [70+(int(ident)*31)%170, 70+(int(ident)*61)%170, 70+(int(ident)*97)%170]
        rgb[labels == ident] = color
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-package', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    a = parser.parse_args()
    source, out = a.source_package.resolve(), a.out_dir.resolve()
    if out.exists():
        raise FileExistsError('preserve prior preparation')
    out.mkdir(parents=True)
    config = json.loads(a.config.read_text())
    write_json(out/'protocol_snapshot.json', config)
    checks = regression_checks()
    all_stats, previews, preview_tiles = [], [], []
    for old_fold in sorted((source/'folds/Fluo-N2DL-HeLa').iterdir()):
        fold_name = str(old_fold.relative_to(source/'folds'))
        fold = out/'folds'/fold_name
        requests = read_rows(old_fold/'generation_requests.jsonl')
        parents = {r['frame_id']: r for r in read_rows(old_fold/'train_parent_pairs.jsonl')}
        write_rows(fold/'train_parent_pairs.jsonl', list(parents.values()))
        write_rows(fold/'test_pairs.jsonl', read_rows(old_fold/'test_pairs.jsonl'))
        instance_requests, native_requests, area_lookup, cached_labels = {}, [], {}, {}
        for req in requests:
            pid = req['parent_frame_id']
            if pid not in cached_labels:
                labels = tifffile.imread(req['parent_mask_path'])
                if pixel_sha(labels) != req['parent_mask_sha256']:
                    raise ValueError('frozen training mask changed')
                cached_labels[pid] = labels
            labels = cached_labels[pid]
            y, x, n = req['crop_top'], req['crop_left'], req['crop_size']
            crop = labels[y:y+n, x:x+n]
            ids = [int(i) for i in np.unique(crop) if i > 0]
            refs = []
            for ident in ids:
                key = f'{parents[pid]["sample_id"]}__instance{ident:05d}'
                refs.append(key)
                if key not in instance_requests:
                    g = geometry(labels, ident)
                    latent_pixels = int((normalized_support(labels, ident, g)[::8, ::8] > 0).sum())
                    area_lookup[key] = g['target_area_native']
                    instance_requests[key] = {'instance_request_id': key, 'dataset': req['dataset'], 'train_sequence': req['train_sequence'],
                        'test_sequence': req['test_sequence'], 'parent_frame_id': pid, 'parent_image_path': req['parent_image_path'],
                        'parent_mask_path': req['parent_mask_path'], 'parent_image_sha256': req['parent_image_sha256'],
                        'parent_mask_sha256': req['parent_mask_sha256'], 'target_instance_id': ident, **g,
                        'generator_source': 'dynamicnuclearnet', 'generator_state_key': 'dynamicnuclearnet::hela',
                        'generator_seed': rng_seed(key), 'renderer_shape': [512, 512],
                        'source_role': 'training_side_only', 'latent64_target_pixels_CPU': latent_pixels,
                        'target_represented_after512_to64_CPU': latent_pixels > 0,
                        'support_path': f'single_conditions/{fold_name}/{key}/support.png',
                        'background_path': f'single_conditions/{fold_name}/{key}/background.png',
                        'normalized_real_path': f'single_conditions/{fold_name}/{key}/real.png',
                        'normalized_integer_label_path': f'single_conditions/{fold_name}/{key}/instances.tif',
                        'generated_image_path': f'generated_single/{fold_name}/{key}.png',
                        'target_sequence_input_used': False, 'pair_status': 'planned_single_instance_not_generated'}
            native_requests.append({**req, 'instance_render_refs': refs, 'renderer_receives_native_union': False,
                'real_image_path': f'real_roundtrip/{fold_name}/{req["condition_id"]}.tif',
                'instance_mask_path': f'native_labels/{fold_name}/{req["condition_id"]}.tif',
                'synthetic_image_path': f'generated_native/{fold_name}/{req["condition_id"]}.png',
                'pair_status': 'planned_native_scene_all_labeled_instances_rendered'})
        write_rows(fold/'generation_requests.jsonl', native_requests)
        write_rows(fold/'single_instance_requests.jsonl', instance_requests.values())
        pools = {arm: [] for arm in ('real', 'synthetic', 'mixed')}
        for idx, req in enumerate(native_requests):
            for arm in pools:
                role = ('real' if idx % 4 < 2 else 'synthetic') if arm == 'mixed' else arm
                pools[arm].append({'sample_id': req['condition_id'], 'source': role,
                    'image_path': req['real_image_path'] if role == 'real' else req['synthetic_image_path'],
                    'mask_path': req['instance_mask_path'], 'parent_frame_id': req['parent_frame_id'],
                    'train_sequence': req['train_sequence'], 'test_sequence': req['test_sequence']})
        for arm, pool in pools.items():
            write_rows(fold/f'{arm}_pool.jsonl', pool)
        scales = [r['renderer_scale'] for r in instance_requests.values()]
        all_stats.append({'fold': fold_name, 'native_training_scenes': len(native_requests), 'unique_single_instance_render_requests': len(instance_requests),
                          'instance_appearances_in_scenes': sum(len(r['instance_render_refs']) for r in native_requests),
                          'renderer_scale_min_median_max': [min(scales), float(np.median(scales)), max(scales)],
                          'single_instance_conditions_without_target_at_latent64_CPU': sum(not r['target_represented_after512_to64_CPU'] for r in instance_requests.values()),
                          'downstream_native_geometry_scale': 1.0, 'mixed_scenes_total': len(pools['mixed'])})
        # Four representative scenes/anchors per training side, selected using
        # labels only. Native output examples retain ALL neighboring labels.
        density = sorted(native_requests, key=lambda r: (len(r['instance_render_refs']), r['condition_id']))
        pick_indices = (0, len(density)//3, 2*len(density)//3, len(density)-1)
        for number, rank in enumerate(pick_indices):
            req = density[rank]
            candidates = [instance_requests[k] for k in req['instance_render_refs']]
            candidates.sort(key=lambda r: (r['target_area_native'], r['instance_request_id']))
            chosen = candidates[0] if number == 0 else candidates[-1] if number == 1 else candidates[len(candidates)//2]
            pid = req['parent_frame_id']
            labels = cached_labels[pid]
            original = tifffile.imread(req['parent_image_path'])
            if pixel_sha(original) != req['parent_image_sha256']:
                raise ValueError('frozen training image changed')
            rgb = normalized_rgb(original)
            pool, bg_diag = background_pool(rgb, labels)
            y, x, n = req['crop_top'], req['crop_left'], req['crop_size']
            cropped_labels = labels[y:y+n, x:x+n].copy()
            original_crop = rgb[y:y+n, x:x+n].copy()
            real_roundtrip = original_crop.copy()
            for r in candidates:
                normal, support, g = normalize_instance(rgb, labels, r['target_instance_id'], r['generator_seed'], pool)
                projected = inverse_project(normal, g)
                paste_instance(real_roundtrip, labels, req, r['target_instance_id'], projected, g)
                if r['instance_request_id'] == chosen['instance_request_id']:
                    target_normal, target_support, target_g = normal, support, g
                    existing, old_support, _ = isolate_instance(rgb, labels, r['target_instance_id'], g['bbox_xywh'], config=IsolationConfig(), seed=r['generator_seed'])
                    assert np.array_equal(normal, np.asarray(existing)) and np.array_equal(support, np.asarray(old_support))
            assert np.array_equal(real_roundtrip[cropped_labels == 0], original_crop[cropped_labels == 0])
            assert np.array_equal(cropped_labels, labels[y:y+n, x:x+n])
            latent_support = target_support[::8, ::8] > 0
            assert latent_support.any()
            condition_dir = out/Path(chosen['support_path']).parent
            condition_dir.mkdir(parents=True)
            Image.fromarray(target_normal).save(out/chosen['normalized_real_path'])
            Image.fromarray(target_support).save(out/chosen['support_path'])
            bg = target_normal.copy(); bg[target_support > 0] = 0
            Image.fromarray(bg).save(out/chosen['background_path'])
            tifffile.imwrite(out/chosen['normalized_integer_label_path'], (target_support > 0).astype(labels.dtype)*chosen['target_instance_id'], compression='zlib')
            (out/req['real_image_path']).parent.mkdir(parents=True, exist_ok=True)
            (out/req['instance_mask_path']).parent.mkdir(parents=True, exist_ok=True)
            gray = np.rint(real_roundtrip.astype(np.float32).mean(-1)*257).astype(np.uint16)
            tifffile.imwrite(out/req['real_image_path'], gray, compression='zlib')
            tifffile.imwrite(out/req['instance_mask_path'], cropped_labels, compression='zlib')
            preview_dir = out/'previews'/old_fold.name/req['condition_id']; preview_dir.mkdir(parents=True)
            Image.fromarray(original_crop).save(preview_dir/'original_native.png')
            Image.fromarray(real_roundtrip).save(preview_dir/'real_roundtrip_native.png')
            Image.fromarray(color_label(cropped_labels, chosen['target_instance_id'])).save(preview_dir/'all_integer_labels.png')
            previews.append({'condition_id': req['condition_id'], 'train_sequence': req['train_sequence'], 'target_instance_id': chosen['target_instance_id'],
                'instance_request_id': chosen['instance_request_id'], 'all_neighbor_IDs_retained': len(candidates), **target_g, **bg_diag,
                'normalized_single_instance_label_IDs': [0, chosen['target_instance_id']],
                'latent64_target_present': bool(latent_support.any()), 'latent64_target_pixels': int(latent_support.sum()),
                'native_crop_label_sha256': pixel_sha(cropped_labels),
                'real_roundtrip_foreground_MAE_8bit': float(np.abs(real_roundtrip.astype(float)-original_crop)[cropped_labels > 0].mean()),
                'native_background_unchanged': True, 'actual_synthetic_pair_created': False})
            preview_tiles.append((previews[-1], [original_crop, target_normal, np.repeat(target_support[..., None], 3, 2), real_roundtrip,
                                                color_label(cropped_labels, chosen['target_instance_id'])]))
            print(f'CPU native reconstruction {old_fold.name} example{number+1}/4: {len(candidates)} individually normalized objects', flush=True)
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    font = ImageFont.truetype(font_path, 17) if Path(font_path).is_file() else ImageFont.load_default()
    board = Image.new('RGB', (5*256, 72+len(preview_tiles)*294), 'white')
    draw = ImageDraw.Draw(board)
    draw.text((8, 5), 'CPU ONLY: REAL CONTROLS + CONDITIONS / NO SYNTHETIC OUTPUT', fill='black', font=font)
    titles = ['Original native context', 'Single real/control, 512', 'One-instance support, 512', 'All real -> 512 -> native', 'Native integer labels']
    for col, title in enumerate(titles): draw.text((col*256+5, 39), title, fill='black', font=font)
    for row, (meta, tiles) in enumerate(preview_tiles):
        top = 72+row*294
        text = f'Train{meta["train_sequence"]}  target ID {meta["target_instance_id"]}  render x{meta["renderer_scale"]:.2f} -> native x1  all {meta["all_neighbor_IDs_retained"]} IDs retained'
        draw.text((5, top), text, fill='black', font=font)
        for col, tile in enumerate(tiles):
            board.paste(Image.fromarray(tile).resize((256, 256), Image.Resampling.NEAREST if col in (2, 4) else Image.Resampling.BILINEAR), (col*256, top+28))
    board.save(out/'previews/single_instance_to_native_contact_sheet.png')
    write_rows(out/'preview_manifest.jsonl', previews)
    result = {'status': 'CPU_single_instance_native_scale_preparation_not_generated', 'folds': all_stats, 'CPU_regression_checks': checks,
              'representative_single_conditions_materialized': len(previews), 'native_all_neighbor_real_roundtrip_examples': len(previews),
              'actual_synthetic_outputs': 0, 'old_32_image_GPU_proposal': 'WITHDRAWN', 'formal_GPU_training': False,
              'downstream_training_and_full_FOV_test_scale': 'native x1 in every arm; generation-only normalization inversely projected',
              'source_package': str(source)}
    write_json(out/'preparation_receipt.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
