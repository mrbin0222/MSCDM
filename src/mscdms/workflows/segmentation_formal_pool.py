#!/usr/bin/env python3
"""Segmentation formal pool."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from mscdms.workflows.segmentation_prepare import ROOT, pixel_sha, read_rows, sha_file, write_json, write_rows
from mscdms.workflows.segmentation_single_instance_prepare import normalized_rgb, normalize_instance, background_pool
from mscdms.workflows.segmentation_photometry import (fit_renderer_photometry, renderer_tensor_rgb,
    compose_native512, inverse_project_float, paste_native_float, gray_uint16)

GEN_SHA = '71d6534ebe81e8e7b6f0a45337c59c11b68c620c2135eabbee2e926fac521755'
VAE_SHA = '687522faf7077a1a825a74d117a8d767c67bf41eeb1ec085fc8f86bbc45705cc'
RECIPE = 'p14_full_ST_single_instance_p1p99_float_inverse_v3'


def cache_identity(r):
    keys = ('dataset', 'train_sequence', 'parent_frame_id', 'parent_image_sha256',
            'parent_mask_sha256', 'target_instance_id', 'bounds_xyxy',
            'square_side_native', 'canvas_offset_xy', 'generator_seed',
            'generator_source', 'generator_state_key')
    identity = {k: r[k] for k in keys}
    identity.update(recipe=RECIPE, generator_sha256=GEN_SHA, VAE_sha256=VAE_SHA,
                    DDIM_steps=25, precision='cuda_bfloat16_torch2.11.0+cu130',
                    percentiles=[1, 99], normalization_geometry='native_parent_minmax_RGB8_isolate512')
    return identity, hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def compose_scene(original_rgb, labels, crop, projected_by_id):
    """CPU interface. No missing-ID fallback, label interpolation or FG blend."""
    n = crop['crop_size']
    if original_rgb.shape != (n, n, 3) or labels.shape != (n, n):
        raise ValueError('native scene dimensions mismatch')
    if not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError('native nonnegative integer ST required')
    required = {int(i) for i in np.unique(labels) if i > 0}
    if set(projected_by_id) != required:
        raise ValueError(f'all foreground IDs required; missing={required-set(projected_by_id)}, extra={set(projected_by_id)-required}')
    out = original_rgb.astype(np.float32).copy()
    coverage = np.zeros(labels.shape, np.uint8)
    for ident in sorted(required):
        projected, g = projected_by_id[ident]
        if not np.isfinite(projected).all() or projected.min() < 0 or projected.max() > 255:
            raise ValueError('projected RGB outside finite native domain')
        paste_native_float(out, labels, crop, ident, projected, g)
        x,y=crop['crop_left'],crop['crop_top'];gx0,gy0,gx1,gy1=g['bounds_xyxy']
        x0,y0,x1,y1=max(x,gx0),max(y,gy0),min(x+n,gx1),min(y+n,gy1)
        covered=np.zeros(labels.shape,bool)
        covered[y0-y:y1-y,x0-x:x1-x]=labels[y0-y:y1-y,x0-x:x1-x]==ident
        coverage[covered]+=1
    if not np.all(coverage[labels > 0] == 1) or not np.array_equal(out[labels == 0], original_rgb[labels == 0]):
        raise ValueError('foreground coverage or native background preservation failed')
    return out, labels.copy(), {'all_positive_IDs_rendered': sorted(required),
        'every_foreground_pixel_assigned_once': True, 'native_background_exact': True,
        'native_integer_ST_exact': True}


def plan(a):
    if a.source is None:
        raise ValueError('--source is required for plan mode')
    out, source = a.package.resolve(), a.source.resolve()
    if out.exists():
        raise FileExistsError('preserve prior plan')
    out.mkdir(parents=True)
    cfg = copy.deepcopy(json.loads((source/'protocol_snapshot.json').read_text()))
    frames, crops = a.frames_per_direction, a.crops_per_frame
    if frames < 2 or frames > 92 or crops not in (1, 4):
        raise ValueError('supported frozen selections:2..92 frames, crop00 or all4 crops')
    pool_size=frames*crops
    if pool_size%4: raise ValueError('mixed pool requires equal index-mod4 slots')
    cfg.update(schema='exp4.p14.hela_cpsam_formal_pool.v3',
               status='experiment_scope_confirmed_local_preparation_only_destination_authorization_pending' if a.scope_confirmed else 'CPU_prepared_proposal_awaiting_formal_GPU_authorization', models=['cellpose_sam'])
    cfg['training'].update(crops_per_training_frame=crops, selected_training_frames_per_direction=frames,
        real_fraction_of_available_train_frames=frames/92,
        real_normalization=RECIPE, mixed_pool_size=pool_size,
        mixed_pool_slots={'real':pool_size//2, 'synthetic':pool_size//2})
    cfg['synthesis'].update(actual_synthetic_image_label_correspondence='same8_local_retry_verified_full_scene_GPU_not_run',
        formal_cache_recipe=RECIPE, full_scene_assembly_required=True,
        foreground_contrast_condition='per-instance training-reference p1/p99 scalars explicitly used',
        cache_key_rule='parent sequence/frame/ID + pixel hashes + geometry + seed + recipe + checkpoint hashes; excludes patch/arm/downstream seed')
    cfg['runtime']['launch_formal_gpu_in_this_task'] = False
    cfg['runtime']['formal_data_root_relative_to_package'] = 'staged_hela'
    cfg['runtime']['experiment_scope_user_confirmed'] = bool(a.scope_confirmed)
    cfg['runtime']['upload_destination_authorized'] = False
    cfg['runtime']['formal_output_root'] = a.run_root
    write_json(out/'protocol_snapshot.json', cfg)
    old_batch = a.corrected_batch.resolve() if a.corrected_batch is not None else None
    old_inputs, old_outputs = {}, {}
    if old_batch is not None:
        old_inputs = {r['instance_request_id']: r for r in read_rows(old_batch/'batch_manifest.jsonl')}
        old_outputs = {r['instance_request_id']: r for r in read_rows(old_batch/'generation/manifest.jsonl')}
        old_gpu = json.loads((old_batch/'generation/gpu_receipt.json').read_text())
        assert old_gpu['generator_sha256'] == GEN_SHA and old_gpu['VAE_sha256'] == VAE_SHA
    full_counts, minimal_counts, commands, all_instances = [], [], [], []
    for src_fold in sorted((source/'folds/Fluo-N2DL-HeLa').iterdir()):
        name = str(src_fold.relative_to(source/'folds')); fold = out/'folds'/name
        fold.mkdir(parents=True)
        scenes = read_rows(src_fold/'generation_requests.jsonl')
        instances = {r['instance_request_id']: r for r in read_rows(src_fold/'single_instance_requests.jsonl')}
        counts = Counter(k for r in scenes for k in r['instance_render_refs'])
        full_counts.append({'fold':name, 'patches':len(scenes), 'unique_parent_frame_ID':len(counts),
                            'patch_instance_references':sum(counts.values()),
                            'duplicate_references_avoided':sum(counts.values())-len(counts)})
        first = sorted([r for r in scenes if r['condition_id'].endswith('__crop00')], key=lambda r:r['parent_frame_id'])
        if len(first) != 92:
            raise ValueError('frozen source must have crop00 for all92 train frames')
        selected_frames={first[round(i*91/(frames-1))]['parent_frame_id'] for i in range(frames)}
        chosen=[copy.deepcopy(r) for r in sorted(scenes,key=lambda r:r['condition_id'])
                if r['parent_frame_id'] in selected_frames and (crops==4 or r['condition_id'].endswith('__crop00'))]
        if len(chosen)!=pool_size: raise ValueError('frozen selected scene count mismatch')
        refs = sorted({k for r in chosen for k in r['instance_render_refs']})
        planned = []
        for key in refs:
            r = copy.deepcopy(instances[key]); identity, cid = cache_identity(r)
            r.update(cache_key=cid, cache_identity=identity, directory=f'cache/{cid}',
                     parent_image_path_relative_to_exp5=str(Path(r['parent_image_path']).relative_to(a.exp5_root.resolve())),
                     parent_mask_path_relative_to_exp5=str(Path(r['parent_mask_path']).relative_to(a.exp5_root.resolve())))
            if key in old_inputs:
                prev, gen = old_inputs[key], old_outputs[key]
                for k in ('generator_seed','bounds_xyxy','square_side_native','canvas_offset_xy'):
                    if prev[k] != r[k]: raise ValueError('reuse geometry/seed mismatch')
                if prev['parent_image_pixel_sha256'] != r['parent_image_sha256'] or prev['parent_mask_pixel_sha256'] != r['parent_mask_sha256']:
                    raise ValueError('reuse parent lineage mismatch')
                if sha_file(old_batch/gen['native_domain512_path']) != gen['native_domain512_sha256']:
                    raise ValueError('reuse generated output SHA mismatch')
                r['eligible_corrected_reuse'] = {'batch':str(old_batch), 'input':prev, 'output':gen,
                    'batch_relative_to_exp4':str(old_batch.relative_to(ROOT))}
            planned.append(r)
        selected_parents = {r['parent_frame_id'] for r in chosen}
        write_rows(fold/'train_parent_pairs.jsonl',[r for r in read_rows(src_fold/'train_parent_pairs.jsonl') if r['frame_id'] in selected_parents])
        shutil.copy2(src_fold/'test_pairs.jsonl',fold/'test_pairs.jsonl')
        for r in chosen:
            k=r['condition_id']; r.update(real_image_path=f'real/{name}/{k}.tif',
                synthetic_image_path=f'synthetic/{name}/{k}.tif', instance_mask_path=f'labels/{name}/{k}.tif')
        write_rows(fold/'generation_requests.jsonl',chosen)
        write_rows(fold/'single_instance_requests.jsonl',planned)
        for arm in ('real','synthetic','mixed'):
            pool=[]
            for i,r in enumerate(chosen):
                role=arm if arm!='mixed' else ('real' if i%4<2 else 'synthetic')
                pool.append({'sample_id':r['condition_id'],'source':role,
                    'image_path':r['real_image_path'] if role=='real' else r['synthetic_image_path'],
                    'mask_path':r['instance_mask_path'],'parent_frame_id':r['parent_frame_id'],
                    'train_sequence':r['train_sequence'],'test_sequence':r['test_sequence']})
            write_rows(fold/f'{arm}_pool.jsonl',pool)
        minimal_counts.append({'fold':name,'selected_frames':sorted({r['parent_frame_id'] for r in chosen}),
            'real_patches':pool_size,'synthetic_patches':pool_size,'mixed_real_patches':pool_size//2,'mixed_synthetic_patches':pool_size//2,
            'unique_instances':len(planned),'patch_instance_references':sum(len(r['instance_render_refs']) for r in chosen),
            'eligible_corrected_cache_reuse':sum('eligible_corrected_reuse' in r for r in planned)})
        all_instances += planned
        for arm in ('pretrained_only','real','synthetic','mixed'):
            for seed in ([None] if arm=='pretrained_only' else (42,43,44)):
                run=arm if seed is None else f'{arm}/seed{seed}'
                argv=['scripts/segmentation_run.py','--package',str(out), '--fold',name,
                    '--model','cellpose_sam','--arm',arm,'--parent-root',str(a.exp5_root.resolve()),
                    '--initial-checkpoint',str(a.exp5_root.resolve()/cfg['initialization']['cellpose_sam']['path_relative_to_exp5']),
                    '--out-dir',f'{a.run_root}/{name}/{run}','--execute']
                if seed is not None: argv += ['--seed',str(seed)]
                commands.append({'fold':name,'arm':arm,'seed':seed,'argv':argv})
    write_rows(out/'instance_cache_requests.jsonl',all_instances)
    counts={'old_full_pool':full_counts,'proposed_pool':minimal_counts,
        'proposed_unique_instances':len(all_instances),
        'eligible_reuse':sum('eligible_corrected_reuse' in r for r in all_instances),
        'proposed_new_generation_calls':sum('eligible_corrected_reuse' not in r for r in all_instances),
        'cache_reuse_across_arms_and_seeds':True,'same_ID_different_frames_not_merged':True}
    write_json(out/'counts.json',counts); write_json(out/'training_commands.json',commands)
    package=str(out); lines=['#!/usr/bin/env bash','set -euo pipefail',
        '# Proposed full formal action; run only after approval, from exp4.',
        'gen_python="${P14_GENERATOR_PYTHON:-.venv/bin/python}"',
        'cp_python="${P14_CELLPOSE_PYTHON:-.venv-cellpose-cpsam-v2/bin/python}"']
    for mode in ('materialize','generate','assemble'):
        extra=' --execute' if mode=='generate' else ''
        lines.append(f'"$gen_python" -u scripts/segmentation_formal_pool.py {mode} --package {shlex.quote(package)}'+extra)
    lines += ['"$cp_python" -u '+shlex.join(c['argv']) for c in commands]
    (out/'foreground_serial.sh').write_text('\n'.join(lines)+'\n')
    write_json(out/'plan_receipt.json',{'status':'CPU_plan_saved_GPU_not_started','actual_generator_calls':0,
        'actual_Cellpose_calls':0,'optimizer_updates':0,'test_inference_images':0,
        'proposed_training_runs':18,'proposed_baseline_sets':2,'budget_update_sum':36000,
        'effective_patch_presentations':288000,'proposed_full_FOV_test_image_evals':1840,
        'class_balanced_vs_random':'deferred_no_extra_groups','experiment_scope_user_confirmed':bool(a.scope_confirmed),
        'upload_destination_authorized':False,'actual_network_transfers':0})
    print(json.dumps({k:v for k,v in counts.items() if k not in ('old_full_pool','proposed_pool')},ensure_ascii=False))


def parent_paths(r, exp5, package=None):
    paths=tuple(exp5/r[f'parent_{k}_path_relative_to_exp5'] for k in ('image','mask'))
    if package is not None and (package/'staged_hela/manifest.jsonl').is_file():
        paths=tuple(package/'staged_hela'/Path(r[f'parent_{k}_path_relative_to_exp5']).relative_to('data/ctc_segmentation_st_v1') for k in ('image','mask'))
    return paths


def stage_data(a):
    """Local CPU copy for later transfer; immutable originals never changed."""
    package=a.package.resolve(); data=a.exp5_root.resolve()/'data/ctc_segmentation_st_v1'
    rows=[r for r in read_rows(data/'manifest.jsonl') if r['dataset']=='Fluo-N2DL-HeLa']
    if len(rows)!=184: raise ValueError('expected184 HeLa native frames')
    out=package/'staged_hela';out.mkdir(exist_ok=True);files={}
    for r in rows:
        for key in ('image_path','mask_path'):
            src=data/r[key];dst=out/r[key];dst.parent.mkdir(parents=True,exist_ok=True)
            digest=sha_file(src)
            if dst.exists() and sha_file(dst)!=digest: raise ValueError('staged source differs')
            if not dst.exists(): shutil.copy2(src,dst)
            files[r[key]]=digest
    write_rows(out/'manifest.jsonl',rows)
    write_json(out/'file_sha256.json',files)
    write_json(package/'staging_receipt.json',{'status':'CPU_HeLa_copies_staged','native_images':184,'ST_masks':184,
        'data_file_bytes':sum((out/k).stat().st_size for k in files),'actual_generator_calls':0,
        'actual_Cellpose_calls':0,'optimizer_updates':0,'purpose':'selected-side generation and full held-out native evaluation after approval'})


def import_corrected_cache(prepared, directory):
    if 'eligible_corrected_reuse' not in prepared or (directory/'generated.json').exists():
        return
    reuse=prepared['eligible_corrected_reuse']; old=ROOT/reuse['batch_relative_to_exp4']; prev=reuse['input']; gen=reuse['output']
    for name in ('normalized_real.png','support.png','background.png'):
        arr=np.asarray(Image.open(directory/name))
        if pixel_sha(arr)!=prev['files'][name]['pixel_sha256']: raise ValueError('corrected cache input pixels not identical')
    if prev['renderer_photometry']!=prepared['renderer_photometry'] or sha_file(old/gen['native_domain512_path'])!=gen['native_domain512_sha256']:
        raise ValueError('corrected cache mapping/output mismatch')
    tifffile.imwrite(directory/'projected_generated_float8.tif',inverse_project_float(tifffile.imread(old/gen['native_domain512_path']),prepared),photometric='rgb',compression='zlib')
    shutil.copy2(old/gen['generated_path'],directory/'generated_model_domain.png')
    write_json(directory/'generated.json',generation_record(prepared,directory,actual_call=False,reused_from=str(old)))


def materialize(a):
    package=a.package.resolve(); rows=read_rows(package/'instance_cache_requests.jsonl')
    total=len(rows)
    if a.only_corrected_cache:
        rows=[r for r in rows if 'eligible_corrected_reuse' in r]
    grouped=defaultdict(list)
    for r in rows: grouped[r['parent_frame_id']].append(r)
    done=[]
    for group in grouped.values():
        pending=[]
        for r in group:
            directory=package/r['directory']
            if (directory/'prepared.json').exists():
                previous=json.loads((directory/'prepared.json').read_text())
                validate_prepared(directory,r,previous)
                import_corrected_cache(previous,directory)
                done.append(previous)
            else:
                pending.append(r)
        if not pending:
            print(f'CPU prepared {len(done)}/{len(rows)}',flush=True)
            continue
        r=pending[0]; ip,mp=parent_paths(r,a.exp5_root.resolve(),package)
        image,labels=tifffile.imread(ip),tifffile.imread(mp)
        if pixel_sha(image)!=r['parent_image_sha256'] or pixel_sha(labels)!=r['parent_mask_sha256']:
            raise ValueError('training parent pixel SHA changed')
        rgb=normalized_rgb(image); pool,_=background_pool(rgb,labels)
        for r in pending:
            directory=package/r['directory']; directory.mkdir(parents=True,exist_ok=True)
            norm,support,g=normalize_instance(rgb,labels,r['target_instance_id'],r['generator_seed'],pool)
            for k in ('bounds_xyxy','square_side_native','canvas_offset_xy'):
                if g[k]!=r[k]: raise ValueError('native geometry changed')
            meta=fit_renderer_photometry(norm,source_role='training_side_only')
            bg=norm.copy(); bg[support>0]=0
            for name,arr in [('normalized_real.png',norm),('support.png',support),('background.png',bg)]: Image.fromarray(arr).save(directory/name)
            real512=compose_native512(renderer_tensor_rgb(norm,meta),bg,support>0,meta)
            tifffile.imwrite(directory/'projected_real_float8.tif',inverse_project_float(real512,g),photometric='rgb',compression='zlib')
            prepared={**r,'renderer_photometry':meta,'files':{k:sha_file(directory/k) for k in ('normalized_real.png','support.png','background.png','projected_real_float8.tif')}}
            write_json(directory/'prepared.json',prepared)
            import_corrected_cache(prepared,directory)
            done.append(prepared)
        print(f'CPU prepared {len(done)}/{len(rows)}',flush=True)
    write_json(package/('corrected_cache_import_receipt.json' if a.only_corrected_cache else 'materialization_receipt.json'),{'status':'CPU_corrected_cache_subset_prepared' if a.only_corrected_cache else 'CPU_prepared','instances':len(done),'total_planned_instances':total,
        'actual_generator_calls':0,'actual_Cellpose_calls':0,'test_inputs':0,'optimizer_updates':0})


def generation_record(r,directory,*,actual_call,reused_from=None,artifacts=None):
    artifacts=directory if artifacts is None else artifacts
    return {'cache_key':r['cache_key'],'instance_request_id':r['instance_request_id'],
        'cache_identity':r['cache_identity'],'prepared_sha256':sha_file(directory/'prepared.json'),
        'projected_sha256':sha_file(artifacts/'projected_generated_float8.tif'),
        'model_output_sha256':sha_file(artifacts/'generated_model_domain.png'),
        'generator_sha256':GEN_SHA,'VAE_sha256':VAE_SHA,'recipe':RECIPE,
        'actual_generator_call_this_pool':actual_call,'reused_from':reused_from,
        'source_role':'training_side_only','target_sequence_input_used':False}


def validate_prepared(directory,r,prepared):
    # prepared.json is the atomic completion marker. Reuse does not rehash or
    # rematerialize already committed inputs; byte checks remain in assembly.
    if prepared['cache_key']!=r['cache_key'] or prepared['cache_identity']!=r['cache_identity']:
        raise ValueError('prepared cache identity changed')
    if any(not (directory/k).is_file() for k in prepared['files']):
        raise ValueError('prepared cache files missing')


def shard_rows(rows,rank,num_shards):
    if not 1<=num_shards<=8 or not 0<=rank<num_shards:
        raise ValueError('invalid generation shard rank/count')
    return rows[rank::num_shards]


def generation_pending(package,rows):
    pending=[]; actual_cached=0; reused=0
    for r in rows:
        directory=package/r['directory']; prepared=json.loads((directory/'prepared.json').read_text())
        validate_prepared(directory,r,prepared)
        if (directory/'generated.json').exists():
            gen=json.loads((directory/'generated.json').read_text())
            validate_generated(directory,prepared,gen,verify_bytes=False)
            if bool(gen['actual_generator_call_this_pool'])==('eligible_corrected_reuse' in r):
                raise ValueError('generated cache reuse differs from frozen eligibility')
            actual_cached+=bool(gen['actual_generator_call_this_pool'])
            reused+=not gen['actual_generator_call_this_pool']
        else: pending.append(prepared)
    return pending,actual_cached,reused


def prepare_generation_shards(package,num_shards):
    rows=read_rows(package/'instance_cache_requests.jsonl')
    counts=json.loads((package/'counts.json').read_text()); n=len(rows)
    if n!=counts['proposed_unique_instances'] or len({r['cache_key'] for r in rows})!=n or len({r['directory'] for r in rows})!=n:
        raise ValueError('generation manifest count/ownership mismatch')
    if sum('eligible_corrected_reuse' in r for r in rows)!=counts['eligible_reuse']:
        raise ValueError('corrected cache count mismatch')
    if n-counts['eligible_reuse']!=counts['proposed_new_generation_calls']:
        raise ValueError('generation call budget differs from manifest')
    signature=hashlib.sha256(json.dumps([(r['cache_key'],r['generator_seed']) for r in rows]).encode()).hexdigest()
    root=package/'generation_shards'; plan_path=root/'plan.json'
    prior=json.loads(plan_path.read_text()) if plan_path.exists() else None
    if prior and (prior['num_shards']!=num_shards or prior['manifest_signature']!=signature):
        raise ValueError('resume must preserve generation shard ownership')
    shards=[]; actual_cached_total=0
    for rank in range(num_shards):
        own=shard_rows(rows,rank,num_shards); pending,actual_cached,reused=generation_pending(package,own)
        maximum=sum('eligible_corrected_reuse' not in r for r in own)
        receipt=root/f'rank{rank}.json'
        previous=json.loads(receipt.read_text()) if receipt.exists() else {}
        if previous and any(previous.get(k)!=v for k,v in {'rank':rank,'num_shards':num_shards,'manifest_signature':signature}.items()):
            raise ValueError('generation rank receipt ownership changed')
        attempted=max(actual_cached,previous.get('actual_generator_calls',0))
        if attempted+len(pending)>maximum:
            raise ValueError(f'rank{rank}: interrupted uncommitted call exceeds fixed budget; explicit budget decision required')
        if reused!=sum('eligible_corrected_reuse' in r for r in own):
            raise ValueError('corrected cache import must finish before GPU generation')
        write_rows(root/f'rank{rank}_requests.jsonl',own)
        shards.append({'rank':rank,'instances':len(own),'planned_new_calls':maximum,'cached_actual_calls':actual_cached,'reused':reused,'pending':len(pending)})
        actual_cached_total+=actual_cached
    legacy=package/'generation_runtime_receipt.json'
    if legacy.exists() and json.loads(legacy.read_text()).get('actual_generator_calls',0)>actual_cached_total:
        raise ValueError('uncommitted legacy generation call requires explicit budget decision')
    plan={'num_shards':num_shards,'manifest_signature':signature,'instances':n,'planned_new_calls':counts['proposed_new_generation_calls'],'reused':counts['eligible_reuse'],'shards':shards}
    write_json(plan_path,plan)
    return plan


def merge_generation_shards(package,num_shards):
    root=package/'generation_shards'; plan=json.loads((root/'plan.json').read_text())
    if plan['num_shards']!=num_shards: raise ValueError('generation shard count changed')
    receipts=[]
    for expected in plan['shards']:
        r=json.loads((root/f'rank{expected["rank"]}.json').read_text())
        required={'status':'completed','rank':expected['rank'],'num_shards':num_shards,
                  'manifest_signature':plan['manifest_signature'],'completed_instances':expected['instances'],
                  'actual_generator_calls':expected['planned_new_calls'],'reused':expected['reused']}
        if any(r.get(k)!=v for k,v in required.items()): raise ValueError('generation shard incomplete or differs from plan')
        receipts.append(r)
    actual=sum(r['actual_generator_calls'] for r in receipts)
    if actual!=plan['planned_new_calls'] or sum(r['reused'] for r in receipts)!=plan['reused']:
        raise ValueError('merged generation totals differ from frozen budget')
    receipt={'status':'completed','actual_generator_calls':actual,'planned_unique_instances':plan['instances'],
             'num_shards':num_shards,'cached_corrected_reuse':plan['reused'],'generator_sha256':GEN_SHA,
             'VAE_sha256':VAE_SHA,'optimizer_updates':0,'test_inputs':0,'GPU_idle_gate':False}
    write_json(package/'generation_runtime_receipt.json',receipt)
    return receipt


def commit_generated(directory,r,projected,model_out):
    # A rank owns its cache directories exclusively. Commit the JSON marker last.
    with tempfile.TemporaryDirectory(prefix='.generation-',dir=directory) as tmp:
        stage=Path(tmp)
        tifffile.imwrite(stage/'projected_generated_float8.tif',projected,photometric='rgb',compression='zlib')
        Image.fromarray(model_out).save(stage/'generated_model_domain.png')
        write_json(stage/'generated.json',generation_record(r,directory,actual_call=True,artifacts=stage))
        for name in ('projected_generated_float8.tif','generated_model_domain.png'):
            with (stage/name).open('rb') as handle: os.fsync(handle.fileno())
        for name in ('projected_generated_float8.tif','generated_model_domain.png','generated.json'):
            os.replace(stage/name,directory/name)


def generate(a):
    if not a.execute: raise SystemExit('GPU generation requires --execute after user approval')
    package=a.package.resolve(); cfg=json.loads((package/'protocol_snapshot.json').read_text())
    if not (package/'generation_shards/plan.json').exists() and a.num_shards==1:
        prepare_generation_shards(package,1)
    root=package/'generation_shards'; plan=json.loads((root/'plan.json').read_text())
    if plan['num_shards']!=a.num_shards: raise ValueError('generation shard count changed')
    shard_rows([],a.rank,a.num_shards)
    rows=read_rows(root/f'rank{a.rank}_requests.jsonl')
    pending,actual_cached,reused=generation_pending(package,rows)
    receipt_path=root/f'rank{a.rank}.json'
    previous=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    actual=max(actual_cached,previous.get('actual_generator_calls',0))
    maximum=plan['shards'][a.rank]['planned_new_calls']
    if actual+len(pending)>maximum:
        raise ValueError('interrupted uncommitted call requires explicit budget decision')
    base={'rank':a.rank,'num_shards':a.num_shards,'manifest_signature':plan['manifest_signature'],
          'planned_unique_instances':len(rows),'reused':reused,'optimizer_updates':0,'test_inputs':0,'GPU_idle_gate':False}
    if not pending:
        write_json(receipt_path,{**base,'status':'completed','completed_instances':len(rows),'actual_generator_calls':actual})
        if a.num_shards==1: merge_generation_shards(package,1)
        print('All planned generation artifacts cached; no model loaded'); return
    import torch
    from mscdms.latent_foreground_diffusion import (MaskConditionedLatentForegroundDiffusion,
        normalize_high_latent,denormalize_high_latent)
    from mscdms.checkpoint import config_from_checkpoint
    from mscdms.latent_vae import LatentAutoEncoder
    cp=ROOT/cfg['synthesis']['checkpoint_relative_to_exp4']; vp=ROOT/cfg['synthesis']['vae_checkpoint_relative_to_exp4']
    if sha_file(cp)!=GEN_SHA or sha_file(vp)!=VAE_SHA: raise ValueError('audited assets differ')
    if torch.__version__!='2.11.0+cu130': raise ValueError('use audited existing generator environment')
    if torch.serialization.get_unsafe_globals_in_checkpoint(cp) or torch.serialization.get_unsafe_globals_in_checkpoint(vp): raise ValueError('unsafe checkpoint globals')
    device=torch.device(a.device); torch.cuda.set_device(device)
    payload=torch.load(cp,map_location='cpu',weights_only=True)
    if payload['vae_checkpoint_sha256']!=VAE_SHA: raise ValueError('generator VAE binding mismatch')
    mc=dict(payload['model_config']); mc['channel_mult']=tuple(mc['channel_mult'])
    model=MaskConditionedLatentForegroundDiffusion(**mc); model.load_state_dict(payload['model'],strict=True); model.requires_grad_(False).to(device).eval()
    vpay=torch.load(vp,map_location='cpu',weights_only=True); vc=config_from_checkpoint(vpay)
    vae=LatentAutoEncoder(3,vc.latent_channels,base_ch=vc.vae_base_ch,ch_mult=vc.vae_ch_mult,num_heads=vc.num_heads,gradient_checkpointing=False)
    state={k.removeprefix('vae_high.'):v for k,v in vpay['model'].items() if k.startswith('vae_high.')}
    vae.load_state_dict(state,strict=True); vae.requires_grad_(False).to(device).eval()
    calibration={'enabled':bool(vc.latent_normalization),'mean':tuple(vc.latent_mean_high),'std':tuple(vc.latent_std_high)}
    if calibration['enabled']!=payload['latent_calibration']['enabled'] or any(tuple(calibration[k])!=tuple(payload['latent_calibration'][k]) for k in ('mean','std')): raise ValueError('VAE latent calibration mismatch')
    del state,vpay
    started=time.monotonic()
    with torch.inference_mode():
        for i,r in enumerate(pending):
            directory=package/r['directory']; norm=np.asarray(Image.open(directory/'normalized_real.png').convert('RGB'))
            bg=np.asarray(Image.open(directory/'background.png').convert('RGB')); support=np.asarray(Image.open(directory/'support.png'))>0
            model_rgb=renderer_tensor_rgb(norm,r['renderer_photometry'])
            background=torch.from_numpy(model_rgb.transpose(2,0,1).copy())[None].to(device)
            mask=torch.from_numpy(support.astype(np.float32))[None,None].to(device); background*=1-mask
            src=torch.tensor([payload['source_to_id'][r['generator_source']]],device=device)
            st=torch.tensor([payload['state_to_id'][r['generator_state_key']]],device=device)
            rng=torch.Generator(device=device).manual_seed(r['generator_seed'])
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                z,_=vae.encode(background)
                actual+=1
                write_json(receipt_path,{**base,'status':'running','actual_generator_calls':actual})
                latent=model.sample(normalize_high_latent(z.float(),calibration),mask,src,st,steps=25,generator=rng)
                fg=vae.decode(denormalize_high_latent(latent,calibration)).clamp(-1,1)
            if not torch.isfinite(fg).all(): raise FloatingPointError('nonfinite generated foreground')
            image=fg[0].float().cpu().numpy().transpose(1,2,0)
            native512=compose_native512(image,bg,support,r['renderer_photometry'])
            model_out=np.where(support[...,None],image,model_rgb)
            commit_generated(directory,r,inverse_project_float(native512,r),np.rint((model_out.clip(-1,1)+1)*127.5).astype(np.uint8))
            print(f'P14 rank{a.rank} formal render {i+1}/{len(pending)} {r["instance_request_id"]}',flush=True)
    write_json(receipt_path,{**base,'status':'completed','actual_generator_calls':actual,'completed_instances':len(rows),
        'cached_before_this_invocation':len(rows)-len(pending),'wall_this_invocation_seconds':time.monotonic()-started,
        'generator_sha256':GEN_SHA,'VAE_sha256':VAE_SHA,'optimizer_updates':0,'test_inputs':0,'GPU_idle_gate':False})
    if a.num_shards==1: merge_generation_shards(package,1)


def validate_generated(directory,r,gen,*,verify_bytes=True):
    if gen['cache_identity']!=r['cache_identity'] or gen['cache_key']!=r['cache_key'] or gen['instance_request_id']!=r['instance_request_id']:
        raise ValueError('cache identity/condition mismatch')
    if gen['generator_sha256']!=GEN_SHA or gen['VAE_sha256']!=VAE_SHA or gen['recipe']!=RECIPE:
        raise ValueError('wrong cache generator or preprocessing')
    if gen['source_role']!='training_side_only' or gen['target_sequence_input_used'] is not False:
        raise ValueError('test condition cannot be cached')
    if any(not (directory/name).is_file() for name in ('projected_generated_float8.tif','generated_model_domain.png')):
        raise ValueError('cached generated files missing')
    if verify_bytes and (gen['prepared_sha256']!=sha_file(directory/'prepared.json') or gen['projected_sha256']!=sha_file(directory/'projected_generated_float8.tif') or gen['model_output_sha256']!=sha_file(directory/'generated_model_domain.png')):
        raise ValueError('cached generated bytes changed')


def assemble(a):
    package=a.package.resolve(); records=[]
    for fold in sorted((package/'folds/Fluo-N2DL-HeLa').iterdir()):
        instances={r['instance_request_id']:r for r in read_rows(fold/'single_instance_requests.jsonl')}
        parent_cache={}
        for req in read_rows(fold/'generation_requests.jsonl'):
            refs=req['instance_render_refs']; first=instances[refs[0]]
            if first['parent_frame_id'] not in parent_cache:
                ip,mp=parent_paths(first,a.exp5_root.resolve(),package); raw,labels=tifffile.imread(ip),tifffile.imread(mp)
                if pixel_sha(raw)!=first['parent_image_sha256'] or pixel_sha(labels)!=first['parent_mask_sha256']: raise ValueError('assembly source changed')
                parent_cache[first['parent_frame_id']]=(normalized_rgb(raw),labels)
            rgb,full=parent_cache[first['parent_frame_id']]; y,x,n=(req[k] for k in ('crop_top','crop_left','crop_size'))
            lab=full[y:y+n,x:x+n]; real_inputs={}; synthetic_inputs={}; identities=[]
            for key in refs:
                r=instances[key]; directory=package/r['directory']; prepared=json.loads((directory/'prepared.json').read_text())
                gen=json.loads((directory/'generated.json').read_text()); validate_generated(directory,prepared,gen)
                ident=r['target_instance_id']
                if ident in synthetic_inputs: raise ValueError('duplicate instance in native scene')
                real_inputs[ident]=(tifffile.imread(directory/'projected_real_float8.tif'),r)
                synthetic_inputs[ident]=(tifffile.imread(directory/'projected_generated_float8.tif'),r)
                identities.append(r['cache_key'])
            real,truth,_=compose_scene(rgb[y:y+n,x:x+n],lab,req,real_inputs)
            synthetic,_,checks=compose_scene(rgb[y:y+n,x:x+n],lab,req,synthetic_inputs)
            for key,arr in [('real_image_path',gray_uint16(real)),('synthetic_image_path',gray_uint16(synthetic)),('instance_mask_path',truth)]:
                p=package/req[key]; p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists() and not np.array_equal(tifffile.imread(p),arr): raise ValueError('existing assembled scene differs')
                if not p.exists(): tifffile.imwrite(p,arr,compression='zlib')
            records.append({'condition_id':req['condition_id'],**checks,'cache_keys':identities,
                'real_sha256':sha_file(package/req['real_image_path']),
                'synthetic_sha256':sha_file(package/req['synthetic_image_path']),
                'mask_sha256':sha_file(package/req['instance_mask_path']),
                'real_background_retained':True,'real_foreground_retained_in_synthetic':False})
        own=records[-len(read_rows(fold/'generation_requests.jsonl')):]
        write_json(fold/'generation_validation/generated_pair_receipt.json',{
            'status':'generated_pair_structure_verified','recipe':RECIPE,'scenes':len(own),
            'all_positive_ST_IDs_generated':True,'real_background_retained':True,
            'native_integer_ST_exact':True,'appearance_filtering':False,
            'records':own})
    write_json(package/'assembly_receipt.json',{'status':'CPU_complete_actual_generated_cache_assembled',
        'scenes':len(records),'actual_generator_calls':0,'actual_Cellpose_calls':0,'optimizer_updates':0,'test_inputs':0})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','stage_data','materialize','generate','assemble'))
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--source',type=Path,default=None)
    p.add_argument('--corrected-batch',type=Path,default=None)
    p.add_argument('--parent-root','--exp5-root',dest='exp5_root',type=Path,default=ROOT)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--execute',action='store_true')
    p.add_argument('--rank',type=int,default=0); p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--frames-per-direction',type=int,default=16)
    p.add_argument('--crops-per-frame',type=int,default=1)
    p.add_argument('--scope-confirmed',action='store_true')
    p.add_argument('--only-corrected-cache',action='store_true',help='CPU import only already completed corrected outputs')
    p.add_argument('--run-root',default='outputs/segmentation')
    a=p.parse_args(); globals()[a.mode](a)


if __name__=='__main__': main()
