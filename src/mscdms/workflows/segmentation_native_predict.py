#!/usr/bin/env python3
"""Segmentation native predict."""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from mscdms.workflows.segmentation_prepare import pixel_sha,sha_file,read_rows,write_rows,write_json

PARAMS={'batch_size':1,'flow_threshold':0.4,'cellprob_threshold':0.0,'min_size':15,
        'normalize':True,'resample':True,'augment':False,'tile_overlap':0.1}


def verify_frame(record,expected,checkpoint_sha,out):
    if record['frame_id']!=expected['frame_id'] or record['input_pixel_sha256']!=expected['image_pixel_sha256']:
        raise ValueError('cached prediction input/frame differs')
    if record['checkpoint_sha256']!=checkpoint_sha or record['eval_parameters']!=PARAMS:
        raise ValueError('cached prediction checkpoint/eval parameters differ')
    path=out/record['pred_instance_path']
    if not path.is_file() or sha_file(path)!=record['prediction_sha256']:
        raise ValueError('cached prediction output hash mismatch')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--test-pairs',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    expected=read_rows(a.test_pairs);manifest={r['frame_id']:r for r in read_rows(a.data_dir/'manifest.jsonl')}
    if len(expected)!=92 or len({r['sequence'] for r in expected})!=1:
        raise ValueError('exactly92 frozen held-out frames required')
    checkpoint_sha=sha_file(a.checkpoint);rows=[];pending=[]
    for r in expected:
        src=manifest[r['frame_id']]
        if src['dataset']!=r['dataset'] or src['sequence']!=r['sequence']:
            raise ValueError('test manifest sequence mismatch')
        record_path=out/'frame_records'/(r['sample_id']+'.json')
        if record_path.is_file():
            record=json.loads(record_path.read_text());verify_frame(record,r,checkpoint_sha,out);rows.append(record)
        else:pending.append((r,src,record_path))
    receipt_path=out/'gpu_receipt.json';old=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if old and (old['checkpoint_sha256']!=checkpoint_sha or old['eval_parameters']!=PARAMS):
        raise ValueError('prediction progress bound to different checkpoint/parameters')
    actual=old.get('actual_image_eval_calls',0)
    if pending:
        import torch
        from cellpose import models
        device=torch.device(a.device)
        model=models.CellposeModel(gpu=device.type=='cuda',device=device,pretrained_model=str(a.checkpoint))
        model.net.requires_grad_(False).eval()
        with torch.inference_mode():
            for r,src,record_path in pending:
                image=tifffile.imread(a.data_dir/src['image_path'])
                if image.ndim!=2 or pixel_sha(image)!=r['image_pixel_sha256']:
                    raise ValueError('held-out native image pixels differ')
                actual+=1
                write_json(receipt_path,{'status':'running','checkpoint_sha256':checkpoint_sha,'eval_parameters':PARAMS,
                    'actual_image_eval_calls':actual,'completed_frames':len(rows),'optimizer_updates':0})
                mask=np.asarray(model.eval(image,**PARAMS)[0],dtype=np.int32)
                path=Path('instance_maps')/r['dataset']/r['sequence']/(r['sample_id']+'.tif')
                (out/path).parent.mkdir(parents=True,exist_ok=True);tifffile.imwrite(out/path,mask,compression='zlib')
                row={**r,'pred_instance_path':str(path),'checkpoint_sha256':checkpoint_sha,
                     'input_pixel_sha256':pixel_sha(image),'eval_parameters':PARAMS,'prediction_sha256':sha_file(out/path)}
                write_json(record_path,row);rows.append(row)
                print(f'P14 full FOV {len(rows)}/92 {r["frame_id"]}',flush=True)
    by_id={r['frame_id']:r for r in rows};write_rows(out/'prediction_manifest.jsonl',[by_id[r['frame_id']] for r in expected])
    write_json(receipt_path,{'status':'completed','checkpoint_sha256':checkpoint_sha,'eval_parameters':PARAMS,
        'actual_image_eval_calls':actual,'completed_frames':92,'optimizer_updates':0,
        'resume_reuses_complete_checkpoint_bound_frames':True})


if __name__=='__main__':main()
