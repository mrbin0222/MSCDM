#!/usr/bin/env python3
"""Attach the actual released text by sample ID, without language-model calls."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from mscdms.manifests import ManifestCatalog,DEFAULT_CATALOG,read_jsonl
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,default=DEFAULT_CATALOG)
    parser.add_argument('--texts',type=Path,help='Optional external expanded text manifest; omit to use the fixed catalog')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    # Backward compatibility for callers that explicitly passed the old default path.
    compact=args.texts is not None and args.texts.name=='samples.jsonl' and (args.texts.parent/'catalog.json').is_file()
    if args.texts is None or compact:
        values=ManifestCatalog(args.texts.parent if compact else args.catalog).iter_profile('samples')
    else:values=read_jsonl(args.texts)
    texts={r['sample_id']:r for r in values}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8')as handle:
        for row in read_jsonl(args.input):
            text=texts[row['sample_id']]
            if row['split']!=text['split']:raise ValueError('Split mismatch')
            row.update({k:text[k]for k in ['subject','state','background']})
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
