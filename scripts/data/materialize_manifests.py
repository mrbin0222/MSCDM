#!/usr/bin/env python3
"""Expand the fixed sample catalog into local runtime manifests, without images or models."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from mscdms.manifests import ManifestCatalog,DEFAULT_CATALOG
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,default=DEFAULT_CATALOG)
    parser.add_argument('--output-dir',type=Path,default=Path('data/processed'))
    args=parser.parse_args()
    print(json.dumps(ManifestCatalog(args.catalog).materialize(args.output_dir),indent=2))
