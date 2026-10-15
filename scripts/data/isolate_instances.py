#!/usr/bin/env python3
from pathlib import Path
import sys, runpy
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
if __name__ == '__main__':
    runpy.run_module('mscdms.workflows.isolate_instances', run_name='__main__')
