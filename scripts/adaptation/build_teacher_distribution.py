from pathlib import Path
import sys,runpy
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
if __name__=='__main__':runpy.run_module('mscdms.workflows.build_teacher_distribution',run_name='__main__')
