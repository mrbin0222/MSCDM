from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from mscdms.workflows.prepare_mask_geometry import main
if __name__=='__main__':main()
