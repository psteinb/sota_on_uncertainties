import sys
from pathlib import Path

thisdir = Path(__file__).absolute().parent
rootdir = thisdir.parent
wfdir = rootdir / "workflow"

if not str(wfdir) in sys.path:
    sys.path.append(str(wfdir))
    # print(f"[conftest.py] added {wfdir} to {sys.path}")
