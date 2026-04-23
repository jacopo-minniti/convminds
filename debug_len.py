import sys
from unittest.mock import MagicMock
sys.modules['h5py'] = MagicMock()
sys.modules['osfclient'] = MagicMock()
sys.modules['osfclient.api'] = MagicMock()
import convminds as cm
from convminds.data.benchmarks.huth_alignment import HuthAlignmentDataset

try:
    ds = HuthAlignmentDataset(subject_ids=["S1"], split="train")
    print("Train set size:", len(ds))
except Exception as e:
    print("Error:", e)
