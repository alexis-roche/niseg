from .pve import PVE, BrainT1PVE, MultichannelPVE, FuzzyCMean, MultichannelFuzzyCMean
from .vem import VEM
from .brain_segmentation import BrainT1Segmentation
from .moment_matching import moment_matching

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
