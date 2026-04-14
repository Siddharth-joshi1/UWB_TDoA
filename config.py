import numpy as np
C         = 3e8
AREA_SIZE = 5.0
HEIGHT    = 3.0
GRID_RES  = 0.75   # coarser for speed; use 0.5 for denser grid
ANCHORS   = np.array([[0.,0.,2.],[5.,0.,2.],[5.,5.,2.],[0.,5.,2.]])
REF_TAG_POS = np.array([AREA_SIZE/2, AREA_SIZE/2, HEIGHT/2])
FS        = 1e9
NOISE_STD = 0.1e-9
SYNC_ERROR_STD = 1e-9
SYNC_ERROR_LIST = [0,0.5e-9,1e-9,2e-9,5e-9,10e-9]
MC_RUNS   = 3    # per-point MC runs (increase for accuracy)
MAX_ERROR = 5
SR_N = 16   # 16 blocks → √16 = 4× SNR gain (~12 dB)
SR_M = 4    # 4× oversample → 0.25 ns resolution (7.5 cm)