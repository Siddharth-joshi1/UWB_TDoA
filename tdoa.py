# tdoa.py
import numpy as np
from config import C, NOISE_STD, SYNC_ERROR_LIST
from phy import generate_uwb_signal
from toa_estimator import estimate_toa


## withi=out phy:

# def generate_tdoa(point, anchors, sync_error_std):
#     d = np.linalg.norm(anchors - point, axis=1)
#     t = d / C

#     noise = np.random.normal(0, NOISE_STD, size=t.shape)
#     sync_error = np.random.normal(0, sync_error_std, size=t.shape)

#     t_noisy = t + noise + sync_error

#     # ALL pairwise TDoA
#     tdoa = []
#     for i in range(len(t_noisy)):
#         for j in range(i+1, len(t_noisy)):
#             tdoa.append(t_noisy[i] - t_noisy[j])

#     return np.array(tdoa)


# tdoa.py


# tdoa.py

import numpy as np
from config import C
from phy import generate_uwb_signal
from toa_estimator import estimate_toa
from channel import add_nlos_bias


def generate_tdoa(point, anchors, fs=1e9):

    toas = []

    for anchor in anchors:

        d = np.linalg.norm(anchor - point)

        delay = d / C

        delay = add_nlos_bias(delay)

        signal = generate_uwb_signal(
            delay,
            d,
            fs,
            duration=1e-7,
            noise_std=0.01
        )

        toa = estimate_toa(signal, fs)

        toas.append(toa)

    toas = np.array(toas)

    tdoa = []

    for i in range(len(toas)):
        for j in range(i+1, len(toas)):
            tdoa.append(toas[i] - toas[j])

    return np.array(tdoa)