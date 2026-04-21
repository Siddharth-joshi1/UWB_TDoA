# nlos_classifier.py  –  Physics-based NLOS/LOS classifier for UWB signals
#
# Features extracted from the received UWB signal:
#   1. kurtosis         – NLOS signals have lower kurtosis (multipath spreads energy)
#   2. first-path ratio – ratio of first-peak energy to total energy (low in NLOS)
#   3. RMS delay spread – width of the power-delay profile (high in NLOS)
#   4. peak-to-noise    – SNR at the detected peak
#   5. pulse width      – NLOS pulses appear wider due to multipath
#
# Model: LogisticRegression trained on simulated LOS/NLOS signals
# This upgrades the channel from "random 15% NLOS bias" to physics-based detection.
#
# References:
#   Maranò et al. (2010) – NLOS identification and mitigation for UWB localization
#   Wymeersch et al. (2012) – UWB localization in NLOS environments

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from config import C, ANCHORS
from phy    import generate_uwb_signal, gaussian_pulse
from channel import add_nlos_bias, generate_multipath, generate_amplitudes, apply_path_loss


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation for classifier training
# ─────────────────────────────────────────────────────────────────────────────

def _generate_los_signal(distance: float, fs: float = 1e9,
                          noise_std: float = 0.005) -> np.ndarray:
    """LOS signal: direct path + weak echoes, NO NLOS bias."""
    true_delay = distance / C
    n_samples  = int(6e-7 * fs)
    signal     = np.zeros(n_samples)
    sigma      = 1e-9

    # Direct path (strong)
    center = int(true_delay * fs)
    half   = int(5 * sigma * fs) + 1
    for k in range(-half, half + 1):
        idx = center + k
        if 0 <= idx < n_samples:
            signal[idx] += gaussian_pulse(k / fs, sigma)

    # Weak multipath echoes (0.1–0.3× amplitude, 5–15 ns later)
    n_paths = np.random.randint(1, 4)
    for _ in range(n_paths):
        amp   = np.random.uniform(0.1, 0.3)
        extra = np.random.uniform(5e-9, 15e-9)
        c2    = int((true_delay + extra) * fs)
        for k in range(-half, half + 1):
            idx = c2 + k
            if 0 <= idx < n_samples:
                signal[idx] += amp * gaussian_pulse(k / fs, sigma)

    path_loss = 1.0 / max(distance, 0.5) ** 1.5
    signal   *= path_loss
    signal   += np.random.normal(0, noise_std, n_samples)
    return signal


def _generate_nlos_signal(distance: float, fs: float = 1e9,
                           noise_std: float = 0.005) -> np.ndarray:
    """NLOS signal: first path attenuated/blocked, stronger echoes."""
    nlos_bias  = np.random.uniform(2e-9, 8e-9)  # 0.6–2.4 m extra delay
    true_delay = distance / C + nlos_bias
    n_samples  = int(6e-7 * fs)
    signal     = np.zeros(n_samples)
    sigma      = 1e-9

    # First path is weakened (blocked by obstacle, 0.2–0.6× amplitude)
    first_amp = np.random.uniform(0.2, 0.6)
    center    = int(true_delay * fs)
    half      = int(5 * sigma * fs) + 1
    for k in range(-half, half + 1):
        idx = center + k
        if 0 <= idx < n_samples:
            signal[idx] += first_amp * gaussian_pulse(k / fs, sigma)

    # Multiple strong multipath echoes (more energy spread = NLOS signature)
    n_paths = np.random.randint(3, 7)
    for _ in range(n_paths):
        amp   = np.random.uniform(0.3, 0.9)   # echoes can be as strong as direct
        extra = np.random.uniform(3e-9, 20e-9)
        c2    = int((true_delay + extra) * fs)
        for k in range(-half, half + 1):
            idx = c2 + k
            if 0 <= idx < n_samples:
                signal[idx] += amp * gaussian_pulse(k / fs, sigma)

    path_loss = 1.0 / max(distance, 0.5) ** 1.5
    signal   *= path_loss
    signal   += np.random.normal(0, noise_std, n_samples)
    return signal


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(signal: np.ndarray, fs: float = 1e9) -> np.ndarray:
    """
    Extract 6 discriminative features from a UWB baseband signal.

    Features
    --------
    f0 : kurtosis of signal amplitude
         LOS: high kurtosis (sharp single peak)
         NLOS: low kurtosis (energy spread over multiple peaks)

    f1 : first-path energy ratio = E_first / E_total
         LOS: > 0.6  (direct path dominates)
         NLOS: < 0.3  (echoes carry most energy)

    f2 : RMS delay spread (ns)
         LOS: narrow  (< 5 ns)
         NLOS: wide   (> 10 ns)

    f3 : peak-to-RMS-noise ratio (dB)
         LOS: high SNR
         NLOS: lower (obstacle attenuates direct path)

    f4 : pulse width at 50% max (samples)
         LOS: narrow  (≈ 2–4 samples)
         NLOS: wide   (spread by multipath)

    f5 : second peak ratio = second_peak / first_peak
         LOS: low (< 0.3)
         NLOS: high (strong echoes nearly as large as direct path)
    """
    sig2    = signal ** 2                           # power
    t_ax    = np.arange(len(signal)) / fs * 1e9    # time in ns

    # f0: kurtosis
    f0 = float(scipy_kurtosis(signal, fisher=True))

    # f1: first-path energy ratio
    peak_idx   = int(np.argmax(np.abs(signal)))
    window     = int(5e-9 * fs)                    # 5 ns window around first peak
    fp_start   = max(0, peak_idx - window)
    fp_end     = min(len(signal), peak_idx + window)
    e_first    = np.sum(sig2[fp_start:fp_end])
    e_total    = np.sum(sig2) + 1e-30
    f1         = float(e_first / e_total)

    # f2: RMS delay spread
    if e_total > 1e-20:
        mean_delay = np.sum(t_ax * sig2) / e_total
        f2 = float(np.sqrt(np.sum(sig2 * (t_ax - mean_delay)**2) / e_total))
    else:
        f2 = 0.0

    # f3: peak SNR (dB)
    peak_power = np.max(sig2)
    # estimate noise from tail of signal (last 20%)
    tail_start = int(0.8 * len(signal))
    noise_power = np.mean(sig2[tail_start:]) + 1e-30
    f3 = float(10 * np.log10(peak_power / noise_power + 1e-10))

    # f4: pulse width at 50% max amplitude
    half_max = 0.5 * np.max(np.abs(signal))
    above    = np.where(np.abs(signal) >= half_max)[0]
    f4 = float(len(above))

    # f5: second peak ratio
    # Find second distinct peak (> 5 ns away from first)
    sep  = int(5e-9 * fs)
    sig_abs = np.abs(signal).copy()
    sig_abs[max(0, peak_idx - sep):min(len(signal), peak_idx + sep)] = 0
    second_peak = np.max(sig_abs) if len(sig_abs) > 0 else 0
    first_peak  = np.max(np.abs(signal)) + 1e-30
    f5 = float(second_peak / first_peak)

    return np.array([f0, f1, f2, f3, f4, f5])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_samples: int = 2000, fs: float = 1e9,
                     dist_range: tuple = (0.5, 7.0),
                     seed: int = 0) -> tuple:
    """
    Generate balanced LOS/NLOS dataset with extracted features.

    Returns (X, y) where y=0 is LOS, y=1 is NLOS.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for _ in range(n_samples // 2):
        d  = rng.uniform(*dist_range)
        # LOS
        sig = _generate_los_signal(d, fs)
        X.append(extract_features(sig, fs))
        y.append(0)
        # NLOS
        sig = _generate_nlos_signal(d, fs)
        X.append(extract_features(sig, fs))
        y.append(1)

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
# NLOSClassifier class
# ─────────────────────────────────────────────────────────────────────────────

class NLOSClassifier:
    """
    Physics-based LOS/NLOS classifier for UWB signals.

    Usage
    -----
    clf = NLOSClassifier()
    clf.train(n_samples=2000)
    label, prob = clf.predict(signal)   # label: 0=LOS, 1=NLOS
    """

    FEATURE_NAMES = [
        'Kurtosis', 'First-path ratio', 'RMS delay spread (ns)',
        'Peak SNR (dB)', 'Pulse width (samples)', 'Second-peak ratio'
    ]

    def __init__(self, model: str = 'logistic'):
        self.scaler  = StandardScaler()
        self.model   = model
        self._trained = False

        if model == 'logistic':
            self.clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        elif model == 'svm':
            self.clf = SVC(kernel='rbf', C=2.0, probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model: {model}")

    def train(self, n_samples: int = 2000, fs: float = 1e9,
              verbose: bool = True) -> dict:
        """Generate training data and fit classifier."""
        if verbose:
            print(f"  Generating {n_samples} LOS/NLOS signal pairs …")
        X, y = generate_dataset(n_samples, fs, seed=42)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation
        cv_scores = cross_val_score(self.clf, X_scaled, y, cv=5, scoring='accuracy')

        # Final fit on all data
        self.clf.fit(X_scaled, y)
        self._trained = True

        if verbose:
            print(f"  5-fold CV accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

        return {
            'cv_mean':  float(cv_scores.mean()),
            'cv_std':   float(cv_scores.std()),
            'n_samples': n_samples,
            'model':    self.model,
        }

    def predict(self, signal: np.ndarray, fs: float = 1e9) -> tuple:
        """
        Classify a single received signal.

        Returns
        -------
        label : int   0 = LOS,  1 = NLOS
        prob  : float probability of NLOS
        """
        if not self._trained:
            raise RuntimeError("Call train() before predict()")
        feat  = extract_features(signal, fs).reshape(1, -1)
        feat_s = self.scaler.transform(feat)
        label  = int(self.clf.predict(feat_s)[0])
        prob   = float(self.clf.predict_proba(feat_s)[0][1])
        return label, prob

    def predict_batch(self, signals: list, fs: float = 1e9) -> tuple:
        """Classify a list of signals. Returns (labels, probs)."""
        feats  = np.array([extract_features(s, fs) for s in signals])
        feats_s = self.scaler.transform(feats)
        labels  = self.clf.predict(feats_s).astype(int)
        probs   = self.clf.predict_proba(feats_s)[:, 1]
        return labels, probs

    def evaluate(self, n_test: int = 500, fs: float = 1e9) -> dict:
        """Evaluate on a held-out test set."""
        X_test, y_test = generate_dataset(n_test, fs, seed=99)
        X_scaled = self.scaler.transform(X_test)
        y_pred   = self.clf.predict(X_scaled)
        report   = classification_report(y_test, y_pred,
                                          target_names=['LOS','NLOS'],
                                          output_dict=True)
        return report

    def feature_importance(self) -> dict:
        """Return feature weights (logistic only)."""
        if self.model != 'logistic':
            return {}
        coefs = self.clf.coef_[0]
        return dict(zip(self.FEATURE_NAMES, coefs))