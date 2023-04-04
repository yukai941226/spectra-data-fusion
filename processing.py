import numpy as np
from scipy.signal import savgol_filter
from scipy.linalg import solveh_banded


def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e4, max_iters=5, conv_thresh=1e-5, verbose=False):
    """
    The als_baseline method, including the WhittakerSmoother Class was taken from
    https://github.com/all-umass/superman/blob/master/superman/baseline/als.py
    with permission from github user CJ Carey (perimosocordiae)

    Computes the asymmetric least squares baseline.
    * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
    smoothness_param: Relative importance of smoothness of the predicted response.
    asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                         Setting p=1 is effectively a hinge loss.
    """
    z = intensities
    if max_iters > 0:
        smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
        # Rename p for concision.
        p = asymmetry_param
        # Initialize weights.
        w = np.ones(intensities.shape[0])
        for i in range(max_iters):
            z = smoother.smooth(w)
            mask = intensities > z
            new_w = p * mask + (1 - p) * (~mask)
            conv = np.linalg.norm(new_w - w)
            if verbose:
                print(i + 1, conv)
            if conv < conv_thresh:
                break
            w = new_w

    return z


class WhittakerSmoother(object):
    def __init__(self, signal: np.ndarray, smoothness_param, deriv_order=1):
        self.y = signal.copy()
        assert deriv_order > 0, 'deriv_order must be an int > 0'
        # Compute the fixed derivative of identity (D).
        d = np.zeros(deriv_order * 2 + 1, dtype=int)
        d[deriv_order] = 1
        d = np.diff(d, n=deriv_order)
        n = self.y.shape[0]
        k = len(d)
        s = float(smoothness_param)

        # Here be dragons: essentially we're faking a big banded matrix D,
        # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
        diag_sums = np.vstack([
            np.pad(s * np.cumsum(d[-i:] * d[:i]), ((k - i, 0),), 'constant')
            for i in range(1, k + 1)])
        upper_bands = np.tile(diag_sums[:, -1:], n)
        upper_bands[:, :k] = diag_sums
        for i, ds in enumerate(diag_sums):
            upper_bands[i, -i - 1:] = ds[::-1][:i + 1]
        self.upper_bands = upper_bands

    def smooth(self, w):
        foo = self.upper_bands.copy()
        foo[-1] += w  # last row is the diagonal
        return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)


def mapSpecToWavenumbers(spec: np.ndarray, targetWavenumbers: np.ndarray) -> np.ndarray:
    newSpec = np.zeros((len(targetWavenumbers), 2))
    newSpec[:, 0] = targetWavenumbers
    for i in range(newSpec.shape[0]):
        closestIndex = np.argmin(np.abs(spec[:, 0] - targetWavenumbers[i]))
        newSpec[i, 1] = spec[closestIndex, 1]
    return newSpec


def normalizeIntensities(intensities: np.ndarray) -> np.ndarray:
    intensities -= intensities.min()
    intensities /= intensities.max()
    return intensities


def snv(input_data: np.ndarray) -> np.ndarray:
    """
    Standard normal variate Correction.
    :param input_data: Shape (NxM) array of N samples with M features
    :return: corrected data in same shape
    """
    output_data: np.ndarray = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    return output_data


def mean_center(input_data: np.ndarray) -> np.ndarray:
    """
    Mean Centering.
    :param input_data: Shape (NxM) array of N samples with M features
    :return: corrected data in same shape
    """
    output_data: np.ndarray = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        output_data[i, :] = input_data[i, :] - np.mean(input_data[i, :])
    return output_data


def detrend(input_data: np.ndarray) -> np.ndarray:
    """
    Removes a linear baseline.
    :param input_data: Shape (NxM) array of N samples with M features
    :return: corrected data in same shape
    """
    output_data: np.ndarray = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        baseline = np.linspace(input_data[i, 0], input_data[i, -1], len(input_data[i, :]))
        output_data[i, :] = input_data[i, :] - baseline
    return output_data


def smooth(input_data: np.ndarray, windowSize: int) -> np.ndarray:
    """
    Applies Savitzky Golay smoothing to all given data.
    :param input_data: Shape (NxM) array of N samples with M features
    :param windowSize: integer, the window size for smoothing
    :return: corrected data in same shape
    """
    output_data: np.ndarray = np.zeros_like(input_data)
    if windowSize % 2 == 0:
        windowSize += 1  # has to be an odd number!
    for i in range(input_data.shape[0]):
        output_data[i, :] = savgol_filter(input_data[i, :], windowSize, polyorder=3)
    return output_data
