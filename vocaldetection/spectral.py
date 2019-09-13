import numpy as np
from scipy.signal import chebwin

def bandwise_contraction(X_log, freq_ax_log, f_start=164, f_end=10548, n_bands=17, bandwith=240, bands_offset=30):
        # get indices for frequency range E3 (164 Hz) to E9 (10548 Hz)
        f_start_idx = np.argmin(np.abs(freq_ax_log - f_start))
        f_end_idx = np.argmin(np.abs(freq_ax_log - f_end))
        X_log = X_log[f_start_idx:f_end_idx+1, :]

        bw_contraction = np.zeros((n_bands, X_log.shape[1]))

        # extract the subbands
        for cur_band_idx in np.arange(n_bands):
            cur_band_start = cur_band_idx * bands_offset
            cur_band_end = cur_band_start + bandwith

            # assign the subbands
            cur_band = X_log[cur_band_start:cur_band_end, :].copy()

            # Call standard spectral contraction for each band
            bw_contraction[cur_band_idx, :] = spectral_contraction(cur_band)

        return bw_contraction


def spectral_contraction(X_mag):
    """Spectral Contraction measure.

       As suggested in _[1].

    Parameters
    ----------
    X_mag : ndarray
        Magnitude spectrum of a time frame.

    Returns
    -------
    spectral_contraction :

    References
    ----------
    .. [1] Bernhard Lehner, Gerhard Widmer, Reinhard Sonnleitner
           "ON THE REDUCTION OF FALSE POSITIVES IN SINGING VOICE DETECTION",
           ICASSP 2014

    """
    window = chebwin(X_mag.shape[0], 200)
    if X_mag.ndim > 1:
        window = np.tile(window, (X_mag.shape[1], 1)).T

    spectral_contraction = np.sum(X_mag * window, axis=0) / (np.sum(X_mag, axis=0) + np.finfo(float).eps)

    return spectral_contraction


def bandwise_flatness(X_log, freq_ax_log, f_start=164, f_end=10548, n_bands=17, bandwith=240, bands_offset=30):
        f_start_idx = np.argmin(np.abs(freq_ax_log - f_start))
        f_end_idx = np.argmin(np.abs(freq_ax_log - f_end))
        X_log = X_log[f_start_idx:f_end_idx+1, :]

        bw_flatness = np.zeros((n_bands, X_log.shape[1]))

        # extract the subbands
        for cur_band_idx in np.arange(n_bands):
            cur_band_start = cur_band_idx * bands_offset
            cur_band_end = cur_band_start + bandwith

            # assign the subbands
            cur_band = X_log[cur_band_start:cur_band_end, :].copy()

            # Call standard spectral flatness for each band
            bw_flatness[cur_band_idx, :] = spectral_flatness(cur_band)

        return bw_flatness


def spectral_flatness(X_mag):
    """Spectral Flatness measure.

    We use the log-spec version e use the log-spec version as suggested by _[1].

    Parameters
    ----------
    X_mag : ndarray
        Magnitude spectrum of a time frame.

    Returns
    -------
    spectral_flatness :

    References
    ----------
    .. [1] Alexander Lerch, "Audio Content Analysis"

    """
    # geometric mean
    spectral_flatness = np.exp(np.mean(np.log(X_mag + np.finfo(float).eps), axis=0))

    # divided by arithmetic mean
    spectral_flatness /= np.mean(X_mag, axis=0) + np.finfo(float).eps

    return spectral_flatness
