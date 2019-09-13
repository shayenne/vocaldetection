from __future__ import division
import numpy as np


class Fluctogram:
    def __init__(self, spec_log, f_log, f_start=164, f_end=10548):
        # get indices for frequency range E3 (164 Hz) to E9 (10548 Hz)
        f_start_idx = np.argmin(np.abs(f_log - f_start))
        f_end_idx = np.argmin(np.abs(f_log - f_end))
        self.spec_log = spec_log[f_start_idx:f_end_idx+1, :]
        # self.spec_log = spec_log

        # parameters for the subbands
        self.n_bands = 17
        self.bandwith = 240  # in bins
        self.bands_offset = 30  # in bins

        # parameter for the correlation
        self.bin_shift = np.arange(-5, 6)

        self.fluctogram = np.zeros((self.n_bands, self.spec_log.shape[1]))

        self.extract()

    def extract(self):
        # get window function as a matrix
        win = self._get_triangle_window((self.bandwith, self.spec_log.shape[1]))

        # extract the subbands
        for cur_band_idx in np.arange(self.n_bands):
            cur_band_start = cur_band_idx * self.bands_offset
            cur_band_end = cur_band_start + self.bandwith

            # assign the subbands
            cur_band = self.spec_log[cur_band_start:cur_band_end, :].copy()

            # weight the subbands with the triangular window
            cur_band *= win

            for cur_frame in np.arange(self.spec_log.shape[1]-1):
                cur_frame_spec = cur_band[:, cur_frame]
                next_frame_spec = cur_band[:, cur_frame+1]

                # cross-correlate both frames
                xc = np.correlate(cur_frame_spec, next_frame_spec, 'same')

                # normalize according to Pearson at lag 0 (center bin)
                center_bin = int(np.floor(len(xc)/2))
                xc /= xc[center_bin]

                # Bins of interest: get +- 5 bins around center
                boi = self.bin_shift + center_bin
                xc_boi = xc[boi.tolist()]

                # take maximum idx and center it
                self.fluctogram[cur_band_idx, cur_frame] = np.argmax(xc_boi) + np.min(self.bin_shift)

    def visualize(self):
        import matplotlib.pyplot as plt

        for cur_band in np.arange(self.n_bands):
            plt.plot(self.fluctogram[cur_band, :]+(cur_band+1)*3, 'k')

    @staticmethod
    def _get_triangle_window(shape):
        win = np.bartlett(shape[0])

        return np.tile(win, (shape[1], 1)).T


def stft_interp(spec, source_freqs, target_freqs):
    """Compute an interpolated version of the spectrogram. Uses scipy.interp1d to map
       to the new frequency axis.
    """
    # magnitude spectrogram
    spec = np.abs(spec)

    from scipy.interpolate import interp1d
    set_interp = interp1d(source_freqs, spec, kind='linear', axis=0)
    spec_interp = set_interp(target_freqs)

    return spec_interp


if __name__ == '__main__':
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import sys
    import spectral

    audio_file  = sys.argv[1]

    # load the audio
    y, sr = librosa.load(audio_file)

    # get log freq axis
    bins_per_octave = 120
    target_freqs = librosa.cqt_frequencies(6*bins_per_octave, fmin=librosa.note_to_hz('E3'),
                                           bins_per_octave=bins_per_octave)

    n_fft = 4096
    hop_length = 441
    y_stft = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)

    y_stft_log = stft_interp(y_stft, librosa.core.fft_frequencies(sr=sr, n_fft=n_fft), target_freqs)
    librosa.display.specshow(np.log(1 + y_stft_log), sr=sr, x_axis='time', y_axis='linear',
                             cmap=plt.get_cmap('gray_r'))

    # Calculate the fluctogram
    print ("Spectral Flatness")
    print (spectral.bandwise_flatness(y_stft_log, target_freqs))
    print ("Spectral Contraction")
    print (spectral.bandwise_contraction(y_stft_log, target_freqs))
    fluctogram = Fluctogram(y_stft_log, target_freqs)
    plt.figure()
    fluctogram.visualize()

    plt.show()

