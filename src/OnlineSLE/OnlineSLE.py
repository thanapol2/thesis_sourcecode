import numpy as np
from scipy.fft import fft, fftfreq
import src.utility.utility_frequency_analysis as utility_frequency

class OnlineSLE():

    def __init__(self, W: np.ndarray, SPLE_type = 'HAQSE'):

        if SPLE_type not in ['HAQSE', 'QSE', None]:
            raise ValueError("Estimator is not correct")

        self.SPLE_type = SPLE_type
        self.window_size = len(W)
        self.W = W
        self.mathcal_F = fft(W)
        self.N = len(self.mathcal_F)
        self.k = np.arange(self.N)
        self.twiddle = np.exp(2j * np.pi * self.k / self.N)
        self.xfreq, self.pden = utility_frequency.peridogram(self.window_size, self.mathcal_F)
        self.initial_flag = False
        self.frequency_tone = 0


    def _get_season_length(self):
        indices_above_threshold, peak_index = utility_frequency.get_period_hints(self.pden)
        if self.SPLE_type == None:
            if peak_index > 1:
                self.frequency_tone = self.xfreq[peak_index]
                SLE_result = round(1 / self.frequency_tone)
            else:
                self.frequency_tone = 0
                SLE_result = 1
        else:
            if  self.SPLE_type == 'QSE':
                k_hat, frequency_tone = utility_frequency.qse(self.W, peak_index, self.xfreq[1] - self.xfreq[0])
            elif self.SPLE_type == 'HAQSE':
                k_hat, frequency_tone = utility_frequency.haqse(self.pden, self.W)
            self.frequency_tone = frequency_tone
            if k_hat > 1:
                SLE_result = round(1 / self.frequency_tone)
            else:
                SLE_result = 1
        return SLE_result

    def initial_phase(self):
        self.initial_flag = True
        return self._get_season_length()

    def online_phase(self, y_t):
        if not self.initial_flag:
            raise ValueError("Please compute initial phase ")
        else:
            y_oldest = self.W[0]
            self.W = self.W[1:]
            self.W = np.append(self.W, y_t)
            self.mathcal_F = utility_frequency.update_sDFT(self.mathcal_F, y_oldest, y_t, self.twiddle)
            self.xfreq, self.pden = utility_frequency.peridogram(self.window_size, self.mathcal_F)
            return self._get_season_length()

    def get_periodogram(self):
        return self.xfreq, self.pden