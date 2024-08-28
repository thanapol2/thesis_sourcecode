from numpy import sqrt, mean, log


class Mean_EBinning:
    """
    Mean-Ebinning a novel approach to solve the problem of sketching time-series data (TSD).
    The Mean-EBinning automatically specifies the appropriate bin size (𝑛) for each bin in a window (𝑊).

    References
    ----------
    [1] T. Phungtua-eng, Y. Yamamoto, and S. Sako. 2023. Elastic Data Binning for
    Transient Pattern Analysis in Time-Domain Astrophysics.
    In Proceedings of the 38th Annual ACM Symposium on Applied Computing (SAC '23).
    DOI : https://doi.org/10.1145/3555776.3577606
    dataset and supplementary material : https://sites.google.com/view/elasticdatabinning

    """

    def __init__(self, max_window_size=100, ini_binsize=8, alpha=0.95):
        self.max_window_size = max_window_size
        self.window = []  # W
        self.score_profile = []
        self.buffer = []
        self.ini_binsize = ini_binsize
        self.alpha = alpha

    def reset_memory(self):
        self.window = []
        self.buffer = []

    def reset_buffer(self):
        self.buffer = []

    """
    Algorithm 1. Baseline EBinning 
    """

    def add_element(self, value):
        self.buffer.append(value)
        if len(self.buffer) >= self.ini_binsize:
            self.insert_bin_into_window()

    def insert_bin_into_window(self):
        self.window.append(self.initialization_bin(self.buffer))  # Bin initialization
        self.reset_buffer()
        # Compute Score (bin_{|w|}, bin_{|w|-1})
        if len(self.window) == 1:
            k = 0
        else:
            latest_index = len(self.window) - 1
            k = self.compute_score(prior_bin=self.window[latest_index - 1],
                                   latest_bin=self.window[latest_index],
                                   alpha=self.alpha)
        self.score_profile.append(k)
        # Line 7 - 11
        if len(self.window) == self.max_window_size:
            p = self.find_index_of_min_k(self.score_profile)
            self.window[p - 1](self.window[p])  # merge bin_{p-1} and bin_{p} into bin_{p-1}
            self.window.pop(p)
            self.score_profile.pop(p)
            self.update_score_profile(update_index=p - 1)

    """
      if the buffer is not empty and X_t ends, we will put buffer into window.
    """

    def insert_latest_buffer(self):
        if len(self.buffer) > 0:
            self.insert_bin_into_window()

    """
    update score profile after merging bin_{p} into bin_{p-1}.
    prior_index is {p-1}
    """

    def update_score_profile(self, update_index):
        self.score_profile[update_index] = self.compute_score(prior_bin=self.window[update_index - 1],
                                                              latest_bin=self.window[update_index],
                                                              alpha=self.alpha)
        if update_index + 1 < len(self.window):
            self.score_profile[update_index + 1] = self.compute_score(prior_bin=self.window[update_index],
                                                                      latest_bin=self.window[update_index + 1],
                                                                      alpha=self.alpha)

    def get_skectching_result(self):
        sketching_result = []
        for bin_list in self.window:
            temp = [bin_list.get_mu] * bin_list.get_n
            sketching_result = sketching_result + temp

        return sketching_result

    def get_starting_bin_index_list(self):
        start_list = []
        start = 0
        for index, bin in enumerate(self.window):
            n = int(bin.get_n)
            start_list.append(start)
            start = start + n
        return start_list

    """
    Score computation function
    """

    @staticmethod
    def compute_score(prior_bin, latest_bin, alpha):
        prior_c = prior_bin.get_c
        prior_mu = prior_bin.get_mu
        prior_n = prior_bin.get_n
        prior_e = (prior_c * prior_c) / (2 * prior_n) * log(2 / alpha)
        prior_e = sqrt(prior_e)  # Eq.1

        latest_c = latest_bin.get_c
        latest_mu = latest_bin.get_mu
        latest_n = latest_bin.get_n
        latest_e = (latest_c * latest_c) / (2 * latest_n) * log(2 / alpha)
        latest_e = sqrt(latest_e)  # Eq.1

        score = abs(prior_mu - latest_mu) / min(prior_e, latest_e)  # Eq.2
        return score

    @staticmethod
    def initialization_bin(buffer):
        latest_bin = bin_item(buffer)
        return latest_bin

    # return the index of minimum value
    @staticmethod
    def find_index_of_min_k(score_profile):
        min_k = min(score_profile[1:])
        min_index = score_profile[1:].index(min_k) + 1
        return min_index

    @property
    def get_window(self):
        return self.window


class bin_item:
    """
    bin = <n,mu>
        n denotes the length of X_{u,v} (i.e., n = u - v + 1)
        mu denotes the mean of X_{u,v}
    """

    def __init__(self, buffer):
        self.n = len(buffer)
        self.mu = mean(buffer)
        self.min_range = min(buffer)
        self.max_range = max(buffer)

    # merge bin_{prior} and bin_{current} into bin_{prior}
    def __call__(self, current_bin):
        new_n = self.get_n + current_bin.get_n  # Eq.3a
        dividend_term = (self.get_n * self.get_mu) + (
                current_bin.get_n * current_bin.get_mu)
        new_mu = dividend_term / new_n  # Eq.3b
        self.min_range = min(self.get_min_range, current_bin.get_min_range)
        self.max_range = max(self.get_max_range, current_bin.get_max_range)
        self.n = new_n
        self.mu = new_mu

    @property
    def get_n(self):
        return self.n

    @property
    def get_mu(self):
        return self.mu

    @property
    def get_min_range(self):
        return self.min_range

    @property
    def get_max_range(self):
        return self.max_range

    @property
    def get_c(self):
        return self.max_range - self.min_range
