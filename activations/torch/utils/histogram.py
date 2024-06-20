import torch
import scipy.stats as sts



def get_bin_size(min, max):
    """Computes the number `x` such that `x` fits `100` times into `max - min`."""
    dist = 1/int(1/(max - min))
    bin_size = dist * 0.01

    return bin_size


class NeuronsHistogram:
    def __init__(self, bin_size=None, device="cpu"):
        if bin_size is None:
            self.auto_bin_size = True
            self.bin_size = None
        else:
            self.auto_bin_size = False
            self.bin_size = bin_size

        self.device = device
        self.bins = None
        self.counts = None
        self.n_neurons = None

    def initialize(self, n_neurons=None):
        self.n_neurons = n_neurons
        self.bins = [torch.empty(0, device=self.device) for _ in range(n_neurons)]
        self.counts = [torch.empty(0, device=self.device) for _ in range(n_neurons)]

    @property
    def weights(self):
        return [self.counts[n] / self.counts[n].sum() for n in range(self.n_neurons)]

    def _fill_n(self, n, input):
        left_edge, right_edge = input.min(), input.max() + self.bin_size/2
        if len(self.bins) > 0:
            left_edge = torch.minimum(left_edge, self.bins[n][0])
            right_edge = torch.maximum(right_edge, self.bins[n][-1])

        new_bins = torch.arange(left_edge, right_edge, self.bin_size, device=self.device)
        new_counts = torch.histogram(input, self.bins[n], density=False).hist

        if len(new_counts) == len(self.counts):  # no new bins added
            self.counts[n] += new_counts
        else:  # find indices to insert `self.counts`
            m = len(self.bins[n])  # len(self.bins) <= len(bins)
            if new_bins[n][0] != self.bins[n][0]:
                idx0 = torch.any(self.bins[n] == new_bins[n][:m]).nonzero()[0]
            if new_bins[n][-1] != self.bins[n][0]:
                idx1 = torch.any(self.bins[n] == new_bins[n][-m:]).nonzero()[0]

            new_counts[idx0:idx1+1] += self.counts[n]
            self.counts[n] = new_counts
        self.bins[n] = new_bins

    def fill_n(self, input):
        if self.n_neurons is None:  # hist is uninitialized
            self.initialize(input.shape[0])

        if self.auto_bin_size:
            self.bin_size = get_bin_size(input.min(), input.max())
            self.auto_bin_size = False

        input = input.to(self.bins[0])  # move to correct device/dtype
        for n in range(self.n_neurons):
            self._fill_n(n, input[n].view(-1))
    
    def kde(self, n, bw_method=0.13797296614612148):
        return sts.gaussian_kde(self.bins[n], bw_method=bw_method, weights=self.weights[n]).pdf

    
class Histogram(NeuronsHistogram):
    def __init__(self, bin_size=None, device="cpu"):
        super().__init__(bin_size, device)

    def initialize(self):
        super().initialize(n_neurons=1)

    def fill_n(self, input):
        super().fill_n(input.view(1, -1))

    @property
    def weights(self):
        return super().weights[0]
    
    def kde(self, n, bw_method=0.13797296614612148):
        return super().kde(0, bw_method)
        
    