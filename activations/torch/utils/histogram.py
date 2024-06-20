import torch
import scipy.stats as sts



def get_bin_size(min, max):
    """Computes the number `x` such that `x` fits `100` times into `max - min`."""
    dist = 1/int(1/(max - min))
    bin_size = dist * 0.01

    return bin_size


class Histogram:
    def __init__(self, bin_size=None, device="cpu"):
        if bin_size is None:
            self.auto_bin_size = True
            self.bin_size = None
        else:
            self.auto_bin_size = False
            self.bin_size = bin_size
        self.bins = torch.empty(0, device=device)
        self.counts = torch.empty(0, device=device)

    @property
    def weights(self):
        return self.counts / self.counts.sum()

    def fill_n(self, input):
        input = input.to(self.bins)  # cast to correct dtype/device

        if self.auto_bin_size:
            self.bin_size = get_bin_size(input.min(), input.max())
            self.auto_bin_size = False

        left_edge, right_edge = min, max + self.bin_size/2
        if len(self.bins) > 0:
            left_edge = torch.minimum(left_edge, self.bins[0])
            right_edge = torch.maximum(right_edge, self.bins[-1])

        new_bins = torch.arange(left_edge, right_edge, self.bin_size, device=self.bins.device)
        new_counts = torch.histogram(input, self.bins, density=False).hist

        if len(new_counts) == len(self.counts):  # no new bins added
            self.counts += new_counts
        else:  # find indices to insert `self.counts`
            n = len(self.bins)  # len(self.bins) <= len(bins)
            if new_bins[0] != self.bins[0]:
                idx0 = torch.any(self.bins == new_bins[:n]).nonzero()[0]
            if new_bins[-1] != self.bins[0]:
                idx1 = torch.any(self.bins == new_bins[-n:]).nonzero()[0]

            new_counts[idx0:idx1+1] += self.counts
            self.counts = new_counts
        self.bins = new_bins
    
    def kde(self, bw_method=0.13797296614612148):
        return sts.gaussian_kde(self.bins, bw_method=bw_method, weights=self.weights).pdf

    