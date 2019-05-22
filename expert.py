import numpy as np

class Sampler():
    def __init__(self, batch_size):
        self.obs = np.loadtxt('trajectory/obs.txt')
        self.act = np.loadtxt('trajectory/act.txt')
        self.batch_size = batch_size
        self.ptr = 0

    def sample(self):
        nbatches = self.obs.shape[0] // self.batch_size
        assert  nbatches > 0, "expert data not enough"
        start = self.ptr
        end = self.ptr+1
        self.ptr = (self.ptr +1) % nbatches
        return self.obs[start:end, :], self.act[start:end, :]
