import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KernelDensity
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution

class GaussianKDE(Distribution):  # 已经检验过与sklearn计算结果一致
    def __init__(self, X, bw, lam=1e-4, device=None):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims).to(device),
                                      covariance_matrix=torch.eye(self.dims).to(device))
        self.lam = lam

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.
        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.
        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X

        # 注意此处取log当值接近0时会产生正负无穷的数
        # 利用with autograd.detect_anomaly()检测出算法发散的原因在于torch.log变量值接近0,需要探究接近0的原因
        log_probs = torch.log(
            (self.bw ** (-self.dims) *
             torch.exp(self.mvn.log_prob((X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n + self.lam)

        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.
        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        X_chunks = self.X.split(1000)
        Y_chunks = Y.split(1000)

        log_prob = 0

        for x in X_chunks:
            for y in Y_chunks:
                log_prob += self.score_samples(y, x).sum(dim=0)

        return log_prob


class UniformLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, y_pred):
        expected_score = torch.linspace(0, self.args.confidence_margin, self.args.batch_size).cuda()
        y_pred = torch.abs(y_pred)
        loss = torch.mean(torch.abs(y_pred - expected_score[torch.argsort(y_pred, dim=0)]))
        return loss
        # return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    device = "cuda"
    x_num = 1000
    d = 1
    s_u = torch.normal(mean=0., std=torch.full([30], 1.)).cuda().unsqueeze(dim = 1)
    confidence_margin = 5
    s_u = (confidence_margin) * torch.rand(30, 1).cuda()
    n_u = s_u.size(0)
    bw_u = (n_u * (d + 2) / 4.) ** (-1. / (d + 4))
    kde_u = GaussianKDE(X=s_u, bw=bw_u, device=device)

    # the range of x axis
    xmin = torch.min(s_u)
    xmax = torch.max(s_u)

    dx = 0.2 * (xmax - xmin)
    xmin -= dx
    xmax += dx
    x = torch.linspace(xmin.detach(), xmax.detach(), x_num).to(device)

    # estimated pdf
    kde_u_x = torch.exp(kde_u.score_samples(x.reshape(-1, 1)))
    loss = torch.mean(kde_u_x)
    loss.backward()
    plt.plot(x.detach().cpu(), kde_u_x.detach().cpu(), color='blue')
    plt.savefig("./uniform_kde.pdf", format="pdf")
    
    