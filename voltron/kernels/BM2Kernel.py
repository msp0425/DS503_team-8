import torch
from torch.nn.functional import softplus
from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval
def CumTrapz(y, x):
    dx = x[..., 1] - x[..., 0]
    dx = dx if x.ndim == 1 else dx.unsqueeze(-1)
    wghts = dx * torch.ones_like(x)
    wghts[..., -1] *= 0
    return torch.cat((torch.tensor([0]).cuda(),torch.cumsum(wghts * y, -1)[:-1]), dim=-1)

class BM2Kernel(Kernel):
    has_lengthscale = False
    def __init__(self, vol=0.2, batch_shape=None, vol_constraint=None, **kwargs):

        vol_constraint = Interval(0., 1.) if not vol_constraint else vol_constraint

        if batch_shape is None:
            batch_shape = torch.Size()
            vol_size = [1]
        else:
            vol_size = [*batch_shape, 1]

        super(BM2Kernel, self).__init__(batch_shape=batch_shape, lengthscale_constraint=vol_constraint, **kwargs)
        
        self.register_parameter("raw_vol", torch.nn.Parameter(torch.zeros(*vol_size)))
        self.register_constraint("raw_vol", vol_constraint)
        self.vol = vol


    def _set_vol(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_vol)

        self.initialize(raw_vol=self.raw_vol_constraint.inverse_transform(value))

    @property
    def vol(self):
        return self.raw_vol_constraint.transform(self.raw_vol)

    @vol.setter
    def vol(self, value):
        return self._set_vol(value)
    
    def forward(self, x, y, **kwargs):
        x1s = x
        x2s = x  
        if self.batch_shape == torch.Size():
            X1, X2 = torch.meshgrid(x1s[:, 0], x2s[:, 0])
            cov1 = self.vol * torch.minimum(X1,X2)
        else:
            X1, X2 = torch.meshgrid(x1s[0,:, 0], x2s[0,:, 0])
            X1 = X1.unsqueeze(0).repeat(*self.batch_shape, 1, 1)
            X2 = X2.unsqueeze(0).repeat(*self.batch_shape, 1, 1)
            cov1 = self.vol.unsqueeze(-1) * torch.minimum(X1, X2)
        if x1s.shape[-1] == 1:
            x1s = x1s.squeeze()
        vol_int = CumTrapz(y.squeeze() * y.squeeze(), x1s)
        
        idx = torch.arange(x1s.shape[-1])
        idx1, idx2 = torch.meshgrid(idx, idx)
        idx = torch.minimum(idx1, idx2)
        cov2 = vol_int[..., idx]
        cov = cov1 + cov2
        
        diag = kwargs.pop("diag", False)
        if diag:
            return cov.diag()
        else:
            return cov
    
    