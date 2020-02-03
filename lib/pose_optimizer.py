import torch
from lib.expm.expm32 import expm32
from lib.expm.expm64 import expm64

def expm_frechet(A, E, expm):
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=A.dtype, device=A.device, requires_grad=False)
    M[:n, :n] = A
    M[n:, n:] = A
    M[:n, n:] = E
    return expm(M)[:n, n:]

class expm_class(torch.autograd.Function):
    @staticmethod
    def _expm_func(A):
        if A.element_size() > 4:
            return expm64
        else:
            return expm32

    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        expm = expm_class._expm_func(A)
        return expm(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        expm = expm_class._expm_func(A)
        return expm_frechet(A.t(), G, expm)
expm = expm_class.apply

class PoseOptimizer():
    def __init__(
        self,
        model,
        neural_rendering_wrapper,
        loss_handler,
        maps,
        HK,
        R0,
        t0,
        obj_id,
        ambient_weight,
    ):
        self._model = model
        self._neural_rendering_wrapper = neural_rendering_wrapper
        self._loss_handler = loss_handler
        self._maps = maps
        self._HK = HK
        self._R0 = R0
        self._t0 = t0
        self._t = self._t0.detach()
        self._w = self._R_to_w(self._R0.detach())
        self._obj_id = obj_id
        self._ambient_weight = ambient_weight

    def _R_to_w(self, R):
        """
        https://en.wikipedia.org/wiki/Axis–angle_representation#Log_map_from_SO(3)_to_'"%60UNIQ--postMath-0000000D-QINU%60"'(3)
        R: (batch_size, 3, 3)
        w: (batch_size, 3)
        """
        trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
        theta = torch.acos(0.5*(trace-1.0))
        zerorot_mask = theta < 1e-5
        w = 0.5 * torch.stack([
            R[:,2,1] - R[:,1,2],
            R[:,0,2] - R[:,2,0],
            R[:,1,0] - R[:,0,1],
        ], dim=1)
        if torch.any(zerorot_mask):
            # Close to zero - linear approximation of theta/sin(theta) is used rather than getting numerical errors.
            w[~zerorot_mask,:] *= theta[~zerorot_mask,None] / torch.sin(theta[~zerorot_mask,None])
        return w

    def _w_to_R(self, w):
        """
        w: (batch_size, 3)
        R: (batch_size, 3, 3)
        """
        batch_size = w.shape[0]
        A = w.new_zeros((batch_size, 3, 3)) # Same dtype & device as w
        A[:,1,2] = -w[:,0]
        A[:,0,2] = w[:,1]
        A[:,0,1] = -w[:,2]
        A -= A.permute((0,2,1))
        R = torch.stack([expm(A_curr) for A_curr in A], dim=0)
        # R = expm(A)
        return R

    def forward(self, t, w):
        self._maps.query_img[:,:,:,:] = self._neural_rendering_wrapper.render(
            self._HK,
            self._w_to_R(w),
            t,
            self._obj_id,
            self._ambient_weight,
        )
        nn_out = self._model((self._maps, None))
        pred_features_raw = self._loss_handler.get_pred_features(nn_out)
        pred_features = self._loss_handler.apply_activation(pred_features_raw)
        self._err_est = pred_features['avg_reproj_err'] # (batch_size, 1)

    def backward(self, t, w):
        # TODO: clear gradients via optimizer.
        # TODO: recompute gradients.
        pass

    def step(self):
        self.forward(self._t, self._w)
        # NOTE: forward method is done - output stored in self._err_est.
        # TODO: Initialize optimizer.
        # TODO: torch "Parameters" vs tensors as inputs?
        # TODO: Impl & call backward.
        # TODO: Dummy experiments: Initialize at or close to GT. Batch size 1?

    def optimize(self):
        self.step()
