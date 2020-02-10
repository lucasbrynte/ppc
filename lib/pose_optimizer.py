import numpy as np
import torch
from torch import nn
from lib.expm.expm32 import expm32
from lib.expm.expm64 import expm64
from lib.utils import get_rotation_axis_angle

from torchvision.transforms.functional import normalize
from lib.constants import TV_MEAN, TV_STD

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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

def R_to_w(R):
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
    if torch.any(~zerorot_mask):
        # Close to zero - linear approximation of theta/sin(theta) is used rather than getting numerical errors.
        # NOTE: Division with sin(theta) could be replaced by normalization (final norm should be theta)
        w[~zerorot_mask,:] *= theta[~zerorot_mask,None] / torch.sin(theta[~zerorot_mask,None])
    return w

def w_to_R(w):
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

def _retrieve_input_img(image_tensor):
    img = normalize(image_tensor, mean=-TV_MEAN/TV_STD, std=1/TV_STD)
    img = torch.clamp(img / 255., 0.0, 1.0)
    img = np.moveaxis(img.numpy(), 0, -1)
    return img

class FullPosePipeline(nn.Module):
    def __init__(
        self,
        model,
        neural_rendering_wrapper,
        loss_handler,
        maps,
        HK,
        obj_id,
        ambient_weight,
    ):
        super().__init__()
        self._model = model
        self._neural_rendering_wrapper = neural_rendering_wrapper
        self._loss_handler = loss_handler
        self._maps = maps
        self._HK = HK
        self._obj_id = obj_id
        self._ambient_weight = ambient_weight

    def forward(self, t, w, R_refpt=None, fname='out.png'):
        # # Punish w
        # return torch.norm(w, dim=1)

        # # Punish theta
        # R = w_to_R(w)
        # trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
        # theta = torch.acos(0.5*(trace-1.0))
        # return theta

        R = w_to_R(w)
        if R_refpt is not None:
            R = torch.bmm(R, R_refpt)
        query_img = self._neural_rendering_wrapper.render(
            self._HK,
            R,
            t,
            self._obj_id,
            self._ambient_weight,
        )

        fig, axes_array = plt.subplots(nrows=1, ncols=2, squeeze=False)
        axes_array[0,0].imshow(_retrieve_input_img(self._maps.ref_img[0,:,:,:].detach().cpu()))
        axes_array[0,1].imshow(_retrieve_input_img(query_img[0,:,:,:].detach().cpu()))
        fig.savefig(fname)

        # # Punish pixels
        # sh = query_img.shape
        # punish_img = query_img[0,:,:sh[2]//2,:sh[3]//2]
        # # punish_img = query_img[0,:,::,:]
        # punish_img = normalize(punish_img, mean=-TV_MEAN/TV_STD, std=1/TV_STD) / 255.
        # return torch.mean(punish_img**2)
        # # return torch.mean(punish_img**2, dim=(1,2,3))

        maps = self._maps._asdict()
        maps['query_img'] = query_img
        maps = self._maps.__class__(**maps)
        nn_out = self._model((maps, None))

        pred_features_raw = self._loss_handler.get_pred_features(nn_out)
        pred_features = self._loss_handler.apply_activation(pred_features_raw)
        return pred_features['avg_reproj_err'] # (batch_size, 1)

class PoseOptimizer():
    def __init__(
        self,
        pipeline,
        R0,
        t0,
    ):
        self._pipeline = pipeline

        self._batch_size = R0.shape[0]
        self._dtype = R0.dtype
        self._device = R0.device

        # self._R0 = R0
        deg_perturb = 0.
        # deg_perturb = 5.
        # deg_perturb = 20.
        R_perturb = torch.tensor(get_rotation_axis_angle(np.array([0., 1., 0.]), deg_perturb*3.1416/180.)[:3,:3], dtype=self._dtype, device=self._device)
        self._R0 = torch.matmul(R_perturb[None,:,:], R0)

        self._t0 = t0
        self._t = nn.Parameter(self._t0.detach())

        # self._N = 10
        # self._N = 50
        self._N = 300

        self._R_refpt_mode = 'eye'
        # self._R_refpt_mode = 'R0'
        # self._R_refpt_mode = 'R_prev'

        # Optim
        self._optimize = True
        # self._w_dir = None
        self._w_dir = [1.0, 0.0, 0.0]
        self._xrange = None

        # # Plot along line
        # self._optimize = False
        # self._w_dir = [1.0, 0.0, 0.0]
        # # self._xrange = None
        # # x_delta = 0.001
        # # x_delta = 0.01
        # # x_delta = 0.1
        # x_delta = 0.5
        # # x_delta = 1.5
        # self._xrange = torch.linspace(-x_delta, x_delta, steps=self._N, dtype=self._dtype, device=self._device)
        # self._xrange = [ nn.Parameter(x) for x in self._xrange ]

        if self._R_refpt_mode == 'eye':
            w = R_to_w(self._R0.detach())
            self._R_refpt = None
        else:
            w = nn.Parameter(self._R0.new_zeros((self._R0.shape[0], 3)))
            self._R_refpt = self._R0.detach()

        if self._w_dir is None:
            self._x = w
            self._x2w = lambda x: x
            assert self._optimize
        else:
            assert self._R_refpt_mode in ('eye', 'R0')
            self._w_dir = torch.tensor(self._w_dir, dtype=self._dtype, device=self._device).detach()
            assert self._w_dir.shape == (3,)
            self._w0 = w.detach()
            self._x2w = lambda x: self._w0 + x*self._w_dir
            self._x = torch.tensor(0.0, dtype=self._dtype, device=self._device).detach()
        self._x = nn.Parameter(self._x)

        if self._R_refpt_mode in ('eye', 'R0'):
            self._optimizer = self._init_optimizer()

    def _init_optimizer(self):
        return torch.optim.SGD(
            [
                # self._t,
                self._x,
            ],
            # lr = 1.,
            # lr = 5e-1,
            # lr = 1e-2,
            # lr = 1e-3,
            lr = 1e-5,
            # lr = 1e-6,
            # lr = 1e-7,
            # lr = 1e-8,
            # lr = 4e-6,
            # lr = 0e-6,
            momentum = 0.0,
            # momentum = 0.5,
        )

    def optimize(self):
        err_est_list = []
        grads_list = []
        x_list = []
        for j in range(self._N):
            if not self._optimize:
                self._x = self._xrange[j]

            w = self._x2w(self._x)

            if self._R_refpt_mode == 'R_prev':
                w = nn.Parameter(self._R0.new_zeros((self._R0.shape[0], 3)))
                self._optimizer = self._init_optimizer()

            err_est = self._pipeline(self._t, w, R_refpt=self._R_refpt, fname='experiments/out_{:03}.png'.format(j+1))
            # print(R_refpt)
            # print(x)
            # print(w)
            print(err_est)
            self._optimizer.zero_grad()
            # Sum over batch for aggregated loss. Each term will only depend on its corresponding elements in the parameter tensors anyway.
            agg_loss = torch.sum(err_est)
            agg_loss.backward()

            # Store iterations
            x_list.append(self._x.detach().clone())
            if self._optimize:
                grads_list.append(self._x.grad.data.clone())
            else:
                grads_list.append(self._xrange[j].grad.data)

            if self._optimize:
                self._optimizer.step()

            # Store iterations
            err_est_list.append(err_est.detach())

            if self._R_refpt_mode == 'R_prev':
                self._R_refpt = torch.bmm(w_to_R(w), self._R_refpt).detach()

        grads_list = torch.stack(grads_list, dim=0)
        err_est_list = torch.stack(err_est_list, dim=0)
        x_list = torch.stack(x_list, dim=0)

        if len(x_list.shape) == 1:
            # Scalar parameter x.
            fig, axes_array = plt.subplots(nrows=2, ncols=3, squeeze=False)
            axes_array[0,0].plot(x_list.detach().cpu().numpy())
            axes_array[0,1].plot(err_est_list[:,0].detach().cpu().numpy())
            axes_array[0,2].plot(grads_list.detach().cpu().numpy())
            axes_array[1,1].plot(x_list.detach().cpu().numpy(), err_est_list[:,0].detach().cpu().numpy())
            axes_array[1,2].plot(x_list.detach().cpu().numpy(), grads_list.detach().cpu().numpy())
            fig.savefig('experiments/00_func.png')

        assert False