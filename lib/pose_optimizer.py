import numpy as np
import torch
from torch import nn
from torch.autograd import grad
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
        # deg_perturb = 0.
        # deg_perturb = 5.
        # deg_perturb = 10.
        deg_perturb = 15.
        # deg_perturb = 20.
        R_perturb = torch.tensor(get_rotation_axis_angle(np.array([0., 1., 0.]), deg_perturb*3.1416/180.)[:3,:3], dtype=self._dtype, device=self._device)
        self._R0 = torch.matmul(R_perturb[None,:,:], R0)

        self._t0 = t0
        self._x2t = lambda x: self._t0
        # self._x2t = lambda x: self._t0.clone().detach().requires_grad_(True)

        # self._N = 10
        # self._N = 50
        self._N = 100
        # self._N = 300

        self._R_refpt_mode = 'eye'
        # self._R_refpt_mode = 'R0'

        # Optim
        self._optimize = True
        self._w_dir = None
        # self._w_dir = [1.0, 0.0, 0.0]
        # self._w_dir = [0.0, 1.0, 0.0]
        # self._w_dir = [0.0, 0.0, 1.0]
        self._xrange = None

        # # Plot along line
        # self._optimize = False
        # # self._w_dir = [1.0, 0.0, 0.0]
        # # self._w_dir = [0.0, 1.0, 0.0]
        # self._w_dir = [0.0, 0.0, 1.0]
        # # self._xrange = None
        # # x_delta = 0.001
        # # x_delta = 0.01
        # # x_delta = 0.1
        # # x_delta = 0.5
        # x_delta = 1.5
        # self._xrange = torch.linspace(-x_delta, x_delta, steps=self._N, dtype=self._dtype, device=self._device, requires_grad=True)[:,None,None].repeat(1, self._batch_size, 1)
        # # self._xrange = torch.linspace(-0.06, -0.03, steps=self._N, dtype=self._dtype, device=self._device)[:,None,None].repeat(1, self._batch_size, 1)
        # # self._xrange = torch.linspace(-4e-2-5e-9, -4e-2-2.5e-9, steps=self._N, dtype=self._dtype, device=self._device)[:,None,None].repeat(1, self._batch_size, 1)
        # self._xrange = list(self._xrange)

        if self._R_refpt_mode == 'eye':
            w = R_to_w(self._R0.detach())
            self._R_refpt = None
        else:
            w = self._R0.new_zeros((self._R0.shape[0], 3))
            self._R_refpt = self._R0.detach()

        if self._w_dir is None:
            self._x = w
            self._x2w = lambda x: x
            assert self._optimize
        else:
            x_perturb = 0.0
            # x_perturb = 0.5
            self._w_dir = torch.tensor(self._w_dir, dtype=self._dtype, device=self._device)[None,:].repeat(self._batch_size, 1)
            self._w0 = w.detach()
            self._x2w = lambda x: self._w0 + x*self._w_dir
            self._x = torch.tensor([x_perturb], dtype=self._dtype, device=self._device)[None,:].repeat(self._batch_size, 1)
            assert self._w0.shape == (self._batch_size, 3)
            assert self._w_dir.shape == (self._batch_size, 3)
            assert self._x.shape == (self._batch_size, 1)
        self._x = nn.Parameter(self._x)
        self._optimizer = self._init_optimizer()

    def _init_optimizer(self):
        # return torch.optim.SGD(
        #     [
        #         self._x,
        #     ],
        #     # lr = 1.,
        #     # lr = 5e-1,
        #     # lr = 1e-2,
        #     # lr = 1e-3,
        #     lr = 1e-4,
        #     # lr = 1e-5,
        #     # lr = 1e-6,
        #     # lr = 1e-7,
        #     # lr = 1e-8,
        #     # lr = 4e-6,
        #     # lr = 0e-6,
        #     # momentum = 0.0,
        #     momentum = 0.5,
        #     # momentum = 0.9,
        # )
        # return torch.optim.Adadelta(
        #     [
        #         self._x,
        #     ],
        #     # rho = 0.9,
        # )
        return torch.optim.Adam(
            [
                self._x,
            ],
            # lr = 1e-2,
            # betas = (0.9, 0.999),
            # lr = 1e-2,
            # betas = (0.0, 0.9),
            lr = 1e-2,
            betas = (0.5, 0.99),
        )

    def eval_func(self, x, R_refpt=None, fname='out.png'):
        t = self._x2t(x)
        w = self._x2w(x)
        err_est = self._pipeline(t, w, R_refpt=R_refpt, fname=fname)
        return err_est

    def eval_func_and_calc_analytical_grad(self, fname='out.png'):
        """
        Eval function and calculate analytical gradients
        """
        err_est = self.eval_func(self._x, R_refpt=self._R_refpt, fname=fname)
        # Sum over batch for aggregated loss. Each term will only depend on its corresponding elements in the parameter tensors anyway.
        agg_loss = torch.sum(err_est)
        return err_est, grad((agg_loss,), (self._x,))[0]

    def eval_func_and_calc_numerical_grad(self, step_size, fname='out.png'):
        """
        Eval function and calculate numerical gradients
        """
        nbr_params = self._x.shape[1]

        x1 = self._x
        y1 = self.eval_func(x1, R_refpt=self._R_refpt, fname=fname)
        grad = torch.empty_like(self._x)
        assert grad.shape == (self._batch_size, nbr_params)
        for x_idx in range(nbr_params):
            x2 = self._x.clone()
            forward_diff = np.random.random() < 0.5
            x2[:,x_idx] += float(forward_diff)*step_size
            y2 = self.eval_func(x2, R_refpt=self._R_refpt, fname=fname).squeeze(1)
            assert y2.shape == (self._batch_size,)
            grad[:,x_idx] = float(forward_diff) * (y2-y1) / float(step_size)
        grad = grad.detach()
        err_est = y1
        return err_est, grad

    def optimize(self):
        err_est_list = []
        grads_list = []
        x_list = []
        for j in range(self._N):
            if not self._optimize:
                self._x = self._xrange[j]

            # err_est, curr_grad = self.eval_func_and_calc_analytical_grad(fname='experiments/out_{:03}.png'.format(j+1))
            err_est, curr_grad = self.eval_func_and_calc_numerical_grad(1e-2, fname='experiments/out_{:03}.png'.format(j+1))
            print(
                j,
                err_est.detach().cpu().numpy(),
                self._x.detach().cpu().numpy(),
                curr_grad.detach().cpu().numpy(),
            )
            self._x.grad = curr_grad
            # self._optimizer.zero_grad()
            # agg_loss.backward()

            # Store iterations
            x_list.append(self._x.detach().clone())
            if self._optimize:
                grads_list.append(self._x.grad.clone())
            else:
                grads_list.append(self._xrange[j].grad)

            if self._optimize:
                self._optimizer.step()

            # Store iterations
            err_est_list.append(err_est.detach())

        grads_list = torch.stack(grads_list, dim=0)
        err_est_list = torch.stack(err_est_list, dim=0)
        x_list = torch.stack(x_list, dim=0)

        sample_idx = 0
        # Scalar parameter x.
        fig, axes_array = plt.subplots(nrows=2, ncols=3, squeeze=False)
        axes_array[0,0].plot(x_list[:,sample_idx,:].detach().cpu().numpy())
        axes_array[0,1].plot(err_est_list[:,sample_idx].detach().cpu().numpy())
        
        # Some printouts for detecting actual steps in function (constant floating point numbers)
        # tmp = err_est_list[:,sample_idx].detach().cpu().numpy()
        # print('{:e}'.format(tmp.min()))
        # print('{:e}'.format(tmp.max()))
        # print(np.sum(tmp==tmp[0]))
        # print(np.sum(tmp!=tmp[0]))
        # print(np.sum(tmp==tmp[-1]))
        
        axes_array[0,2].plot(grads_list[:,sample_idx,:].detach().cpu().numpy())
        axes_array[1,1].plot(x_list[:,sample_idx,:].detach().cpu().numpy(), err_est_list[:,sample_idx].detach().cpu().numpy())
        axes_array[1,2].plot(x_list[:,sample_idx,:].detach().cpu().numpy(), grads_list[:,sample_idx,:].detach().cpu().numpy())
        fig.savefig('experiments/00_func.png')

        assert False
