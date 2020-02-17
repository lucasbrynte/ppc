import json
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
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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
        obj_id_list,
        ambient_weight,
    ):
        super().__init__()
        self._model = model
        self._neural_rendering_wrapper = neural_rendering_wrapper
        self._loss_handler = loss_handler
        self._maps = maps
        self._HK = HK
        self._obj_id_list = obj_id_list
        self._ambient_weight = ambient_weight

    def forward(self, t, w, R_refpt=None, batch_interleaved_repeat_factor=1, fname='out.png'):
        # # Punish w
        # return torch.norm(w, dim=1)

        # # Punish theta
        # R = w_to_R(w)
        # trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
        # theta = torch.acos(0.5*(trace-1.0))
        # return theta

        HK = self._HK.repeat_interleave(batch_interleaved_repeat_factor, dim=0)
        obj_id_list = [ obj_id for obj_id in self._obj_id_list for _ in range(batch_interleaved_repeat_factor) ]

        R = w_to_R(w)
        if R_refpt is not None:
            R = torch.bmm(R, R_refpt)
        query_img = self._neural_rendering_wrapper.render(
            HK,
            R,
            t,
            obj_id_list,
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
        maps['ref_img'] = maps['ref_img'].repeat_interleave(batch_interleaved_repeat_factor, dim=0)
        maps['query_img'] = query_img
        maps = self._maps.__class__(**maps)
        nn_out = self._model((maps, None))

        pred_features_raw = self._loss_handler.get_pred_features(nn_out)
        pred_features = self._loss_handler.apply_activation(pred_features_raw)
        return pred_features['avg_reproj_err'] # (batch_size, 1)

def cross_normalized(v1,v2):
    assert v1.shape[1] == 3
    assert v2.shape[1] == 3
    v3 = torch.cross(v1, v2, dim=1)
    v3 /= v3.norm(dim=1, keepdim=True)
    return v3

def find_orthonormal(v1):
    """
    For each normalized vector in v1 batch, find an arbitrary orthonogonal and normalized vector, adn return these in new v2 batch.
    """
    bs = v1.shape[0]
    assert v1.shape == (bs, 3)
    tmp = torch.zeros_like(v1)
    min_axis_indices = v1.abs().argmin(dim=1)
    tmp[list(range(bs)),min_axis_indices] = 1.0
    return cross_normalized(v1, tmp)

class PoseOptimizer():
    def __init__(
        self,
        pipeline,
        K,
        R_gt,
        t_gt,
        R_refpt,
    ):
        self._pipeline = pipeline
        self._orig_K = K

        self._orig_batch_size = R_gt.shape[0]
        self._dtype = R_gt.dtype
        self._device = R_gt.device

        self._orig_R_gt = R_gt
        self._orig_t_gt = t_gt

        self._orig_R_refpt = R_refpt
        # # self._orig_R_refpt = torch.eye(3, dtype=self._dtype, device=self._device)[None,:,:].repeat(self._batch_size, 1, 1)
        # self._orig_R_refpt = R0.detach()
        # # self._orig_R_refpt = R0_before_perturb.detach()
        # # self._orig_R_refpt = R_gt.detach()

    def _repeat_onedim(self, T, nbr_reps, dim=0, interleave=False):
        if interleave:
            return torch.repeat_interleave(T, nbr_reps, dim=dim)
        else:
            old_shape = T.shape
            rep_def = len(old_shape) * [1]
            rep_def[dim] = nbr_reps
            return T.repeat(*rep_def)

    @property
    def _batch_size(self):
        return self._orig_batch_size * self._num_optim_runs

    @property
    def _K(self):
        return self._repeat_onedim(self._orig_K, self._num_optim_runs, dim=0, interleave=True)

    @property
    def _R_gt(self):
        return self._repeat_onedim(self._orig_R_gt, self._num_optim_runs, dim=0, interleave=True)

    @property
    def _t_gt(self):
        return self._repeat_onedim(self._orig_t_gt, self._num_optim_runs, dim=0, interleave=True)

    @property
    def _R_refpt(self):
        return self._repeat_onedim(self._orig_R_refpt, self._num_optim_runs, dim=0, interleave=True)

    def _get_w_basis(self, R0, R_gt):
        # Set the origin of the w basis to the R0 point, expressed in relation to the R_refpt.
        w_basis_origin = R_to_w(torch.matmul(R0, self._R_refpt.permute((0,2,1)))).detach()
        w_gt = R_to_w(torch.matmul(R_gt, self._R_refpt.permute((0,2,1)))).detach()
        w_perturb = w_gt - w_basis_origin
        w_perturb_norm = w_perturb.norm(dim=1)
        mask = w_perturb_norm > 1e-5
        w_basis_vec1 = torch.zeros_like(w_perturb)
        w_basis_vec1[mask,:] = w_perturb[mask,:] / w_perturb_norm[mask,None] # (batch_size, 3)
        w_basis_vec1[~mask,0] = 1.0 # If direction was undefined - might as well use x-axis
        w_basis_vec2 = find_orthonormal(w_basis_vec1)
        w_basis_vec3 = cross_normalized(w_basis_vec1, w_basis_vec2)
        w_basis = torch.stack([
            w_basis_vec1,
            w_basis_vec2,
            w_basis_vec3,
        ], dim=2) # (batch_size, 3, self._num_xdims)
        w_basis /= w_basis.norm(dim=1, keepdim=True)
        w_basis = torch.tensor(w_basis, dtype=self._dtype, device=self._device)
        return w_basis_origin, w_basis


    def _init_optimizer(self, params):
        # return torch.optim.SGD(
        #     params,
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
        #         x,
        #     ],
        #     # rho = 0.9,
        # )
        return torch.optim.Adam(
            params,
            # lr = 1e-2,
            # betas = (0.9, 0.999),
            # lr = 1e-2,
            # betas = (0.0, 0.9),
            # lr = 1e-1,
            # lr = 5e-2,
            lr = 3e-2,
            # lr = 1e-2,
            # lr = 3e-3,
            # betas = (0.95, 0.99),
            betas = (0.8, 0.9),
            # betas = (0.5, 0.99),
        )

    def _x2t(self, x):
        t = self._t_gt
        # t = self._t_gt.clone().detach().requires_grad_(True)
        return t

    def _x2w(self, x):
        w = self._w_basis_origin + torch.bmm(self._w_basis[:,:,:self._num_xdims], x[:,:,None]).squeeze(2)
        return w

    def eval_func(self, x, R_refpt=None, fname='out.png'):
        t = self._x2t(x)
        w = self._x2w(x)
        err_est = self._pipeline(t, w, batch_interleaved_repeat_factor=self._num_optim_runs, R_refpt=R_refpt, fname=fname)
        return err_est

    def eval_func_and_calc_analytical_grad(self, x, fname='out.png'):
        """
        Eval function and calculate analytical gradients
        """
        err_est = self.eval_func(x, R_refpt=self._R_refpt, fname=fname)
        # Sum over batch for aggregated loss. Each term will only depend on its corresponding elements in the parameter tensors anyway.
        agg_loss = torch.sum(err_est)
        return err_est, grad((agg_loss,), (x,))[0]

    def eval_func_and_calc_numerical_grad(self, x, step_size, fname='out.png'):
        """
        Eval function and calculate numerical gradients
        """
        nbr_params = x.shape[1]

        x1 = x
        y1 = self.eval_func(x1, R_refpt=self._R_refpt, fname=fname).squeeze(1)
        assert y1.shape == (self._batch_size,)
        grad = torch.empty_like(x)
        assert grad.shape == (self._batch_size, nbr_params)
        for x_idx in range(nbr_params):
            x2 = x.clone()
            forward_diff = 2.*(torch.rand(self._batch_size, device=self._device) < 0.5).float() - 1.
            x2[:,x_idx] += forward_diff*step_size
            y2 = self.eval_func(x2, R_refpt=self._R_refpt, fname=fname).squeeze(1)
            assert y2.shape == (self._batch_size,)
            grad[:,x_idx] = forward_diff * (y2-y1) / float(step_size)
        grad = grad.detach()
        err_est = y1
        return err_est, grad

    def calc_add_metric(self, pts_objframe, object_diameter, R_est, t_est, R_gt, t_gt):
        # NOTE: ADD can be computed more efficiently by considering relative euclidean transformations, rather than transforming to camera frame twice. Impossible to exploit this for reprojection error however.
        # # pts_camframe_gt: R_gt @ pts_objframe + t_gt
        # # pts_camframe_est: R_est @ pts_objframe + t_est
        # # pts_objframe = R_est.T @ (pts_camframe_est - t_est)
        # # pts_objframe \approx R_est.T @ (pts_camframe_gt - t_est) = R_est.T @ (R_gt @ pts_objframe + t_gt - t_est) = (R_est.T @ R_gt) @ pts_objframe + R_est.T @ (t_gt - t_est)
        # # => R_combined = R_est.T @ R_gt
        # # => t_combined = R_est.T @ (t_gt - t_est)
        # R_combined = torch.bmm(R_est.permute(0,2,1), R_gt)
        # t_combined = torch.bmm(R_est.permute(0,2,1), t_gt - t_est)

        pts_camframe_est = torch.bmm(R_est, pts_objframe) + t_est
        pts_camframe_gt = torch.bmm(R_gt, pts_objframe) + t_gt

        eps = 1e-5
        pflat = lambda pts: pts[:,:2,:] / torch.max(pts[:,[2],:], torch.tensor(eps, device=self._device))
        pts_reproj_est = pflat(torch.bmm(self._K, pts_camframe_est))
        pts_reproj_gt = pflat(torch.bmm(self._K, pts_camframe_gt))

        add_metric_unnorm = torch.mean(torch.norm(pts_camframe_est.squeeze(2) - pts_camframe_gt.squeeze(2), dim=1), dim=1) # The old dim=2 is the new dim=1
        add_metric = add_metric_unnorm / object_diameter
        return list(zip(
            add_metric.detach().cpu().numpy().tolist(),
            add_metric_unnorm.detach().cpu().numpy().tolist(),
        ))

    def calc_avg_reproj_metric(self, pts_objframe, R_est, t_est, R_gt, t_gt):
        pts_camframe_est = torch.bmm(R_est, pts_objframe) + t_est
        pts_camframe_gt = torch.bmm(R_gt, pts_objframe) + t_gt

        eps = 1e-5
        pflat = lambda pts: pts[:,:2,:] / torch.max(pts[:,[2],:], torch.tensor(eps, device=self._device))
        pts_reproj_est = pflat(torch.bmm(self._K, pts_camframe_est))
        pts_reproj_gt = pflat(torch.bmm(self._K, pts_camframe_gt))

        avg_reproj_err = torch.mean(torch.norm(pts_reproj_est.squeeze(2) - pts_reproj_gt.squeeze(2), dim=1), dim=1) # The old dim=2 is the new dim=1
        return avg_reproj_err.detach().cpu().numpy().tolist()

    def calc_deg_cm_err(self, R_est, t_est, R_gt, t_gt):
        R_rel = torch.bmm(R_est.permute(0,2,1), R_gt)
        deg_err = 180. / np.pi * R_to_w(R_rel).norm(dim=1)
        t_rel = t_gt - t_est
        cm_err = 10.*t_rel.norm(dim=1).squeeze(1) # mm -> cm
        return list(zip(
            deg_err.detach().cpu().numpy().tolist(),
            cm_err.detach().cpu().numpy().tolist(),
        ))

    def eval_pose_single_object(self, obj_id, x_est, err_est, R_gt, t_gt):
        t_est = self._x2t(x_est)
        w_est = self._x2w(x_est)
        R_est = w_to_R(w_est)
        if self._R_refpt is not None:
            R_est = torch.bmm(R_est, self._R_refpt)

        # Define batch of model points
        pts_objframe = self._pipeline._neural_rendering_wrapper._models[obj_id]['vertices'].permute(0,2,1)
        pts_objframe = pts_objframe.expand(self._batch_size,-1,-1)

        # Determine object diameter
        object_diameter = self._pipeline._neural_rendering_wrapper._models_info[obj_id]['diameter']

        metrics = [ {
            'add_metric': add_metric,
            'avg_reproj_metric': avg_reproj_metric,
            'deg_cm_err': deg_cm_err,
            'err_est': curr_err_est,
        } for (
            add_metric,
            avg_reproj_metric,
            deg_cm_err,
            curr_err_est
        ) in zip(
            self.calc_add_metric(pts_objframe, object_diameter, R_est, t_est, R_gt, t_gt),
            self.calc_avg_reproj_metric(pts_objframe, R_est, t_est, R_gt, t_gt),
            self.calc_deg_cm_err(R_est, t_est, R_gt, t_gt),
            err_est.detach().cpu().numpy().tolist(),
        ) ]
        return metrics

    def eval_pose(self, x_est, err_est, R_gt, t_gt):
        # NOTE: Assuming constant object ID. Since these methods rely on torch.expand on a single object model, the most efficient way to support multiple object IDs would probably be to define separate batches for the different objects.
        assert len(set(self._pipeline._obj_id_list)) == 1
        obj_id = self._pipeline._obj_id_list[0]
        return self.eval_pose_single_object(obj_id, x_est, err_est, R_gt, t_gt)

    def _init_scheduler(self):
        def get_cos_anneal_lr(x):
            """
            Cosine annealing.
            """
            # x_max = 30
            x_max = 50
            # y_min = 1e-1
            y_min = 1e-2
            y_min = float(y_min)
            x = float(min(x, x_max))
            return y_min + 0.5 * (1.0-y_min) * (1.0 + np.cos(x/x_max*np.pi))

        def get_exp_lr(x):
            """
            Exponential decay
            """
            half_life = 5.
            # min_reduction = 1.0
            # min_reduction = 1e-1
            min_reduction = 5e-2
            reduction = np.exp(float(x) * np.log(0.5**(1./half_life)))
            return max(reduction, min_reduction)

        return torch.optim.lr_scheduler.LambdaLR(
            self._optimizer,
            get_cos_anneal_lr,
            # get_exp_lr,
            # lambda x: 1.0,
        )

    def optimize(
            self,
            R0_before_perturb,
            t0_before_perturb,
            N = 100,
            deg_perturb = [0.],
            axis_perturb = [[0., 1., 0.]],
        ):
        self._num_optim_runs = len(deg_perturb)
        assert len(axis_perturb) == len(deg_perturb)

        get_perturb = lambda deg_perturb, axis_perturb: torch.tensor(get_rotation_axis_angle(np.array(axis_perturb), deg_perturb*3.1416/180.)[:3,:3], dtype=self._dtype, device=self._device)
        R_perturb = torch.stack([ get_perturb(curr_deg, curr_axis) for curr_deg, curr_axis in zip(deg_perturb, axis_perturb) ], dim=0)

        # NOTE: interleave=True along optim runs, and False along batch, since this allows for reshaping to (batch_size, num_optim_runs, ..., ...) in the end
        R0_before_perturb = self._repeat_onedim(R0_before_perturb, self._num_optim_runs, dim=0, interleave=True)
        t0_before_perturb = self._repeat_onedim(t0_before_perturb, self._num_optim_runs, dim=0, interleave=True)
        R_perturb = self._repeat_onedim(R_perturb, self._orig_batch_size, dim=0, interleave=False)

        R0 = torch.matmul(R_perturb, R0_before_perturb)
        t0 = t0_before_perturb

        # self._w_basis_origin = torch.zeros((self._batch_size, 3), dtype=self._dtype, device=self._device)
        # self._w_basis = np.tile(np.eye(3)[None,:,:], (self._batch_size, 1, 1))
        self._w_basis_origin, self._w_basis = self._get_w_basis(R0, self._R_gt)

        # self._num_xdims = 1
        # self._num_xdims = 2
        self._num_xdims = 3
        x = torch.zeros((self._batch_size, self._num_xdims), dtype=self._dtype, device=self._device)

        x = nn.Parameter(x)
        self._optimizer = self._init_optimizer([
            x,
        ])
        self._scheduler = self._init_scheduler()

        all_err_est = torch.empty((self._batch_size, N), dtype=self._dtype, device=self._device)
        all_grads = torch.empty((self._batch_size, N, self._num_xdims), dtype=self._dtype, device=self._device)
        all_x = torch.empty((self._batch_size, N, self._num_xdims), dtype=self._dtype, device=self._device)
        for j in range(N):
            # err_est, curr_grad = self.eval_func_and_calc_analytical_grad(x, fname='experiments/out_{:03}.png'.format(j+1))
            err_est, curr_grad = self.eval_func_and_calc_numerical_grad(x, 1e-2, fname='experiments/out_{:03}.png'.format(j+1))
            print(
                j,
                self._scheduler.get_lr(),
                err_est.detach().cpu().numpy(),
                x.detach().cpu().numpy(),
                curr_grad.detach().cpu().numpy(),
            )
            x.grad = curr_grad

            # Store iterations
            all_x[:,j,:] = x.detach().clone()
            all_grads[:,j,:] = x.grad.clone()

            self._optimizer.step()
            self._scheduler.step()

            # Store iterations
            all_err_est[:,j] = err_est.detach()

        best_iter = torch.argmin(all_err_est, dim=1)
        best_err_est = torch.min(all_err_est, dim=1)[0]
        first_err_est = all_err_est[:,0]
        last_err_est = all_err_est[:,-1]
        best_x = torch.stack([ all_x[:,:,x_idx][list(range(self._batch_size)),best_iter] for x_idx in range(self._num_xdims) ], dim=1)
        first_x = all_x[:,0,:]
        last_x = all_x[:,-1,:]
        assert best_iter.shape == (self._batch_size,)
        assert best_x.shape == (self._batch_size, self._num_xdims)
        assert first_x.shape == (self._batch_size, self._num_xdims)
        assert last_x.shape == (self._batch_size, self._num_xdims)
        best_metrics = self.eval_pose(best_x, best_err_est, self._R_gt, self._t_gt)
        first_metrics = self.eval_pose(first_x, first_err_est, self._R_gt, self._t_gt)
        last_metrics = self.eval_pose(last_x, last_err_est, self._R_gt, self._t_gt)
        print(json.dumps(best_metrics, indent=4))
        print(json.dumps(first_metrics, indent=4))
        print(json.dumps(last_metrics, indent=4))

        def plot(sample_idx, fname):
            # Scalar parameter x.
            nrows = 1
            if self._num_xdims == 2:
                nrows += 1
            fig, axes_array = plt.subplots(nrows=nrows, ncols=3, squeeze=False)
            axes_array[0,0].plot(all_x[sample_idx,:,:].detach().cpu().numpy())
            axes_array[0,1].plot(all_err_est[sample_idx,:].detach().cpu().numpy())
            axes_array[0,2].plot(all_grads[sample_idx,:,:].detach().cpu().numpy())
            if self._num_xdims == 2:
                axes_array[1,0].plot(all_x[sample_idx,:,0].detach().cpu().numpy(), all_x[sample_idx,:,1].detach().cpu().numpy())
            fig.savefig(fname)

        for sample_idx in range(self._batch_size):
            fname = 'experiments/00_func_sample{:02d}_run{:02d}.png'.format(sample_idx//self._num_optim_runs, sample_idx%self._num_optim_runs)
            plot(sample_idx, fname)

        assert False

    def evaluate(self, calc_grad=False):
        self._num_optim_runs = 1
        self._w_basis_origin = torch.zeros((self._batch_size, 3), dtype=self._dtype, device=self._device)
        self._w_basis = np.tile(np.eye(3)[None,:,:], (self._batch_size, 1, 1))
        # self._w_basis_origin, self._w_basis = self._get_w_basis(R0, self._R_gt)

        def vec(T, N_each):
            N = np.prod(N_each)
            old_shape = list(T.shape)
            assert np.all(np.array(old_shape[-self._num_xdims:]) == np.array(N_each))
            new_shape = old_shape[:-self._num_xdims] + [N]
            return T.view(new_shape)
        def unvec(T, N_each):
            N = np.prod(N_each)
            old_shape = list(T.shape)
            assert old_shape[-1] == N
            new_shape = old_shape[:-1] + list(N_each)
            return T.view(new_shape)
        # self._num_xdims = 1
        self._num_xdims = 2
        x = torch.zeros((self._batch_size, self._num_xdims), dtype=self._dtype, device=self._device)

        # N_each = [4] * self._num_xdims
        # N_each = [10] * self._num_xdims
        # N_each = [2] * self._num_xdims
        # N_each = [7] * self._num_xdims
        # N_each = [10] * self._num_xdims
        N_each = [20] * self._num_xdims
        # N_each = [25] * self._num_xdims
        # N_each = [40] * self._num_xdims
        # N_each = [100] * self._num_xdims
        # N_each = [300] * self._num_xdims

        N = np.prod(N_each)

        # Plot along line
        # x_delta = 0.001
        # x_delta = 0.01
        # x_delta = 0.1
        # x_delta = 0.5
        x_delta = 1.5

        x_range_limits = [ (-x_delta, x_delta) for x_idx in range(self._num_xdims) ]

        def get_range(a, b, N):
            return torch.linspace(a, b, steps=N, dtype=self._dtype, device=self._device, requires_grad=True)
        x_ranges = [ get_range(limits[0], limits[1], N_each[x_idx]) for x_idx, limits in zip(range(self._num_xdims), x_range_limits) ]
        all_x = torch.stack(
            torch.meshgrid(*x_ranges),
            dim=0,
        ) # shape: (x_idx, idx0, idx1, idx2, ...). If x_idx=0, then the values are the ones corresponding to idx0
        all_x = torch.stack(self._batch_size*[all_x], dim=0)

        # all_x = torch.linspace(-x_delta, x_delta, steps=N, dtype=self._dtype, device=self._device, requires_grad=True)[None,:,None].repeat(self._batch_size, 1, 1)
        # # self._xgrid = torch.meshgrid(*(self._num_xdims*[all_x]))

        all_err_est = torch.empty([self._batch_size]+N_each, dtype=self._dtype, device=self._device)
        if calc_grad:
            all_grads = torch.empty([self._batch_size, self._num_xdims]+N_each, dtype=self._dtype, device=self._device)
        for j in range(N):
            x = vec(all_x, N_each)[:,:,j] # shape: (sample_idx, x_idx)

            if calc_grad:
                # err_est, curr_grad = self.eval_func_and_calc_analytical_grad(x, fname='experiments/out_{:03}.png'.format(j+1))
                err_est, curr_grad = self.eval_func_and_calc_numerical_grad(x, 1e-2, fname='experiments/out_{:03}.png'.format(j+1))
            else:
                err_est = self.eval_func(x, R_refpt=self._R_refpt, fname='experiments/out_{:03}.png'.format(j+1))
                err_est = err_est.squeeze(1)
            print(
                j,
                err_est.detach().cpu().numpy(),
                x.detach().cpu().numpy(),
                curr_grad.detach().cpu().numpy() if calc_grad else None,
            )
            if calc_grad:
                x.grad = curr_grad

            # Store iterations
            vec(all_x, N_each)[:,:,j] = x.detach().clone()
            if calc_grad:
                vec(all_grads, N_each)[:,:,j] = x.grad.clone()

            # Store iterations
            vec(all_err_est, N_each)[:,j] = err_est.detach()

        def plot_surf(axes_array, j, k, all_x, mapvals):
            fig.delaxes(axes_array[j,k])
            axes_array[j,k] = fig.add_subplot(nrows, ncols, j*ncols+k+1, projection='3d')
            axes_array[j,k].plot_surface(
                all_x[0,:,:],
                all_x[1,:,:],
                mapvals,
            )
            axes_array[j,k].set_xlabel('x1')
            axes_array[j,k].set_ylabel('x2')

        def plot_heatmap(axes_array, j, k, mapvals):
            axes_array[j,k].imshow(
                np.flipud(mapvals.T),
                extent = [
                    x_range_limits[0][0] - 0.5*(x_range_limits[0][1]-x_range_limits[0][0]) / (N_each[0]-1),
                    x_range_limits[0][1] + 0.5*(x_range_limits[0][1]-x_range_limits[0][0]) / (N_each[0]-1),
                    x_range_limits[1][0] - 0.5*(x_range_limits[1][1]-x_range_limits[1][0]) / (N_each[1]-1),
                    x_range_limits[1][1] + 0.5*(x_range_limits[1][1]-x_range_limits[1][0]) / (N_each[1]-1),
                ],
            )
            axes_array[j,k].set_xlabel('x1')
            axes_array[j,k].set_ylabel('x2')

        sample_idx = 0
        # Scalar parameter x.
        if self._num_xdims == 1:
            nrows = 1
            ncols = 1
            if calc_grad:
                ncols += 1
            fig, axes_array = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
            axes_array[0,0].plot(all_x[sample_idx,:,:].detach().cpu().numpy().T, all_err_est[sample_idx,:].detach().cpu().numpy())
            if calc_grad:
                axes_array[0,1].plot(all_x[sample_idx,:,:].detach().cpu().numpy().T, all_grads[sample_idx,:,:].detach().cpu().numpy().T)
        elif self._num_xdims == 2:
            nrows = 2
            ncols = 1
            if calc_grad:
                ncols += 1
            fig, axes_array = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
            plot_surf(axes_array, 0, 0, all_x[sample_idx,:,:,:].detach().cpu().numpy(), all_err_est[sample_idx,:,:].detach().cpu().numpy())
            plot_heatmap(axes_array, 1, 0, all_err_est[sample_idx,:,:].detach().cpu().numpy())
            if calc_grad:
                plot_surf(axes_array, 0, 1, all_x[sample_idx,:,:,:].detach().cpu().numpy(), all_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())
                plot_heatmap(axes_array, 1, 1, all_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())

        fig.savefig('experiments/00_func.png')

        assert False
