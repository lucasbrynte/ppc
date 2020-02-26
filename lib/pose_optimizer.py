import os
import json
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from lib.expm.expm32 import expm32
from lib.expm.expm64 import expm64
from lib.utils import get_rotation_axis_angle, order_dict

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
        configs,
        model,
        neural_rendering_wrapper,
        loss_handler,
        maps,
        HK,
        obj_id_list,
        ambient_weight,
    ):
        super().__init__()
        self._configs = configs
        self._model = model
        self._neural_rendering_wrapper = neural_rendering_wrapper
        self._loss_handler = loss_handler
        self._maps = maps
        self._HK = HK
        self._obj_id_list = obj_id_list
        self._ambient_weight = ambient_weight

        self._out_path = os.path.join(self._configs.experiment_path, 'eval_poseopt')

    def forward(self, t, w, R_refpt=None, batch_interleaved_repeat_factor=1, fname_dict={}):
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

        ref_img = self._maps.ref_img.repeat_interleave(batch_interleaved_repeat_factor, dim=0)

        for sample_idx, fname in fname_dict.items():
            fig, axes_array = plt.subplots(nrows=1, ncols=2, squeeze=False)
            axes_array[0,0].imshow(_retrieve_input_img(ref_img[sample_idx,:,:,:].detach().cpu()))
            axes_array[0,1].imshow(_retrieve_input_img(query_img[sample_idx,:,:,:].detach().cpu()))
            full_fpath = os.path.join(self._out_path, fname)
            os.makedirs(os.path.dirname(full_fpath), exist_ok=True)
            fig.savefig(full_fpath)

        # # Punish pixels
        # sh = query_img.shape
        # punish_img = query_img[0,:,:sh[2]//2,:sh[3]//2]
        # # punish_img = query_img[0,:,::,:]
        # punish_img = normalize(punish_img, mean=-TV_MEAN/TV_STD, std=1/TV_STD) / 255.
        # return torch.mean(punish_img**2)
        # # return torch.mean(punish_img**2, dim=(1,2,3))

        maps = self._maps._asdict()
        maps['ref_img'] = ref_img
        maps['query_img'] = query_img
        maps = self._maps.__class__(**maps)
        nn_out = self._model((maps, None))

        pred_features_raw = self._loss_handler.get_pred_features(nn_out)
        pred_features = self._loss_handler.apply_activation(pred_features_raw)
        interp_pred_features = self._loss_handler.calc_human_interpretable_features(pred_features)
        return interp_pred_features

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
        configs,
        pipeline,
        K,
        HK,
        R_gt,
        t_gt,
        R_refpt,
        ref_img_path,
        numerical_grad = True,
    ):
        self._configs = configs
        self._pipeline = pipeline
        self._orig_K = K
        self._orig_HK = HK
        self._orig_HK_inv = torch.inverse(self._orig_HK)
        self._ref_img_path = ref_img_path
        self._numerical_grad = numerical_grad

        self._orig_batch_size = R_gt.shape[0]
        self._dtype = R_gt.dtype
        self._device = R_gt.device

        self._orig_R_gt = R_gt
        self._orig_t_gt = t_gt

        self._out_path = os.path.join(self._configs.experiment_path, 'eval_poseopt')

    def _repeat_onedim(self, T, nbr_reps, dim=0, interleave=False):
        if interleave:
            return torch.repeat_interleave(T, nbr_reps, dim=dim)
        else:
            old_shape = T.shape
            rep_def = len(old_shape) * [1]
            rep_def[dim] = nbr_reps
            return T.repeat(*rep_def)

    @property
    def _num_optim_runs(self):
        return len(self._optim_runs)

    @property
    def _batch_size(self):
        return self._orig_batch_size * self._num_optim_runs

    @property
    def _K(self):
        return self._repeat_onedim(self._orig_K, self._num_optim_runs, dim=0, interleave=True)

    @property
    def _HK(self):
        return self._repeat_onedim(self._orig_HK, self._num_optim_runs, dim=0, interleave=True)

    @property
    def _HK_inv(self):
        return self._repeat_onedim(self._orig_HK_inv, self._num_optim_runs, dim=0, interleave=True)

    @property
    def _R_gt(self):
        return self._repeat_onedim(self._orig_R_gt, self._num_optim_runs, dim=0, interleave=True)

    @property
    def _t_gt(self):
        return self._repeat_onedim(self._orig_t_gt, self._num_optim_runs, dim=0, interleave=True)

    def _get_canonical_w_basis(self):
        w_basis = torch.eye(3, dtype=self._dtype, device=self._device)[None,:,:].repeat(self._batch_size, 1, 1)
        return w_basis

    def _get_w_basis(self, primary_w_dir=None):
        if primary_w_dir is None:
            w_basis = torch.eye(3, dtype=self._dtype, device=self._device)[None,:,:].repeat(self._batch_size, 1, 1)
            return w_basis
        primary_w_dir_norm = primary_w_dir.norm(dim=1)
        mask = primary_w_dir_norm > 1e-5
        w_basis_vec1 = torch.zeros_like(primary_w_dir)
        w_basis_vec1[mask,:] = primary_w_dir[mask,:] / primary_w_dir_norm[mask,None] # (batch_size, 3)
        w_basis_vec1[~mask,0] = 1.0 # If direction was undefined - might as well use x-axis
        w_basis_vec2 = find_orthonormal(w_basis_vec1)
        w_basis_vec3 = cross_normalized(w_basis_vec1, w_basis_vec2)
        w_basis = torch.stack([
            w_basis_vec1,
            w_basis_vec2,
            w_basis_vec3,
        ], dim=2) # (batch_size, 3, self._num_wxdims)
        w_basis /= w_basis.norm(dim=1, keepdim=True)
        w_basis = w_basis.detach()
        return w_basis

    def _get_u_basis(self):
        u_basis = torch.eye(2, dtype=self._dtype, device=self._device)[None,:,:].repeat(self._batch_size, 1, 1)
        return u_basis

    def _backproject_pixels(self, u):
        assert u.shape == (self._batch_size, 2, 1)
        ux = u[:,[0],:]
        uy = u[:,[1],:]
        uz = torch.ones_like(ux)
        u = torch.cat((ux, uy, uz), dim=1)
        u_normalized = torch.bmm(self._HK_inv, u)
        t_normalized_depth = u_normalized / u_normalized[:,[2],:]
        t = t_normalized_depth * self._ref_depth[:,None,None]
        return t

    def _x2t_inplane(self, tx):
        # t_inplane is a 2D translation from t_origin, with preserved depth.
        # Assuming t base vectors to be in this plane, e.g. x-axis & y-axis.
        if self._num_txdims > 0:
            u = self._u_basis_origin + torch.bmm(self._u_basis[:,:,:self._num_txdims], tx[:,:,None])
        else:
            assert self._num_txdims == 0
            u = self._u_basis_origin
        t = self._backproject_pixels(u)
        return t

    def _x2t_vr(self, t_inplane, desired_depth):
        # Viewing ray of t_inplane. Unit length in depth direction.
        viewing_ray = t_inplane / t_inplane[:,[-1],:]

        # Realize the desired depth by translation along viewing ray, thus preserving projection of object center.
        current_depth = t_inplane[:,[-1],:]
        t_vr = viewing_ray * (desired_depth - current_depth)

        return t_vr

    def _x2t(self, tx, d):
        t_inplane = self._x2t_inplane(tx)
        assert self._num_ddims in (0, 1)
        assert len(d.shape) == 2 and d.shape[1] == self._num_ddims
        d0 = self._d_origin.clone()
        if self._num_ddims == 1:
            d0 *= torch.exp(d)
        t_vr = self._x2t_vr(t_inplane, d0[:,:,None])
        return t_inplane + t_vr

    def _x2w(self, wx):
        w = self._w_basis_origin.clone()
        if self._num_wxdims > 0:
            w = w + torch.bmm(self._w_basis[:,:,:self._num_wxdims], wx[:,:,None]).squeeze(2)
        return w

    def eval_func(self, wx, tx, d, R_refpt=None, fname_dict={}):
        t = self._x2t(tx, d)
        w = self._x2w(wx)
        pred_features = self._pipeline(t, w, batch_interleaved_repeat_factor=self._num_optim_runs, R_refpt=R_refpt, fname_dict=fname_dict)
        return pred_features

    def eval_func_and_calc_analytical_grad(self, wx, tx, d, fname_dict={}):
        """
        Eval function and calculate analytical gradients
        """
        pred_features = self.eval_func(wx, tx, d, R_refpt=self._R_refpt, fname_dict=fname_dict)
        err_est = pred_features['avg_reproj_err']
        pixel_offset_est = pred_features['pixel_offset']
        rel_depth_est = pred_features['rel_depth_error']
        # Sum over batch for aggregated loss. Each term will only depend on its corresponding elements in the parameter tensors anyway.
        agg_loss = torch.sum(err_est)
        wx_grad, tx_grad = grad((agg_loss,), (x,))
        return err_est, wx_grad, tx_grad

    def eval_func_and_calc_numerical_wx_grad(self, wx1, tx, d, y1, step_size, x_indices=None):
        """
        Eval function and calculate numerical gradients
        """
        nbr_params = wx1.shape[1]
        if x_indices is None:
            x_indices = list(range(nbr_params))
        assert y1.shape == (self._batch_size,)
        grad = torch.zeros_like(wx1)
        assert grad.shape == (self._batch_size, nbr_params)
        for x_idx in x_indices:
            wx2 = wx1.clone()
            forward_diff = 2.*(torch.rand(self._batch_size, device=self._device) < 0.5).float() - 1.
            wx2[:,x_idx] += forward_diff*step_size
            pred_features = self.eval_func(wx2, tx, d, R_refpt=self._R_refpt, fname_dict={})
            y2 = pred_features['avg_reproj_err'].squeeze(1)
            assert y2.shape == (self._batch_size,)
            grad[:,x_idx] = forward_diff * (y2-y1) / float(step_size)
        grad = grad.detach()
        return grad

    def eval_func_and_calc_numerical_tx_grad(self, wx, tx1, d, y1, step_size, x_indices=None):
        """
        Eval function and calculate numerical gradients
        """
        nbr_params = tx1.shape[1]
        if x_indices is None:
            x_indices = list(range(nbr_params))
        assert y1.shape == (self._batch_size,)
        grad = torch.zeros_like(tx1)
        assert grad.shape == (self._batch_size, nbr_params)
        for x_idx in x_indices:
            tx2 = tx1.clone()
            forward_diff = 2.*(torch.rand(self._batch_size, device=self._device) < 0.5).float() - 1.
            tx2[:,x_idx] += forward_diff*step_size
            pred_features = self.eval_func(wx, tx2, d, R_refpt=self._R_refpt, fname_dict={})
            y2 = pred_features['avg_reproj_err'].squeeze(1)
            assert y2.shape == (self._batch_size,)
            grad[:,x_idx] = forward_diff * (y2-y1) / float(step_size)
        grad = grad.detach()
        return grad

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

        pts_camframe_est = torch.matmul(R_est, pts_objframe) + t_est
        pts_camframe_gt = torch.matmul(R_gt, pts_objframe) + t_gt

        add_metric_unnorm = torch.mean(torch.norm(pts_camframe_est.squeeze(3) - pts_camframe_gt.squeeze(3), dim=2), dim=2) # The old dim=3 is the new dim=2
        add_metric = add_metric_unnorm / object_diameter
        return np.stack([
            add_metric.detach().cpu().numpy(),
            add_metric_unnorm.detach().cpu().numpy(),
        ], axis=2)

    def calc_avg_reproj_metric(self, K, pts_objframe, R_est, t_est, R_gt, t_gt):
        pts_camframe_est = torch.matmul(R_est, pts_objframe) + t_est
        pts_camframe_gt = torch.matmul(R_gt, pts_objframe) + t_gt

        eps = 1e-5
        pflat = lambda pts: pts[:,:,:2,:] / torch.max(pts[:,:,[2],:], torch.tensor(eps, device=self._device))
        pts_reproj_est = pflat(torch.matmul(K, pts_camframe_est))
        pts_reproj_gt = pflat(torch.matmul(K, pts_camframe_gt))

        avg_reproj_err = torch.mean(torch.norm(pts_reproj_est.squeeze(3) - pts_reproj_gt.squeeze(3), dim=2), dim=2) # The old dim=3 is the new dim=2
        return avg_reproj_err.detach().cpu().numpy()

    def calc_deg_cm_err(self, R_est, t_est, R_gt, t_gt):
        R_rel = torch.matmul(R_est.permute(0,1,3,2), R_gt)

        N = R_est.shape[1]
        R_to_w_wrapper = lambda R: torch.stack([ R_to_w(R[:,j,:,:]) for j in range(N) ], dim=1)

        deg_err = 180. / np.pi * R_to_w_wrapper(R_rel).norm(dim=2)
        t_rel = t_gt - t_est
        cm_err = 1e-1*t_rel.norm(dim=2).squeeze(2) # mm -> cm
        return np.stack([
            deg_err.detach().cpu().numpy(),
            cm_err.detach().cpu().numpy(),
        ], axis=2)

    def eval_pose_single_object(self, obj_id, wx_est, tx_est, d_est, err_est, R_gt, t_gt):
        R_gt = R_gt[:,None,:,:]
        t_gt = t_gt[:,None,:,:]

        N = err_est.shape[1]
        t_est = torch.stack(
            [ self._x2t(tx_est[:,j,:], d_est[:,j,:]) for j in range(N) ],
            dim=1,
        )
        R_est = torch.stack(
            [ w_to_R(self._x2w(wx_est[:,j,:])) for j in range(N) ],
            dim=1,
        )
        if self._R_refpt is not None:
            R_est = torch.matmul(R_est, self._R_refpt[:,None,:,:])

        # Define batch of model points
        pts_objframe = self._pipeline._neural_rendering_wrapper._models[obj_id]['vertices'].permute(0,2,1)
        pts_objframe = pts_objframe[:,None,:,:] # Extra dimension for iterations
        pts_objframe = pts_objframe.expand(self._batch_size,-1,-1,-1)

        # Determine object diameter
        object_diameter = self._pipeline._neural_rendering_wrapper._models_info[obj_id]['diameter']

        add_metrics = self.calc_add_metric(pts_objframe, object_diameter, R_est, t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N, 2)
        avg_reproj_metrics = self.calc_avg_reproj_metric(self._K[:,None,:,:], pts_objframe, R_est, t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N)
        avg_reproj_HK_metrics = self.calc_avg_reproj_metric(self._HK[:,None,:,:], pts_objframe, R_est, t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N)
        deg_cm_errors = self.calc_deg_cm_err(R_est, t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N, 2)
        err_est_numpy = err_est.detach().cpu().numpy().reshape(self._orig_batch_size, len(self._optim_runs), N)
        R_est_numpy = R_est.detach().cpu().numpy().reshape(self._orig_batch_size, len(self._optim_runs), N, 3, 3)
        t_est_numpy = t_est.detach().cpu().numpy().reshape(self._orig_batch_size, len(self._optim_runs), N, 3, 1)
        metrics = [ {
            'ref_img_path': ref_img_path,
            'optim_runs': self._optim_runs,
            'optim_run_names_sorted': list(self._optim_runs.keys()),
            'R_est': curr_R_est.tolist(),
            't_est': curr_t_est.tolist(),
            'metrics': {
                'add_metric': add_metric.tolist(),
                'avg_reproj_metric': avg_reproj_metric.tolist(),
                'avg_reproj_HK_metric': avg_reproj_HK_metric.tolist(),
                'deg_cm_err': deg_cm_err.tolist(),
                'err_est': curr_err_est.tolist(),
            },
        } for (
            ref_img_path,
            add_metric,
            avg_reproj_metric,
            avg_reproj_HK_metric,
            deg_cm_err,
            curr_err_est,
            curr_R_est,
            curr_t_est,
        ) in zip(
            self._ref_img_path,
            add_metrics,
            avg_reproj_metrics,
            avg_reproj_HK_metrics,
            deg_cm_errors,
            err_est_numpy,
            R_est_numpy,
            t_est_numpy,
        ) ]
        return metrics

    def eval_pose(self, all_wx, all_tx, all_d, all_err_est):
        # NOTE: Assuming constant object ID. Since these methods rely on torch.expand on a single object model, the most efficient way to support multiple object IDs would probably be to define separate batches for the different objects.
        assert len(set(self._pipeline._obj_id_list)) == 1
        obj_id = self._pipeline._obj_id_list[0]
        all_metrics = self.eval_pose_single_object(obj_id, all_wx, all_tx, all_d, all_err_est, self._R_gt, self._t_gt)
        return all_metrics

    def store_eval(self, metrics):
        img_dir, img_fname = os.path.split(metrics['ref_img_path'])
        seq, rgb_dir = os.path.split(img_dir)
        assert rgb_dir == 'rgb'
        json_fname = '.'.join([img_fname.split('.')[0], 'json'])
        os.makedirs(os.path.join(self._out_path, 'evaluation', seq), exist_ok=True)
        with open(os.path.join(self._out_path, 'evaluation', seq, json_fname), 'w') as f:
            json.dump(metrics, f)

    def _init_cos_transition_scheduler(
        self,
        optimizer,
        zero_before = None,
        x_min = 0,
        # x_max = 30,
        x_max = 50,
        # y_min = 1e-1,
        y_min = 1e-2,
        y_max = 1.0,
    ):
        y_min = float(y_min)
        assert x_max > x_min
        def get_cos_anneal_lr(x):
            """
            Cosine annealing.
            """
            if zero_before is not None and x < zero_before:
                return 0.0
            x = max(x, x_min)
            x = min(x, x_max)
            x = float(x)
            return y_min + 0.5 * (y_max-y_min) * (1.0 + np.cos((x-x_min)/(x_max-x_min)*np.pi))

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            get_cos_anneal_lr,
        )

    # def _init_exp_scheduler(self, optimizer):
    #     def get_exp_lr(x):
    #         """
    #         Exponential decay
    #         """
    #         half_life = 5.
    #         # min_reduction = 1.0
    #         # min_reduction = 1e-1
    #         min_reduction = 5e-2
    #         reduction = np.exp(float(x) * np.log(0.5**(1./half_life)))
    #         return max(reduction, min_reduction)
    # 
    #     return torch.optim.lr_scheduler.LambdaLR(
    #         optimizer,
    #         get_exp_lr,
    #     )

    def _init_constant_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: 1.0,
        )

    def optimize(
            self,
            R0_before_perturb,
            t0_before_perturb,
            num_wxdims = 3,
            num_txdims = 2,
            num_ddims = 1,
            N = 100,
            optim_runs = {
                'default': {'deg_perturb': 0., 'axis_perturb': [0., 1., 0.], 't_perturb': [0., 0., 0.], 'u_perturb': [0., 0.]},
            },
            enable_plotting = False,
            print_iterates = False,
            store_eval = False,
        ):
        self._num_wxdims = num_wxdims
        self._num_txdims = num_txdims
        self._num_ddims = num_ddims
        self._num_params = self._num_wxdims + self._num_txdims + self._num_ddims

        self._optim_runs = order_dict(optim_runs)
        deg_perturb = np.array([ run_spec['deg_perturb'] for run_spec in self._optim_runs.values() ])
        axis_perturb = np.array([ run_spec['axis_perturb'] for run_spec in self._optim_runs.values() ])
        t_perturb_spec = np.array([ run_spec['t_perturb'] for run_spec in self._optim_runs.values() ])
        u_perturb_spec = np.array([ run_spec['u_perturb'] for run_spec in self._optim_runs.values() ])
        d_perturb_spec = np.array([ run_spec['d_perturb'] for run_spec in self._optim_runs.values() ])
        assert deg_perturb.shape == (self._num_optim_runs,)
        assert axis_perturb.shape == (self._num_optim_runs, 3)
        assert t_perturb_spec.shape == (self._num_optim_runs, 3)
        axis_perturb /= np.linalg.norm(axis_perturb, axis=1, keepdims=True)

        get_R_perturb = lambda deg_perturb, axis_perturb: torch.tensor(get_rotation_axis_angle(np.array(axis_perturb), deg_perturb*3.1416/180.)[:3,:3], dtype=self._dtype, device=self._device)
        R_perturb = torch.stack([ get_R_perturb(curr_deg, curr_axis) for curr_deg, curr_axis in zip(deg_perturb, axis_perturb) ], dim=0)
        get_t_perturb = lambda t_perturb: torch.tensor(t_perturb, dtype=self._dtype, device=self._device).reshape((3,1))
        t_perturb = torch.stack([ get_t_perturb(curr_t) for curr_t in t_perturb_spec ], dim=0)
        get_u_perturb = lambda u_perturb: torch.tensor(u_perturb, dtype=self._dtype, device=self._device).reshape((2,1))
        u_perturb = torch.stack([ get_u_perturb(curr_u) for curr_u in u_perturb_spec ], dim=0)
        get_d_perturb = lambda d_perturb: torch.tensor(d_perturb, dtype=self._dtype, device=self._device).reshape((1,))
        d_perturb = torch.stack([ get_d_perturb(curr_d) for curr_d in d_perturb_spec ], dim=0)

        # NOTE: interleave=True along optim runs, and False along batch, since this allows for reshaping to (batch_size, num_optim_runs, ..., ...) in the end
        R0_before_perturb = self._repeat_onedim(R0_before_perturb, self._num_optim_runs, dim=0, interleave=True)
        t0_before_perturb = self._repeat_onedim(t0_before_perturb, self._num_optim_runs, dim=0, interleave=True)
        R_perturb = self._repeat_onedim(R_perturb, self._orig_batch_size, dim=0, interleave=False)
        t_perturb = self._repeat_onedim(t_perturb, self._orig_batch_size, dim=0, interleave=False)
        u_perturb = self._repeat_onedim(u_perturb, self._orig_batch_size, dim=0, interleave=False)
        d_perturb = self._repeat_onedim(d_perturb, self._orig_batch_size, dim=0, interleave=False)

        R0 = torch.matmul(R_perturb, R0_before_perturb)
        t0_before_u_perturb = (t0_before_perturb + t_perturb).detach()
        self._R_refpt = R0.clone()

        # Set the origin of the w basis to the R0 point, expressed in relation to the R_refpt.
        self._w_basis_origin = R_to_w(torch.matmul(R0, self._R_refpt.permute((0,2,1)))).detach()
        if self._num_wxdims < 3:
            w_gt = R_to_w(torch.matmul(self._R_gt, self._R_refpt.permute((0,2,1)))).detach()
            self._w_basis = self._get_w_basis(primary_w_dir = w_gt-self._w_basis_origin)
        else:
            self._w_basis = self._get_w_basis(primary_w_dir = None)
        self._ref_depth = t0_before_u_perturb[:,2,:].squeeze(1)
        self._u_basis_origin = torch.bmm(self._HK, t0_before_u_perturb)
        self._u_basis_origin = self._u_basis_origin[:,:2,:] / self._u_basis_origin[:,[2],:]
        self._u_basis = self._get_u_basis()
        self._d_origin = t0_before_u_perturb.squeeze(2)[:,[2]]

        # Apply u & d perturbs
        self._u_basis_origin += u_perturb
        self._d_origin *= d_perturb

        wx = torch.zeros((self._batch_size, self._num_wxdims), dtype=self._dtype, device=self._device)
        tx = torch.zeros((self._batch_size, self._num_txdims), dtype=self._dtype, device=self._device)
        d = torch.zeros((self._batch_size, self._num_ddims), dtype=self._dtype, device=self._device)

        wx = nn.Parameter(wx)
        tx = nn.Parameter(tx)
        d = nn.Parameter(d)
        self._wx_optimizer = torch.optim.Adam(
            [
                wx,
            ],
            # lr = 1e-1,
            lr = 7e-2, # better than 3e-2 for max50..?
            # lr = 5e-2,
            # lr = 3e-2, # best for nomax50..?
            # lr = 1e-2,
            # betas = (0.95, 0.99),
            betas = (0.8, 0.9),
        )
        self._tx_optimizer = torch.optim.SGD(
            [
                tx,
            ],
            lr = 1.0,
            momentum = 0.7,
        )
        self._d_optimizer = torch.optim.SGD(
            [
                d,
            ],
            lr = 1.0,
            momentum = 0.1,
        )
        if self._num_txdims > 0:
            nbr_iter_tx_leap = 2
        else:
            nbr_iter_tx_leap = 0
        nbr_iter_tx_leap = nbr_iter_tx_leap
        final_finetune_iter = nbr_iter_tx_leap + 50
        self._wx_scheduler = self._init_cos_transition_scheduler(
            self._wx_optimizer,
            zero_before = nbr_iter_tx_leap,
            x_min = nbr_iter_tx_leap,
            x_max = final_finetune_iter,
            # y_min = 1e-1,
            y_min = 1e-2,
            y_max = 1.0,
        )
        self._tx_scheduler = self._init_cos_transition_scheduler(
            self._tx_optimizer,
            zero_before = nbr_iter_tx_leap,
            x_min = nbr_iter_tx_leap,
            x_max = final_finetune_iter,
            y_min = 1e-1,
            # y_min = 4e-2,
            # y_min = 1e-2,
            y_max = 1.0,
        )
        self._d_scheduler = self._init_cos_transition_scheduler(
            self._d_optimizer,
            zero_before = nbr_iter_tx_leap,
            x_min = final_finetune_iter,
            x_max = final_finetune_iter+10,
            # y_min = 1.0,
            y_min = 1e-1,
            # y_min = 4e-2,
            # y_min = 1e-2,
            y_max = 1.0,
        )
        # self._wx_scheduler = self._init_constant_scheduler(self._wx_optimizer)
        # self._tx_scheduler = self._init_constant_scheduler(self._tx_optimizer)
        # self._d_scheduler = self._init_constant_scheduler(self._d_optimizer)
        # self._d_scheduler = torch.optim.lr_scheduler.LambdaLR(self._d_optimizer, lambda k: 1.0 if k < nbr_iter_tx_leap else 0.0)

        step_size_wx = 1e-2
        step_size_tx = 1.0 # 1 px

        all_err_est = torch.empty((self._batch_size, N), dtype=self._dtype, device=self._device)
        all_wx_grads = torch.empty((self._batch_size, N, self._num_wxdims), dtype=self._dtype, device=self._device)
        all_tx_grads = torch.empty((self._batch_size, N, self._num_txdims), dtype=self._dtype, device=self._device)
        all_exp_avgs = torch.zeros((self._batch_size, N, self._num_params), dtype=self._dtype, device=self._device)
        all_exp_avg_sqs = torch.zeros((self._batch_size, N, self._num_params), dtype=self._dtype, device=self._device)
        all_wx = torch.empty((self._batch_size, N, self._num_wxdims), dtype=self._dtype, device=self._device)
        all_tx = torch.empty((self._batch_size, N, self._num_txdims), dtype=self._dtype, device=self._device)
        all_d = torch.empty((self._batch_size, N, self._num_ddims), dtype=self._dtype, device=self._device)

        for j in range(N):
            if self._numerical_grad:
                pred_features = self.eval_func(wx, tx, d, R_refpt = self._R_refpt, fname_dict = { (sample_idx*self._num_optim_runs + run_idx): 'rendered_iterations/sample{:02}/optim_run_{:s}/iter{:03}.png'.format(sample_idx, run_name, j+1) for sample_idx in range(self._orig_batch_size) for run_idx, run_name in enumerate(self._optim_runs.keys()) } if enable_plotting else {})
                err_est = pred_features['avg_reproj_err'].squeeze(1)
                pixel_offset_est = pred_features['pixel_offset']
                rel_depth_est = pred_features['rel_depth_error']
                if j >= nbr_iter_tx_leap:
                    curr_wx_grad = self.eval_func_and_calc_numerical_wx_grad(wx, tx, d, err_est, step_size_wx)
                else:
                    curr_wx_grad = torch.zeros_like(wx)
                curr_tx_grad = self.eval_func_and_calc_numerical_tx_grad(wx, tx, d, err_est, step_size_tx)
            else:
                err_est, curr_wx_grad, curr_tx_grad = self.eval_func_and_calc_analytical_grad(wx, tx, d, fname_dict = { (sample_idx*self._num_optim_runs + run_idx): 'rendered_iterations/sample{:02}/optim_run_{:s}/iter{:03}.png'.format(sample_idx, run_name, j+1) for sample_idx in range(self._orig_batch_size) for run_idx, run_name in enumerate(self._optim_runs.keys()) } if enable_plotting else {})
            if print_iterates:
                print(
                    j,
                    self._wx_scheduler.get_lr(),
                    self._tx_scheduler.get_lr(),
                    self._d_scheduler.get_lr(),
                    err_est.detach().cpu().numpy(),
                    pixel_offset_est.detach().cpu().numpy(),
                    rel_depth_est.detach().cpu().numpy(),
                    wx.detach().cpu().numpy(),
                    tx.detach().cpu().numpy(),
                    d.detach().cpu().numpy(),
                    curr_wx_grad.detach().cpu().numpy(),
                    curr_tx_grad.detach().cpu().numpy(),
                )
            if self._num_wxdims > 0:
                wx.grad = curr_wx_grad[:,:self._num_wxdims]
            if self._num_txdims > 0:
                if self._num_txdims == 2 and j < nbr_iter_tx_leap:
                    # Take a leap
                    tx -= pixel_offset_est[:,:]
                elif j >= nbr_iter_tx_leap:
                    tx.grad = curr_tx_grad[:,-self._num_txdims:]
            if self._num_ddims == 1:
                d.grad = torch.log(rel_depth_est[:,:])

            if j > 0:
                exp_avg_wx = self._wx_optimizer.state[wx]['exp_avg'] if 'exp_avg' in self._wx_optimizer.state[wx] else torch.zeros_like(wx)
                exp_avg_sq_wx = self._wx_optimizer.state[wx]['exp_avg_sq'] if 'exp_avg_sq' in self._wx_optimizer.state[wx] else torch.zeros_like(wx)
                exp_avg_tx = self._tx_optimizer.state[tx]['exp_avg'] if 'exp_avg' in self._tx_optimizer.state[tx] else torch.zeros_like(tx)
                exp_avg_sq_tx = self._tx_optimizer.state[tx]['exp_avg_sq'] if 'exp_avg_sq' in self._tx_optimizer.state[tx] else torch.zeros_like(tx)
                exp_avg_d = self._d_optimizer.state[d]['exp_avg'] if 'exp_avg' in self._d_optimizer.state[d] else torch.zeros_like(d)
                exp_avg_sq_d = self._d_optimizer.state[d]['exp_avg_sq'] if 'exp_avg_sq' in self._d_optimizer.state[d] else torch.zeros_like(d)
                exp_avg = torch.cat((exp_avg_wx, exp_avg_tx, exp_avg_d), dim=1)
                exp_avg_sq = torch.cat((exp_avg_sq_wx, exp_avg_tx, exp_avg_d), dim=1)

            # Store iterations
            all_wx[:,j,:] = wx.detach().clone()
            all_tx[:,j,:] = tx.detach().clone()
            all_d[:,j,:] = d.detach().clone()
            all_wx_grads[:,j,:] = curr_wx_grad.detach().clone()
            all_tx_grads[:,j,:] = curr_tx_grad.detach().clone()
            if j > 0:
                all_exp_avgs[:,j,:] = exp_avg.detach().clone()
                all_exp_avg_sqs[:,j,:] = exp_avg_sq.detach().clone()

            self._wx_optimizer.step()
            self._tx_optimizer.step()
            self._d_optimizer.step()
            self._wx_scheduler.step()
            self._tx_scheduler.step()
            self._d_scheduler.step()

            # Store iterations
            all_err_est[:,j] = err_est.detach()

        if store_eval:
            all_metrics = self.eval_pose(all_wx, all_tx, all_d, all_err_est)
            for metrics in all_metrics:
                # print(json.dumps(metrics, indent=4))
                self.store_eval(metrics)

        def plot(sample_idx, fname):
            # Scalar parameter x.
            nrows = 1
            nrows += 1
            fig, axes_array = plt.subplots(nrows=nrows, ncols=3, squeeze=False)
            axes_array[0,0].plot(all_x[sample_idx,:,:].detach().cpu().numpy())
            axes_array[0,1].plot(all_err_est[sample_idx,:].detach().cpu().numpy())
            # axes_array[0,2].plot(all_wx_grads[sample_idx,:,:].detach().cpu().numpy())
            # # axes_array[0,2].plot(all_tx_grads[sample_idx,:,:].detach().cpu().numpy())
            axes_array[1,2].plot(all_exp_avgs[sample_idx,:,:].abs().detach().cpu().numpy())
            axes_array[1,2].plot(all_exp_avg_sqs[sample_idx,:,:].abs().detach().cpu().numpy())
            # if self._num_wxdims+self._num_txdims == 2:
            #     axes_array[1,0].plot(all_x[sample_idx,:,0].detach().cpu().numpy(), all_x[sample_idx,:,1].detach().cpu().numpy())
            #     # axes_array[1,1].plot(np.diff(all_x[sample_idx,:,:].detach().cpu().numpy(), axis=0))
            full_fpath = os.path.join(self._out_path, fname)
            os.makedirs(os.path.dirname(full_fpath), exist_ok=True)
            fig.savefig(full_fpath)

        if enable_plotting:
            all_x = torch.cat([all_wx, all_tx], dim=2)
            for sample_idx in range(self._orig_batch_size):
                for run_idx, run_name in enumerate(self._optim_runs.keys()):
                    fname = 'optimplots/sample{:02d}_optim_run_{:s}.png'.format(sample_idx, run_name)
                    plot(sample_idx*self._num_optim_runs + run_idx, fname)

    def evaluate(
        self,
        # num_wxdims = 2,
        # num_txdims = 0,
        # num_ddims = 0,
        num_wxdims = 0,
        num_txdims = 2,
        num_ddims = 0,
        N_each = [20, 20],
        calc_grad=False,
    ):
        self._num_wxdims = num_wxdims
        self._num_txdims = num_txdims
        self._num_ddims = num_ddims
        self._num_params = self._num_wxdims + self._num_txdims + self._num_ddims

        self._optim_runs = { 'dflt_run': None } # Dummy placeholder. len() == 1 -> proper behavior of various methods

        self._R_refpt = self._R_gt.clone()

        self._w_basis_origin = torch.zeros((self._batch_size, 3), dtype=self._dtype, device=self._device)
        self._w_basis = self._get_w_basis(primary_w_dir = None)
        self._ref_depth = self._t_gt[:,2,:].squeeze(1)
        t0 = self._t_gt.detach()
        self._u_basis_origin = torch.bmm(self._HK, t0)
        self._u_basis_origin = self._u_basis_origin[:,:2,:] / self._u_basis_origin[:,[2],:]
        self._u_basis = self._get_u_basis()
        self._d_origin = t0.squeeze(2)[:,[2]]

        def vec(T, N_each):
            N = np.prod(N_each)
            old_shape = list(T.shape)
            assert np.all(np.array(old_shape[-self._num_params:]) == np.array(N_each))
            new_shape = old_shape[:-self._num_params] + [N]
            return T.view(new_shape)
        def unvec(T, N_each):
            N = np.prod(N_each)
            old_shape = list(T.shape)
            assert old_shape[-1] == N
            new_shape = old_shape[:-1] + list(N_each)
            return T.view(new_shape)
        assert len(N_each) == self._num_params
        N = np.prod(N_each)

        # Plot along line
        # param_delta = 0.001
        # param_delta = 0.01
        # param_delta = 0.15
        param_delta = 0.5
        # param_delta = 1.5
        # param_delta = 150.

        param_range_limits = [ (-param_delta, param_delta) for x_idx in range(self._num_params) ]

        def get_range(a, b, N):
            return torch.linspace(a, b, steps=N, dtype=self._dtype, device=self._device, requires_grad=True)
        param_ranges = [ get_range(limits[0], limits[1], N_each[x_idx]) for x_idx, limits in enumerate(param_range_limits) ]
        all_params = torch.stack(
            torch.meshgrid(*param_ranges),
            dim=0,
        ) # shape: (x_idx, idx0, idx1, idx2, ...). If x_idx=0, then the values are the ones corresponding to idx0
        all_params = torch.stack(self._batch_size*[all_params], dim=0)

        step_size_wx = 1e-2
        step_size_tx = 3e-3

        all_err_est = torch.empty([self._batch_size]+N_each, dtype=self._dtype, device=self._device)
        all_pixel_offset_est = torch.empty([self._batch_size]+N_each, dtype=self._dtype, device=self._device)
        all_pixel_offset_x_est = torch.empty([self._batch_size]+N_each, dtype=self._dtype, device=self._device)
        all_pixel_offset_y_est = torch.empty([self._batch_size]+N_each, dtype=self._dtype, device=self._device)
        all_rel_depth_est = torch.empty([self._batch_size]+N_each, dtype=self._dtype, device=self._device)
        if calc_grad:
            all_wx_grads = torch.empty([self._batch_size, self._num_wxdims]+N_each, dtype=self._dtype, device=self._device)
            all_tx_grads = torch.empty([self._batch_size, self._num_txdims]+N_each, dtype=self._dtype, device=self._device)
        for j in range(N):
            param_vec = vec(all_params, N_each)[:,:,j] # shape: (sample_idx, x_idx)
            wx = param_vec[:,:self._num_wxdims]
            tx = param_vec[:,self._num_wxdims:self._num_wxdims+self._num_txdims]
            d = param_vec[:,-self._num_ddims:]

            if calc_grad:
                if self._numerical_grad:
                    pred_features = self.eval_func(wx, tx, d, R_refpt = self._R_refpt, fname_dict = { (sample_idx*self._num_optim_runs + run_idx): 'rendered_iterations/sample{:02}/optim_run_{:s}/iter{:03}.png'.format(sample_idx, run_name, j+1) for sample_idx in range(self._orig_batch_size) for run_idx, run_name in enumerate(self._optim_runs.keys()) })
                    err_est = pred_features['avg_reproj_err'].squeeze(1)
                    pixel_offset_est = pred_features['pixel_offset']
                    rel_depth_est = pred_features['rel_depth_error']
                    curr_wx_grad = self.eval_func_and_calc_numerical_wx_grad(wx, tx, d, err_est, step_size_wx)
                    curr_tx_grad = self.eval_func_and_calc_numerical_tx_grad(wx, tx, d, err_est, step_size_tx)
                else:
                    err_est, curr_wx_grad, curr_tx_grad = self.eval_func_and_calc_analytical_grad(wx, tx, d, fname_dict = { (sample_idx*self._num_optim_runs + run_idx): 'rendered_iterations/sample{:02}/optim_run_{:s}/iter{:03}.png'.format(sample_idx, run_name, j+1) for sample_idx in range(self._orig_batch_size) for run_idx, run_name in enumerate(self._optim_runs.keys()) })
            else:
                pred_features = self.eval_func(wx, tx, d, R_refpt=self._R_refpt, fname_dict = { (sample_idx*self._num_optim_runs + run_idx): 'rendered_iterations/sample{:02}/optim_run_{:s}/iter{:03}.png'.format(sample_idx, run_name, j+1) for sample_idx in range(self._orig_batch_size) for run_idx, run_name in enumerate(self._optim_runs.keys()) })
                err_est = pred_features['avg_reproj_err'].squeeze(1)
                pixel_offset_est = pred_features['pixel_offset']
                rel_depth_est = pred_features['rel_depth_error']
            print(
                j,
                err_est.detach().cpu().numpy(),
                pixel_offset_est.detach().cpu().numpy(),
                rel_depth_est.detach().cpu().numpy(),
                wx.detach().cpu().numpy(),
                tx.detach().cpu().numpy(),
                d.detach().cpu().numpy(),
                curr_wx_grad.detach().cpu().numpy() if calc_grad else None,
                curr_tx_grad.detach().cpu().numpy() if calc_grad else None,
            )

            # Store iterations
            # vec(all_params, N_each)[:,:,j] = x.detach().clone()
            if calc_grad:
                vec(all_wx_grads, N_each)[:,:,j] = curr_wx_grad
                vec(all_tx_grads, N_each)[:,:,j] = curr_tx_grad

            # Store iterations
            vec(all_err_est, N_each)[:,j] = err_est.detach()
            vec(all_pixel_offset_est, N_each)[:,j] = pixel_offset_est.norm(dim=1).detach()
            vec(all_pixel_offset_x_est, N_each)[:,j] = pixel_offset_est[:,0].detach()
            vec(all_pixel_offset_y_est, N_each)[:,j] = pixel_offset_est[:,1].detach()
            vec(all_rel_depth_est, N_each)[:,j] = rel_depth_est.squeeze(1).detach()

        def plot_surf(axes_array, j, k, all_params, mapvals, title=None):
            fig.delaxes(axes_array[j,k])
            axes_array[j,k] = fig.add_subplot(nrows, ncols, j*ncols+k+1, projection='3d')
            axes_array[j,k].plot_surface(
                all_params[0,:,:],
                all_params[1,:,:],
                mapvals,
            )
            axes_array[j,k].set_xlabel('x1')
            axes_array[j,k].set_ylabel('x2')
            if title is not None:
                axes_array[j,k].set_title(title)

        def plot_heatmap(axes_array, j, k, mapvals, title=None):
            axes_array[j,k].imshow(
                np.flipud(mapvals.T),
                extent = [
                    param_range_limits[0][0] - 0.5*(param_range_limits[0][1]-param_range_limits[0][0]) / (N_each[0]-1),
                    param_range_limits[0][1] + 0.5*(param_range_limits[0][1]-param_range_limits[0][0]) / (N_each[0]-1),
                    param_range_limits[1][0] - 0.5*(param_range_limits[1][1]-param_range_limits[1][0]) / (N_each[1]-1),
                    param_range_limits[1][1] + 0.5*(param_range_limits[1][1]-param_range_limits[1][0]) / (N_each[1]-1),
                ],
            )
            axes_array[j,k].set_xlabel('x1')
            axes_array[j,k].set_ylabel('x2')
            if title is not None:
                axes_array[j,k].set_title(title)

        sample_idx = 0
        # Scalar parameter x.
        if self._num_params == 1:
            nrows = 1
            ncols = 1
            ncols += 4
            if calc_grad:
                ncols += 1
            fig, axes_array = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
            axes_array[0,0].plot(all_params[sample_idx,:,:].detach().cpu().numpy().T, all_err_est[sample_idx,:].detach().cpu().numpy())
            axes_array[0,0].set_title('all_err_est')
            axes_array[0,1].plot(all_params[sample_idx,:,:].detach().cpu().numpy().T, all_rel_depth_est[sample_idx,:].detach().cpu().numpy())
            axes_array[0,1].set_title('all_rel_depth_est')
            axes_array[0,2].plot(all_params[sample_idx,:,:].detach().cpu().numpy().T, all_pixel_offset_est[sample_idx,:].detach().cpu().numpy())
            axes_array[0,2].set_title('all_pixel_offset_est')
            axes_array[0,3].plot(all_params[sample_idx,:,:].detach().cpu().numpy().T, all_pixel_offset_x_est[sample_idx,:].detach().cpu().numpy())
            axes_array[0,3].set_title('all_pixel_offset_x_est')
            axes_array[0,4].plot(all_params[sample_idx,:,:].detach().cpu().numpy().T, all_pixel_offset_y_est[sample_idx,:].detach().cpu().numpy())
            axes_array[0,4].set_title('all_pixel_offset_y_est')
            # if calc_grad:
            #     axes_array[0,-1].plot(all_params[sample_idx,:,:].detach().cpu().numpy().T, all_wx_grads[sample_idx,:,:].detach().cpu().numpy().T)
            #     # axes_array[0,-1].plot(all_params[sample_idx,:,:].detach().cpu().numpy().T, all_tx_grads[sample_idx,:,:].detach().cpu().numpy().T)
        elif self._num_params == 2:
            nrows = 2
            ncols = 1
            ncols += 4
            if calc_grad:
                ncols += 1
            fig, axes_array = plt.subplots(figsize=(15,6), nrows=nrows, ncols=ncols, squeeze=False)
            plot_surf(axes_array, 0, 0, all_params[sample_idx,:,:,:].detach().cpu().numpy(), all_err_est[sample_idx,:,:].detach().cpu().numpy(), title='all_err_est')
            plot_heatmap(axes_array, 1, 0, all_err_est[sample_idx,:,:].detach().cpu().numpy())
            plot_surf(axes_array, 0, 1, all_params[sample_idx,:,:,:].detach().cpu().numpy(), all_rel_depth_est[sample_idx,:,:].detach().cpu().numpy(), title='all_rel_depth_est')
            plot_heatmap(axes_array, 1, 1, all_rel_depth_est[sample_idx,:,:].detach().cpu().numpy())
            plot_surf(axes_array, 0, 2, all_params[sample_idx,:,:,:].detach().cpu().numpy(), all_pixel_offset_est[sample_idx,:,:].detach().cpu().numpy(), title='all_pixel_offset_est')
            plot_heatmap(axes_array, 1, 2, all_pixel_offset_est[sample_idx,:,:].detach().cpu().numpy())
            plot_surf(axes_array, 0, 3, all_params[sample_idx,:,:,:].detach().cpu().numpy(), all_pixel_offset_x_est[sample_idx,:,:].detach().cpu().numpy(), title='all_pixel_offset_x_est')
            plot_heatmap(axes_array, 1, 3, all_pixel_offset_x_est[sample_idx,:,:].detach().cpu().numpy())
            plot_surf(axes_array, 0, 4, all_params[sample_idx,:,:,:].detach().cpu().numpy(), all_pixel_offset_y_est[sample_idx,:,:].detach().cpu().numpy(), title='all_pixel_offset_y_est')
            plot_heatmap(axes_array, 1, 4, all_pixel_offset_y_est[sample_idx,:,:].detach().cpu().numpy())
            if calc_grad:
                plot_surf(axes_array, 0, 1, all_params[sample_idx,:,:,:].detach().cpu().numpy(), all_wx_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())
                plot_heatmap(axes_array, 1, 1, all_wx_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())
                # plot_surf(axes_array, 0, 1, all_params[sample_idx,:,:,:].detach().cpu().numpy(), all_tx_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())
                # plot_heatmap(axes_array, 1, 1, all_tx_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())

        fig.savefig(os.path.join(self._out_path, '00_func.png'))
        # fig.savefig(os.path.join(self._out_path, '00_func-default_and_90deg-nomax50-train100.png'))
        # fig.savefig(os.path.join(self._out_path, '00_func-default_and_90deg-nomax50-109.png'))
        # fig.savefig(os.path.join(self._out_path, '00_func-default_and_90deg-nomax50-112.png'))

        assert False
