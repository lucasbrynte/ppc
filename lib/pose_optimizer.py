import os
import json
from collections import OrderedDict
import numpy as np
# np.random.seed(314159)
from scipy import spatial
import torch
from torch import nn
from torch.autograd import grad
from lib.expm.expm32 import expm32
from lib.expm.expm64 import expm64
from lib.utils import get_rotation_axis_angle, order_dict
from lib.utils import square_bbox_around_projected_object_center_pt_batched, crop_and_rescale_pt_batched, get_projectivity_for_crop_and_rescale_pt_batched

from PIL import Image
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
        rendering_wrapper,
        ref_img_full,
        K,
        obj_diameter,
        obj_id_list,
        ambient_weight,
    ):
        super().__init__()
        self._configs = configs
        self._model = model
        self._rendering_wrapper = rendering_wrapper
        self._ref_img_full = ref_img_full
        self._K = K
        self._obj_diameter = obj_diameter
        self._obj_id_list = np.array(obj_id_list)
        self._ambient_weight = np.array(ambient_weight)

        self._out_path = os.path.join(self._configs.experiment_path, 'eval_poseopt')

    def forward(self, R, t, R_refpt=None, selected_samples=None, fname_dict={}):
        """
        selected_samples argument controls which samples to pick from variables defined in constructor. The tensors passed to this function are expected to be filtered already.
        """
        # # Punish w
        # return torch.norm(w, dim=1)

        # # Punish theta
        # R = w_to_R(w)
        # trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
        # theta = torch.acos(0.5*(trace-1.0))
        # return theta

        ref_img_full = self._ref_img_full
        K = self._K
        obj_diameter = self._obj_diameter
        ambient_weight_list = self._ambient_weight
        obj_id_list = self._obj_id_list

        if selected_samples is not None:
            ref_img_full = ref_img_full[selected_samples,:,:,:]
            K = K[selected_samples,:,:]
            obj_diameter = obj_diameter[selected_samples]
            ambient_weight_list = ambient_weight_list[selected_samples]
            obj_id_list = obj_id_list[selected_samples]

        # Define crop box in order to center object in query image
        xc, yc, width, height = square_bbox_around_projected_object_center_pt_batched(t, K, obj_diameter, crop_box_resize_factor = self._configs.data.crop_box_resize_factor)
        H = get_projectivity_for_crop_and_rescale_pt_batched(xc, yc, width, height, self._configs.data.crop_dims)
        ref_img = crop_and_rescale_pt_batched(ref_img_full, H, self._configs.data.crop_dims, interpolation_mode='bilinear')
        HK = torch.bmm(H, K)

        if R_refpt is not None:
            R = torch.bmm(R, R_refpt)
        rendering_out = self._rendering_wrapper.render(
            HK,
            R,
            t,
            obj_id_list,
            ambient_weight_list,
            self._configs.data.crop_dims,
            lowres_render_dims = self._configs.data.query_rendering_opts.lowres_render_size,
        )
        query_img = rendering_out['img']
        query_img = (query_img - TV_MEAN[None,:,None,None].cuda()) / TV_STD[None,:,None,None].cuda()

        for sample_idx, fname in fname_dict.items():
            # img1 = _retrieve_input_img(ref_img[sample_idx,:,:,:].detach().cpu())
            # img2 = _retrieve_input_img(query_img[sample_idx,:,:,:].detach().cpu())
            # Image.fromarray((img1*255.).astype(np.uint8)).save('/workspace/experiments/01_rawimgs_{:02d}_img1.png'.format(sample_idx))
            # Image.fromarray((img2*255.).astype(np.uint8)).save('/workspace/experiments/01_rawimgs_{:02d}_img2.png'.format(sample_idx))

            fig, axes_array = plt.subplots(nrows=1, ncols=2, squeeze=False)
            axes_array[0,0].imshow(_retrieve_input_img(ref_img[sample_idx,:,:,:].detach().cpu()))
            axes_array[0,1].imshow(_retrieve_input_img(query_img[sample_idx,:,:,:].detach().cpu()))
            full_fpath = os.path.join(self._out_path, fname)
            os.makedirs(os.path.dirname(full_fpath), exist_ok=True)
            fig.savefig(full_fpath)

        # assert False

        # # Punish pixels
        # sh = query_img.shape
        # punish_img = query_img[0,:,:sh[2]//2,:sh[3]//2]
        # # punish_img = query_img[0,:,::,:]
        # punish_img = normalize(punish_img, mean=-TV_MEAN/TV_STD, std=1/TV_STD) / 255.
        # return torch.mean(punish_img**2)
        # # return torch.mean(punish_img**2, dim=(1,2,3))

        nn_out = self._model(ref_img, query_img)

        return H, ref_img, query_img, nn_out

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
        nn_out2interp_pred_features,
        K,
        R_gt,
        t_gt,
        R_refpt,
        ref_img_path,
        numerical_grad = True,
    ):
        self._configs = configs
        self._pipeline = pipeline
        self._nn_out2interp_pred_features = nn_out2interp_pred_features
        self._numerical_grad = numerical_grad

        # self._orig_batch_size = R_gt.shape[0]
        self._dtype = R_gt.dtype
        self._device = R_gt.device

        self._orig_R_gt_all_samples = R_gt
        self._orig_t_gt_all_samples = t_gt
        self._orig_K_all_samples = K
        self._ref_img_path_all_samples = np.array(ref_img_path)
        self._orig_batch_size_all_samples = R_gt.shape[0]
        self._samples_with_init_pose = list(range(self._orig_batch_size_all_samples))

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
    def _obj_diameter(self):
        return self._pipeline._obj_diameter[self._samples_with_init_pose]

    @property
    def _obj_id_list(self):
        return self._pipeline._obj_id_list[self._samples_with_init_pose]

    @property
    def _orig_R_gt(self):
        return self._orig_R_gt_all_samples[self._samples_with_init_pose,:,:]

    @property
    def _orig_t_gt(self):
        return self._orig_t_gt_all_samples[self._samples_with_init_pose,:,:]

    @property
    def _orig_K(self):
        return self._orig_K_all_samples[self._samples_with_init_pose,:,:]

    @property
    def _ref_img_path(self):
        return self._ref_img_path_all_samples[self._samples_with_init_pose]

    @property
    def _orig_batch_size(self):
        return len(self._samples_with_init_pose)

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
        u_normalized = torch.bmm(self._H0K_inv, u)
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
        R = w_to_R(w)
        H, ref_img, query_img, nn_out = self._pipeline(R, t, selected_samples=[ sample_idx for sample_idx in self._samples_with_init_pose for run_idx in range(self._num_optim_runs) ], R_refpt=R_refpt, fname_dict=fname_dict)
        return H, self._nn_out2interp_pred_features(nn_out)

    def eval_func_and_calc_analytical_grad(self, wx, tx, d, fname_dict={}):
        """
        Eval function and calculate analytical gradients
        """
        H, pred_features = self.eval_func(wx, tx, d, R_refpt=self._R_refpt, fname_dict=fname_dict)
        err_est = pred_features['avg_reproj_err']
        pixel_offset_est = pred_features['pixel_offset']
        rel_depth_est = pred_features['rel_depth_error']
        # Sum over batch for aggregated loss. Each term will only depend on its corresponding elements in the parameter tensors anyway.
        agg_loss = torch.sum(err_est)
        wx_grad, tx_grad, d_grad = grad((agg_loss,), (wx, tx, d))
        return H, err_est, wx_grad, tx_grad, d_grad

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
            wx2[:,x_idx] += forward_diff * step_size #* 110. / self._obj_diameter
            H, pred_features = self.eval_func(wx2, tx, d, R_refpt=self._R_refpt, fname_dict={})
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
            H, pred_features = self.eval_func(wx, tx2, d, R_refpt=self._R_refpt, fname_dict={})
            y2 = pred_features['avg_reproj_err'].squeeze(1)
            assert y2.shape == (self._batch_size,)
            grad[:,x_idx] = forward_diff * (y2-y1) / float(step_size)
        grad = grad.detach()
        return grad

    def eval_func_and_calc_numerical_d_grad(self, wx, tx, d1, y1, step_size, x_indices=None):
        """
        Eval function and calculate numerical gradients
        """
        nbr_params = d1.shape[1]
        if x_indices is None:
            x_indices = list(range(nbr_params))
        assert y1.shape == (self._batch_size,)
        grad = torch.zeros_like(d1)
        assert grad.shape == (self._batch_size, nbr_params)
        for x_idx in x_indices:
            d2 = d1.clone()
            forward_diff = 2.*(torch.rand(self._batch_size, device=self._device) < 0.5).float() - 1.
            d2[:,x_idx] += forward_diff*step_size
            H, pred_features = self.eval_func(wx, tx, d2, R_refpt=self._R_refpt, fname_dict={})
            y2 = pred_features['avg_reproj_err'].squeeze(1)
            assert y2.shape == (self._batch_size,)
            grad[:,x_idx] = forward_diff * (y2-y1) / float(step_size)
        grad = grad.detach()
        return grad

    def calc_adds_metric(self, allpts_objframe, object_diameter, R_est, t_est, R_gt, t_gt):
        batch_size = R_est.shape[0]
        N = R_est.shape[1]
        nbr_pts = allpts_objframe.shape[3]

        assert allpts_objframe.shape == (batch_size, 1, 3, nbr_pts)
        assert R_est.shape == (batch_size, N, 3, 3)
        assert t_est.shape == (batch_size, N, 3, 1)
        assert R_gt.shape == (batch_size, 1, 3, 3)
        assert t_gt.shape == (batch_size, 1, 3, 1)

        allpts_camframe_est = (torch.matmul(R_est, allpts_objframe) + t_est).detach().cpu().numpy()
        allpts_camframe_gt = (torch.matmul(R_gt, allpts_objframe) + t_gt).detach().cpu().numpy()

        assert allpts_camframe_est.shape == (batch_size, N, 3, nbr_pts)
        assert allpts_camframe_gt.shape == (batch_size, 1, 3, nbr_pts)

        add_metric_unnorm = np.empty((batch_size, N))
        for sample_idx in range(batch_size):
            # Note: if calculating residuals in object frame instead of global frame, and if object can be assumed constant throughout batch, then creating index once would be enough.
            currpts_camframe_gt = allpts_camframe_gt[sample_idx,0,:,:]
            mean_dist_index = spatial.cKDTree(currpts_camframe_gt.T)
            for k in range(N):
                currpts_camframe_est = allpts_camframe_est[sample_idx,k,:,:]
                closest_dists, _ = mean_dist_index.query(currpts_camframe_est.T, k=1)
                # mapped_distances = np.linalg.norm(currpts_camframe_gt - currpts_camframe_est, axis=0)
                # print(closest_dists.shape)
                # print(mapped_distances.shape)
                # print(currpts_camframe_gt.shape)
                # print(currpts_camframe_est.shape)
                # print(closest_dists.mean(), closest_dists.min(), closest_dists.max())
                # print(mapped_distances.mean(), mapped_distances.min(), mapped_distances.max())
                # add_metric_unnorm[sample_idx, k] = np.mean(mapped_distances)
                add_metric_unnorm[sample_idx, k] = np.mean(closest_dists)

        add_metric = add_metric_unnorm / object_diameter
        # print(add_metric.shape)
        # print(add_metric_unnorm.shape)
        # print(np.stack([
        #     add_metric,
        #     add_metric_unnorm,
        # ], axis=2).shape)
        # assert False
        return np.stack([
            add_metric,
            add_metric_unnorm,
        ], axis=2)

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

    def eval_pose_single_object(self, obj_id, H, wx_est, tx_est, d_est, err_est, R_gt, t_gt):
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
        pts_objframe = self._pipeline._rendering_wrapper.get_model_pts(obj_id, numpy_mode=False).T
        pts_objframe = pts_objframe[None,None,:,:] # Extra dimensions for batch & iterations
        pts_objframe = pts_objframe.expand(self._batch_size,-1,-1,-1)

        # Determine object diameter
        object_diameter = self._pipeline._rendering_wrapper.get_model_info(obj_id)['diameter']

        HK = torch.matmul(H, self._K[:,None,:,:])

        # if False:
        if self._pipeline._rendering_wrapper.get_model_info(obj_id)['readable_label'] in ('eggbox', 'glue'):
            # Assuming symmetry of 180 deg rotation around z axis (assumed in modified reprojection error calculation)
            symmetric = True
        else:
            symmetric = False

        # add_metrics1 = self.calc_adds_metric(pts_objframe, object_diameter, R_est, t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N, 2)
        # add_metrics2 = self.calc_add_metric(pts_objframe, object_diameter, R_est, t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N, 2)
        # print(add_metrics1[:,:,0,1].shape)
        # print(add_metrics2[:,:,0,1].shape)
        # print(add_metrics1[:,:,0,1].mean(), add_metrics1[:,:,0,1].min(), add_metrics1[:,:,0,1].max())
        # print(add_metrics2[:,:,0,1].mean(), add_metrics2[:,:,0,1].min(), add_metrics2[:,:,0,1].max())
        # absdiff = np.abs(add_metrics1[:,:,0,1] - add_metrics2[:,:,0,1])
        # print(absdiff.mean(), absdiff.min(), absdiff.max())
        if symmetric:
            add_metrics = self.calc_adds_metric(pts_objframe, object_diameter, R_est, t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N, 2)
        else:
            add_metrics = self.calc_add_metric(pts_objframe, object_diameter, R_est, t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N, 2)
        avg_reproj_metrics = self.calc_avg_reproj_metric(self._K[:,None,:,:], pts_objframe, R_est, t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N)
        avg_reproj_HK_metrics = self.calc_avg_reproj_metric(HK, pts_objframe, R_est, t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N)
        deg_cm_errors = self.calc_deg_cm_err(R_est, t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N, 2)
        if symmetric:
            # Run for 180deg z rot as well, and take minimum reproj error of the two.
            zrot = torch.diag(torch.tensor([-1., -1., 1.], dtype=R_est.dtype, device=R_est.device))[None,None,:,:]
            zrot180deg_avg_reproj_metrics = self.calc_avg_reproj_metric(self._K[:,None,:,:], pts_objframe, torch.matmul(R_est, zrot), t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N)
            symm_avg_reproj_metrics = np.minimum(avg_reproj_metrics, zrot180deg_avg_reproj_metrics)

            zrot180deg_deg_cm_errors = self.calc_deg_cm_err(torch.matmul(R_est, zrot), t_est, R_gt, t_gt).reshape(self._orig_batch_size, len(self._optim_runs), N, 2)
            symm_deg_cm_errors = np.minimum(deg_cm_errors, zrot180deg_deg_cm_errors)
        else:
            symm_avg_reproj_metrics = avg_reproj_metrics
            symm_deg_cm_errors = deg_cm_errors
        err_est_numpy = err_est.detach().cpu().numpy().reshape(self._orig_batch_size, len(self._optim_runs), N)
        R_est_numpy = R_est.detach().cpu().numpy().reshape(self._orig_batch_size, len(self._optim_runs), N, 3, 3)
        t_est_numpy = t_est.detach().cpu().numpy().reshape(self._orig_batch_size, len(self._optim_runs), N, 3, 1)
        HK_numpy = HK.detach().cpu().numpy().reshape(self._orig_batch_size, len(self._optim_runs), N, 3, 3)
        metrics = [ {
            'ref_img_path': ref_img_path,
            'optim_runs': self._optim_runs,
            'optim_run_names_sorted': list(self._optim_runs.keys()),
            'detection_missing': False,
            'R_est': curr_R_est.tolist(),
            't_est': curr_t_est.tolist(),
            'HK': curr_HK.tolist(),
            'metrics': {
                'add_metric': add_metric.tolist(),
                'avg_reproj_metric': avg_reproj_metric.tolist(),
                'avg_reproj_HK_metric': avg_reproj_HK_metric.tolist(),
                'symm_avg_reproj_metric': symm_avg_reproj_metric.tolist(),
                'deg_cm_err': deg_cm_err.tolist(),
                'symm_deg_cm_err': symm_deg_cm_err.tolist(),
                'err_est': curr_err_est.tolist(),
            },
        } for (
            ref_img_path,
            add_metric,
            avg_reproj_metric,
            avg_reproj_HK_metric,
            symm_avg_reproj_metric,
            deg_cm_err,
            symm_deg_cm_err,
            curr_err_est,
            curr_R_est,
            curr_t_est,
            curr_HK,
        ) in zip(
            self._ref_img_path,
            add_metrics,
            avg_reproj_metrics,
            avg_reproj_HK_metrics,
            symm_avg_reproj_metrics,
            deg_cm_errors,
            symm_deg_cm_errors,
            err_est_numpy,
            R_est_numpy,
            t_est_numpy,
            HK_numpy,
        ) ]
        return metrics

    def eval_pose_noinit(self, ref_img_paths):
        # bs = len(ref_img_paths)
        N = 1

        add_metric = np.empty((len(self._optim_runs), N, 2))
        add_metric.fill(np.inf)

        avg_reproj_metric = np.empty((len(self._optim_runs), N))
        avg_reproj_metric.fill(np.inf)

        avg_reproj_HK_metric = np.empty((len(self._optim_runs), N))
        avg_reproj_HK_metric.fill(np.inf)

        symm_avg_reproj_metric = avg_reproj_metric
        deg_cm_err = np.empty((len(self._optim_runs), N, 2))
        deg_cm_err.fill(np.inf)
        symm_deg_cm_err = deg_cm_err

        R_est = np.empty((len(self._optim_runs), N, 3, 3))
        R_est.fill(np.nan)
        t_est = np.empty((len(self._optim_runs), N, 3, 1))
        t_est.fill(np.nan)
        HK = np.empty((len(self._optim_runs), N, 3, 3))
        HK.fill(np.nan)
        err_est = np.empty((len(self._optim_runs), N))
        err_est.fill(np.nan)

        metrics = [ {
            'ref_img_path': ref_img_path,
            'optim_runs': self._optim_runs,
            'optim_run_names_sorted': list(self._optim_runs.keys()),
            'detection_missing': True,
            'R_est': R_est.tolist(),
            't_est': t_est.tolist(),
            'HK': HK.tolist(),
            'metrics': {
                'add_metric': add_metric.tolist(),
                'avg_reproj_metric': avg_reproj_metric.tolist(),
                'avg_reproj_HK_metric': avg_reproj_HK_metric.tolist(),
                'symm_avg_reproj_metric': symm_avg_reproj_metric.tolist(),
                'deg_cm_err': deg_cm_err.tolist(),
                'symm_deg_cm_err': symm_deg_cm_err.tolist(),
                'err_est': err_est.tolist(),
            },
        } for ref_img_path in ref_img_paths ]
        return metrics

    def eval_pose(self, all_H, all_wx, all_tx, all_d, all_err_est):
        # NOTE: Assuming constant object ID. Since these methods rely on torch.expand on a single object model, the most efficient way to support multiple object IDs would probably be to define separate batches for the different objects.
        assert len(set(self._obj_id_list)) == 1
        obj_id = self._obj_id_list[0]
        all_metrics = self.eval_pose_single_object(obj_id, all_H, all_wx, all_tx, all_d, all_err_est, self._R_gt, self._t_gt)
        return all_metrics

    def store_eval(self, metrics):
        img_dir, img_fname = os.path.split(metrics['ref_img_path'])
        seq, rgb_dir = os.path.split(img_dir)
        assert rgb_dir == 'rgb'
        json_fname = '.'.join([img_fname.split('.')[0], 'json'])
        os.makedirs(os.path.join(self._out_path, 'evaluation', seq), exist_ok=True)
        with open(os.path.join(self._out_path, 'evaluation', seq, json_fname), 'w') as f:
            json.dump(metrics, f)

    def _init_cos_transitions_scheduler(
        self,
        optimizer,
        x_vals,
        y_vals,
    ):
        x_vals = np.array(x_vals, np.float64)
        y_vals = np.array(y_vals, np.float64)
        nbr_breakpoints = x_vals.shape[0]
        assert x_vals.shape == (nbr_breakpoints,)
        assert y_vals.shape == (nbr_breakpoints,)
        assert np.all(np.diff(x_vals) >= 0.) # Verify monotonically increasing

        def calc_lr_decay_factor(x):
            for j, bp in enumerate(x_vals):
                if x <= bp:
                    break
            else:
                # Out of range on right side
                return y_vals[-1]
            if j == 0:
                # Out of range on left side
                return y_vals[0]
            assert j >= 1 and j <= nbr_breakpoints
            x1 = x_vals[j-1]
            x2 = x_vals[j]
            y1 = y_vals[j-1]
            y2 = y_vals[j]
            assert x >= x1
            assert x <= x2
            y = y2 + 0.5 * (y1-y2) * (1.0 + np.cos((x-x1)/(x2-x1)*np.pi))
            if y2 >= y1: # verify within range (better safe than sorry)
                assert y >= y1
                assert y <= y2
            else:
                assert y <= y1
                assert y >= y2
            return y

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            calc_lr_decay_factor,
        )

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

    def _get_H0(self, t0, K, obj_diameter):
        """
        Determine a nominal projectivity H0 which will be fixed throughout optimization.
        This determines the "pixel" space of the unknown translation parameters tx.
        """
        # Define crop box in order to center object in query image
        xc, yc, width, height = square_bbox_around_projected_object_center_pt_batched(t0, K, obj_diameter, crop_box_resize_factor = self._configs.data.crop_box_resize_factor)
        H0 = get_projectivity_for_crop_and_rescale_pt_batched(xc, yc, width, height, self._configs.data.crop_dims)
        return H0

    def optimize(
            self,
            init_pose_before_perturb,
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
        # First iteration corresponds to evaluating initialization
        N += 1

        init_names = sorted(init_pose_before_perturb.keys())
        # Add new dimension for different initializations
        R0_before_perturb = torch.stack([init_pose_before_perturb[init_name][0] for init_name in init_names], dim=1)
        t0_before_perturb = torch.stack([init_pose_before_perturb[init_name][1] for init_name in init_names], dim=1)

        # ======================================================================
        # Determine for which samples there are pose proposals available.
        assert t0_before_perturb.shape == (self._orig_batch_size_all_samples, len(init_names), 3, 1)
        # Exploit that NaN == NaN is not true:
        samples_with_init_pose = (t0_before_perturb == t0_before_perturb).squeeze(3).any(dim=2) # (bs, len(init_names))
        assert torch.all(samples_with_init_pose.any(dim=1) == samples_with_init_pose.all(dim=1)) # For convenience, either none or all pose proposals are assumed to be available for each sample.
        samples_with_init_pose = samples_with_init_pose.any(dim=1)
        self._samples_with_init_pose = np.nonzero(samples_with_init_pose.cpu().numpy())[0]

        # All input arguments have to be filtered accordingly, in order to effectively disregard outher samples.
        R0_before_perturb = R0_before_perturb[self._samples_with_init_pose,:,:,:]
        t0_before_perturb = t0_before_perturb[self._samples_with_init_pose,:,:,:]
        # ======================================================================

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

        # Collapse the dimension corresponding to different initializations.
        R0_before_perturb = R0_before_perturb.reshape((len(init_names)*self._orig_batch_size, 3, 3))
        t0_before_perturb = t0_before_perturb.reshape((len(init_names)*self._orig_batch_size, 3, 1))

        # NOTE: interleave=True along optim runs, and False along batch, since this allows for reshaping to (batch_size, num_optim_runs, ..., ...) in the end
        R0_before_perturb = self._repeat_onedim(R0_before_perturb, self._num_optim_runs, dim=0, interleave=True)
        t0_before_perturb = self._repeat_onedim(t0_before_perturb, self._num_optim_runs, dim=0, interleave=True)
        R_perturb = self._repeat_onedim(R_perturb, len(init_names)*self._orig_batch_size, dim=0, interleave=False)
        t_perturb = self._repeat_onedim(t_perturb, len(init_names)*self._orig_batch_size, dim=0, interleave=False)
        u_perturb = self._repeat_onedim(u_perturb, len(init_names)*self._orig_batch_size, dim=0, interleave=False)
        d_perturb = self._repeat_onedim(d_perturb, len(init_names)*self._orig_batch_size, dim=0, interleave=False)

        # Each pose init is subject to all perturbations, i.e. results in more optim runs.
        self._optim_runs = OrderedDict([('{}_{}'.format(init_name, optim_run_name), val) for init_name in init_names for optim_run_name, val in self._optim_runs.items()])

        # ======================================================================
        # Store results for samples without initialization.
        if store_eval:
            all_metrics = self.eval_pose_noinit([ ref_img_path for sample_idx, ref_img_path in enumerate(self._ref_img_path_all_samples) if not sample_idx in self._samples_with_init_pose ])
            for metrics in all_metrics:
                # print(json.dumps(metrics, indent=4))
                self.store_eval(metrics)

        if not len(self._samples_with_init_pose) > 0:
            # Workaround: no need to do anything of the below if all samples lacked initialization (and hence no need to support this case).
            # NOTE: Plenty of the stuff above was unnecessary, but at least optim runs have to be defined in order to be properly stored.
            print('All samples in batch lacked pose proposal, aborting pose optimization.')
            return
        # ======================================================================

        R0 = torch.matmul(R_perturb, R0_before_perturb)
        t0_before_u_perturb = (t0_before_perturb + t_perturb).detach()
        self._R_refpt = R0.clone()

        obj_diameter = self._obj_diameter.repeat_interleave(self._num_optim_runs, dim=0)
        H0 = self._get_H0(t0_before_u_perturb, self._K, obj_diameter)
        self._H0K = torch.matmul(H0, self._K)
        self._H0K_inv = torch.inverse(self._H0K)

        w_gt = R_to_w(torch.matmul(self._R_gt, self._R_refpt.permute((0,2,1)))).detach()

        # ======================================================================
        # NOTE on how to interpret w / u / d parameters.
        # These comments disregard the case when there are perturbations put on w / u / d.
        # 
        # R is parameterized as R = delta_R(w0+W*delta_w) * R_refpt = delta_R(w0+delta_w) * R0
        # w0 (=_w_basis_origin) is chosen so as to let R = R0 for delta_w=0 (where we initialize the optimization).
        # Furthermore, unless _num_wxdims < 3, W is simply chosen as W=I.
        # 
        # t is parameterized by u and d together, and the mapping t <-> (u,d) goes both ways.
        # 
        # u is parameterized as u = u0 + U*delta_u, and represents pixels in HK space.
        # u0 represents the projection of t0 (the center of the initial pose).
        # For now, U is simply chsoen as U=I, which would be problematic if _num_txdims < 2, as u_gt could not be reached.
        # 
        # d is parameterized as d0 * exp(delta_d), where d0 is the depth of t0, and delta_d is initialized to 0.
        # ======================================================================

        # Set the origin of the w basis to the R0 point, expressed in relation to the R_refpt.
        self._w_basis_origin = R_to_w(torch.matmul(R0, self._R_refpt.permute((0,2,1)))).detach()
        if self._num_wxdims < 3:
            # If not all dimensions are free, at least make it possible to reach w_gt from _w_basis_origin, by setting this direction to the primary basis vector.
            self._w_basis = self._get_w_basis(primary_w_dir = w_gt-self._w_basis_origin)
        else:
            self._w_basis = self._get_w_basis(primary_w_dir = None)
        self._ref_depth = t0_before_u_perturb[:,2,:].squeeze(1)
        self._u_basis_origin = torch.bmm(self._H0K, t0_before_u_perturb)
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
            lr = 1.3e-1,
            # lr = 7e-2,
            # lr = 5e-2,
            # lr = 3e-2,
            betas = (0.6, 0.9),
        )
        self._tx_optimizer = torch.optim.SGD(
            [
                tx,
            ],
            # lr = 2.0,
            lr = 1.0,
            # momentum = 0.9,
            # momentum = 0.7,
            momentum = 0.5,
            # momentum = 0.3,
        )
        self._d_optimizer = torch.optim.Adam(
            [
                d,
            ],
            # lr = 7e-2,
            # lr = 5e-2,
            lr = 3e-2,
            # lr = 1e-2,
            betas = (0.4, 0.9),
        )

        # tx_leap_flag = True
        tx_leap_flag = False

        if tx_leap_flag:
            assert self._num_txdims == 2

        if self._num_txdims > 0:
            # nbr_iter_translonly = 2
            nbr_iter_translonly = 0
            # nbr_iter_translonly = 5
            # nbr_iter_translonly = 8
        else:
            nbr_iter_translonly = 0


        if not N > 1:
            self._wx_scheduler = torch.optim.lr_scheduler.LambdaLR(self._wx_optimizer, lr_lambda=lambda x: 1.0)
            self._tx_scheduler = torch.optim.lr_scheduler.LambdaLR(self._tx_optimizer, lr_lambda=lambda x: 1.0)
            self._d_scheduler = torch.optim.lr_scheduler.LambdaLR(self._d_optimizer, lr_lambda=lambda x: 1.0)
        else:
            # "Hack" - if N = 1, there is no need to define schedulers. Which is nice since they have hard-coded iteration break points.
            self._wx_scheduler = self._init_cos_transitions_scheduler(
                self._wx_optimizer,
                [
                    nbr_iter_translonly - 1,
                    nbr_iter_translonly,
                    nbr_iter_translonly + 50,
                ],
                [
                    0.0,
                    1.0,
                    # 1.0,
                    3e-2,
                ],
            )
            
            
            self._tx_scheduler = self._init_cos_transitions_scheduler(
                self._tx_optimizer,
                [
                    nbr_iter_translonly - 1,
                    nbr_iter_translonly,
                    nbr_iter_translonly + 50,
                ],
                [
                    0.0 if tx_leap_flag else 1.0,
                    1.0,
                    1.0,
                    # 1e+1,
                ],
            )
            self._d_scheduler = self._init_cos_transitions_scheduler(
                self._d_optimizer,
                [
                    nbr_iter_translonly + 10,
                    nbr_iter_translonly + 50,
                    N,
                ],
                [
                    # 0.0,
                    5e-1,
                    5e-1,
                    # 1.0,
                    1e-2,
                ],
            )



















        # self._wx_optimizer = torch.optim.Adam(
        #     [
        #         wx,
        #         # d,
        #     ],
        #     # lr = 1e-1,
        #     lr = 7e-2, # better than 3e-2 for max50..?
        #     # lr = 5e-2,
        #     # lr = 3e-2, # best for nomax50..?
        #     # lr = 1e-2,
        #     # betas = (0.95, 0.99),
        #     betas = (0.6, 0.9),
        # )
        # self._tx_optimizer = torch.optim.SGD(
        #     [
        #         tx,
        #     ],
        #     lr = 2.0,
        #     # lr = 1.0,
        #     momentum = 0.7,
        # )
        # self._d_optimizer = torch.optim.Adam(
        #     [
        #         d,
        #         # d.clone(),
        #     ],
        #     # lr = 1e-1,
        #     lr = 7e-2,
        #     # lr = 5e-2,
        #     # lr = 3e-2,
        #     # lr = 1e-2,
        #     betas = (0.6, 0.9),
        # )
        # # self._d_optimizer = torch.optim.SGD(
        # #     [
        # #         d,
        # #     ],
        # #     lr = 1.0,
        # #     momentum = 0.1,
        # # )
        # 
        # # tx_leap_flag = True
        # tx_leap_flag = False
        # 
        # if tx_leap_flag:
        #     assert self._num_txdims == 2
        # 
        # if self._num_txdims > 0:
        #     # nbr_iter_translonly = 2
        #     # nbr_iter_translonly = 0
        #     # nbr_iter_translonly = 5
        #     nbr_iter_translonly = 8
        # else:
        #     nbr_iter_translonly = 0
        # 
        # 
        # # final_finetune_iter = nbr_iter_translonly + 50
        # # self._wx_scheduler = self._init_cos_transitions_scheduler(
        # #     self._wx_optimizer,
        # #     [
        # #         nbr_iter_translonly + 15 - 1,
        # #         nbr_iter_translonly + 15,
        # #         nbr_iter_translonly + 20,
        # #         nbr_iter_translonly + 25,
        # #         # nbr_iter_translonly + 20 - 1,
        # #         # nbr_iter_translonly + 20,
        # #         final_finetune_iter + 20,
        # #     ],
        # #     [
        # #         0.0,
        # #         # 1e-1,
        # #         # 1e-1,
        # #         3e-2,
        # #         3e-2,
        # #         # 1e-2,
        # #         # 1e-2,
        # #         1.0,
        # #         1e-2,
        # #     ],
        # # )
        # 
        # 
        # # NOTE: below seemed to work out actually, but took a while to converge.
        # final_finetune_iter = nbr_iter_translonly + 60
        # self._wx_scheduler = self._init_cos_transitions_scheduler(
        #     self._wx_optimizer,
        #     [
        #         nbr_iter_translonly + 20 - 1,
        #         nbr_iter_translonly + 20,
        #         nbr_iter_translonly + 25,
        #         nbr_iter_translonly + 30,
        #         # nbr_iter_translonly + 20 - 1,
        #         # nbr_iter_translonly + 20,
        #         final_finetune_iter + 20,
        #     ],
        #     [
        #         0.0,
        #         1e-1,
        #         1e-1,
        #         1.0,
        #         1e-2,
        #     ],
        # )
        # 
        # 
        # self._tx_scheduler = self._init_cos_transitions_scheduler(
        #     self._tx_optimizer,
        #     [
        #         nbr_iter_translonly - 1,
        #         nbr_iter_translonly,
        #         final_finetune_iter,
        #     ],
        #     [
        #         0.0 if tx_leap_flag else 1.0,
        #         1.0,
        #         1e-1,
        #     ],
        # )
        # self._d_scheduler = self._init_cos_transitions_scheduler(
        #     self._d_optimizer,
        #     [
        #         nbr_iter_translonly - 1,
        #         nbr_iter_translonly,
        #         # nbr_iter_translonly + 20,
        #         # final_finetune_iter + 25,
        #         # final_finetune_iter + 35,
        #         final_finetune_iter + 30,
        #         N,
        #     ],
        #     [
        #         0.0,
        #         1.0,
        #         # 1.0,
        #         3e-2,
        #         1e-2,
        #     ],
        #     # self._d_optimizer,
        #     # [
        #     #     nbr_iter_translonly - 1,
        #     #     nbr_iter_translonly,
        #     #     # nbr_iter_translonly + 20,
        #     #     final_finetune_iter + 30,
        #     # ],
        #     # [
        #     #     0.0,
        #     #     1.0,
        #     #     # 1.0,
        #     #     3e-2,
        #     #     # 1e-2,
        #     # ],
        # )
        # self._wx_scheduler = self._init_constant_scheduler(self._wx_optimizer)
        # self._tx_scheduler = self._init_constant_scheduler(self._tx_optimizer)
        # self._d_scheduler = self._init_constant_scheduler(self._d_optimizer)
        # self._d_scheduler = torch.optim.lr_scheduler.LambdaLR(self._d_optimizer, lambda k: 1.0 if k < nbr_iter_translonly else 0.0)

        step_size_wx = 1e-2
        step_size_tx = 1.0 # 1 px
        step_size_d = 5e-3

        all_err_est = torch.empty((self._batch_size, N), dtype=self._dtype, device=self._device)
        all_real_rel_depth_error = torch.empty((self._batch_size, N), dtype=self._dtype, device=self._device)
        all_t_errmag = torch.empty((self._batch_size, N), dtype=self._dtype, device=self._device)
        all_w_errmag = torch.empty((self._batch_size, N), dtype=self._dtype, device=self._device)
        all_H = torch.empty((self._batch_size, N, 3, 3), dtype=self._dtype, device=self._device)
        all_wx_grads = torch.empty((self._batch_size, N, self._num_wxdims), dtype=self._dtype, device=self._device)
        all_tx_grads = torch.empty((self._batch_size, N, self._num_txdims), dtype=self._dtype, device=self._device)
        all_d_grads = torch.empty((self._batch_size, N, self._num_ddims), dtype=self._dtype, device=self._device)
        all_exp_avgs = torch.zeros((self._batch_size, N, self._num_params), dtype=self._dtype, device=self._device)
        all_exp_avg_sqs = torch.zeros((self._batch_size, N, self._num_params), dtype=self._dtype, device=self._device)
        all_wx = torch.empty((self._batch_size, N, self._num_wxdims), dtype=self._dtype, device=self._device)
        all_tx = torch.empty((self._batch_size, N, self._num_txdims), dtype=self._dtype, device=self._device)
        all_d = torch.empty((self._batch_size, N, self._num_ddims), dtype=self._dtype, device=self._device)

        for j in range(N):
            if self._numerical_grad:
                H, pred_features = self.eval_func(wx, tx, d, R_refpt = self._R_refpt, fname_dict = { (sample_idx*self._num_optim_runs + run_idx): 'rendered_iterations/sample{:02}/optim_run_{:s}/iter{:03}.png'.format(sample_idx, run_name, j+1) for sample_idx in range(self._orig_batch_size) for run_idx, run_name in enumerate(self._optim_runs.keys()) } if enable_plotting else {})
                err_est = pred_features['avg_reproj_err'].squeeze(1)
                if tx_leap_flag:
                    pixel_offset_est = pred_features['pixel_offset']
                # rel_depth_est = pred_features['rel_depth_error']
                # reproj only "hack":
                # pixel_offset_est = pred_features['avg_reproj_err'].repeat(1,2)
                rel_depth_est = pred_features['avg_reproj_err']
                
                if self._wx_scheduler.get_lr()[0] > 1e-13: # Assuming only one parameter group for each optimizer
                    curr_wx_grad = self.eval_func_and_calc_numerical_wx_grad(wx, tx, d, err_est, step_size_wx)
                else:
                    curr_wx_grad = torch.zeros_like(wx)
                if self._tx_scheduler.get_lr()[0] > 1e-13: # Assuming only one parameter group for each optimizer
                    curr_tx_grad = self.eval_func_and_calc_numerical_tx_grad(wx, tx, d, err_est, step_size_tx)
                else:
                    curr_tx_grad = torch.zeros_like(tx)
                if self._d_scheduler.get_lr()[0] > 1e-13: # Assuming only one parameter group for each optimizer
                    curr_d_grad = self.eval_func_and_calc_numerical_d_grad(wx, tx, d, err_est, step_size_d)
                else:
                    curr_d_grad = torch.zeros_like(d)
            else:
                H, err_est, curr_wx_grad, curr_tx_grad, curr_d_grad = self.eval_func_and_calc_analytical_grad(wx, tx, d, fname_dict = { (sample_idx*self._num_optim_runs + run_idx): 'rendered_iterations/sample{:02}/optim_run_{:s}/iter{:03}.png'.format(sample_idx, run_name, j+1) for sample_idx in range(self._orig_batch_size) for run_idx, run_name in enumerate(self._optim_runs.keys()) } if enable_plotting else {})
            curr_t_est = self._x2t(tx, d)
            curr_w_est = self._x2w(wx)
            real_rel_depth_error = (curr_t_est[:,2] / self._t_gt[:,2]).squeeze(dim=1)
            t_errmag = (curr_t_est - self._t_gt).norm(dim=1).squeeze(dim=1)
            w_errmag = (curr_w_est - w_gt).norm(dim=1)
            if print_iterates:
                print(
                    j,
                    self._wx_scheduler.get_lr(),
                    self._tx_scheduler.get_lr(),
                    self._d_scheduler.get_lr(),
                    err_est.detach().cpu().numpy(),
                    pixel_offset_est.detach().cpu().numpy() if tx_leap_flag else None,
                    rel_depth_est.detach().cpu().numpy(),
                    wx.detach().cpu().numpy(),
                    tx.detach().cpu().numpy(),
                    d.detach().cpu().numpy(),
                    curr_wx_grad.detach().cpu().numpy(),
                    curr_tx_grad.detach().cpu().numpy(),
                    curr_d_grad.detach().cpu().numpy(),
                )
                print('err_est: {}'.format(err_est.cpu().numpy()))
                print('real_rel_depth_error: {}'.format(real_rel_depth_error.cpu().numpy()))
                print('t_errmag: {}'.format(t_errmag.cpu().numpy()))
                print('w_errmag: {}'.format(w_errmag.cpu().numpy()))

            # Store iterations
            # NOTE: Store already, since variables might be updated if tx_leap_flag==True.
            all_H[:,j,:,:] = H.detach().clone()
            all_wx[:,j,:] = wx.detach().clone()
            all_tx[:,j,:] = tx.detach().clone()
            all_d[:,j,:] = d.detach().clone()

            # Store iterations
            all_err_est[:,j] = err_est.detach()
            all_real_rel_depth_error[:,j] = real_rel_depth_error.detach()
            all_t_errmag[:,j] = t_errmag.detach()
            all_w_errmag[:,j] = w_errmag.detach()

            if j == N-1:
                # Skip taking gradient step at last iteration. This way optimizer & scheduler do not need to be defined as well, which makes it easier to run the whole thing with N = 1.
                break

            if self._num_wxdims > 0:
                wx.grad = curr_wx_grad #* 110. / self._obj_diameter[:,None]
            if self._num_txdims > 0:
                if tx_leap_flag and j < nbr_iter_translonly:
                    # Take a leap
                    tx -= pixel_offset_est[:,:]
                else:
                    tx.grad = curr_tx_grad
            if self._num_ddims == 1:
                d.grad = curr_d_grad
                # d.grad = torch.log(rel_depth_est[:,:])

            # Store iterations
            all_wx_grads[:,j,:] = curr_wx_grad.detach().clone()
            all_tx_grads[:,j,:] = curr_tx_grad.detach().clone()
            all_d_grads[:,j,:] = curr_d_grad.detach().clone()

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
                all_exp_avgs[:,j,:] = exp_avg.detach().clone()
                all_exp_avg_sqs[:,j,:] = exp_avg_sq.detach().clone()

            self._wx_optimizer.step()
            self._tx_optimizer.step()
            self._d_optimizer.step()
            self._wx_scheduler.step()
            self._tx_scheduler.step()
            self._d_scheduler.step()

        if store_eval:
            all_metrics = self.eval_pose(all_H, all_wx, all_tx, all_d, all_err_est)
            for metrics in all_metrics:
                # print(json.dumps(metrics, indent=4))
                self.store_eval(metrics)

        def plot(sample_idx, fname):
            # Scalar parameter x.
            nrows = 1
            nrows += 1
            fig, axes_array = plt.subplots(nrows=nrows, ncols=3, squeeze=False)
            # TODO: Plot all_wx, all_tx and all_d separately and neatly.
            axes_array[0,0].plot(all_x[sample_idx,:,:].detach().cpu().numpy())
            axes_array[0,0].set_title('all_x')
            axes_array[0,1].plot(all_err_est[sample_idx,:].detach().cpu().numpy())
            axes_array[0,1].set_title('all_err_est')
            axes_array[1,0].plot(all_real_rel_depth_error[sample_idx,:].detach().cpu().numpy())
            axes_array[1,0].set_title('all_real_rel_depth_error')
            axes_array[1,1].plot(all_t_errmag[sample_idx,:].detach().cpu().numpy())
            axes_array[1,1].set_title('all_t_errmag')
            axes_array[1,2].plot(all_w_errmag[sample_idx,:].detach().cpu().numpy())
            axes_array[1,2].set_title('all_w_errmag')
            # axes_array[0,2].plot(all_wx_grads[sample_idx,:,:].detach().cpu().numpy())
            # # axes_array[0,2].plot(all_tx_grads[sample_idx,:,:].detach().cpu().numpy())
            # axes_array[0,2].plot(all_exp_avgs[sample_idx,:,:].abs().detach().cpu().numpy())
            # axes_array[1,2].plot(all_exp_avg_sqs[sample_idx,:,:].abs().detach().cpu().numpy())
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

        t0 = self._t_gt.detach()

        H0 = self._get_H0(t0, self._K, self._obj_diameter)
        self._H0K = torch.matmul(H0, self._K)
        self._H0K_inv = torch.inverse(self._H0K)

        w_gt = R_to_w(torch.matmul(self._R_gt, self._R_refpt.permute((0,2,1)))).detach()

        self._w_basis_origin = torch.zeros((self._batch_size, 3), dtype=self._dtype, device=self._device)
        self._w_basis = self._get_w_basis(primary_w_dir = None)
        self._ref_depth = self._t_gt[:,2,:].squeeze(1)

        self._u_basis_origin = torch.bmm(self._H0K, t0)
        self._u_basis_origin = self._u_basis_origin[:,:2,:] / self._u_basis_origin[:,[2],:]
        self._u_basis = self._get_u_basis()
        self._d_origin = t0.squeeze(2)[:,[2]]

        # ======================================================================
        # NOTE on how to interpret w / u / d parameters.
        # These comments disregard the case when there are perturbations put on w / u / d.
        # 
        # Note that R0 = R_gt and t0 = t_gt.
        # 
        # R is parameterized as R = delta_R(w0+W*delta_w) * R_refpt = delta_R(w0+delta_w) * R0
        # w0 (=_w_basis_origin) is chosen so as to let R = R0 for delta_w=0 (where we initialize the optimization).
        # W is simply chosen as W=I.
        # 
        # t is parameterized by u and d together, and the mapping t <-> (u,d) goes both ways.
        # 
        # u is parameterized as u = u0 + U*delta_u, and represents pixels in HK space.
        # u0 represents the projection of t0 (the center of the initial pose).
        # For now, U is simply chsoen as U=I, which would be problematic if _num_txdims < 2, as u_gt could not be reached.
        # 
        # d is parameterized as d0 * exp(delta_d), where d0 is the depth of t0, and delta_d is initialized to 0.
        # ======================================================================

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
        # param_delta = 0.0
        # param_delta = 0.001
        # param_delta = 0.01
        # param_delta = 0.02
        # param_delta = 0.05
        param_delta = 0.15
        # param_delta = 0.25
        # param_delta = 0.5
        # param_delta = 1.5
        # param_delta = 50.

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
        step_size_tx = 1.0 # 1 px
        step_size_d = 5e-3

        # has_pixel_offset_flag = True
        has_pixel_offset_flag = False
        # has_rel_depth_error_flag = True
        has_rel_depth_error_flag = False

        all_err_est = torch.empty([self._batch_size]+N_each, dtype=self._dtype, device=self._device)
        if has_pixel_offset_flag:
            all_pixel_offset_est = torch.empty([self._batch_size]+N_each, dtype=self._dtype, device=self._device)
            all_pixel_offset_x_est = torch.empty([self._batch_size]+N_each, dtype=self._dtype, device=self._device)
            all_pixel_offset_y_est = torch.empty([self._batch_size]+N_each, dtype=self._dtype, device=self._device)
        if has_rel_depth_error_flag:
            all_rel_depth_est = torch.empty([self._batch_size]+N_each, dtype=self._dtype, device=self._device)
        if calc_grad:
            all_wx_grads = torch.empty([self._batch_size, self._num_wxdims]+N_each, dtype=self._dtype, device=self._device)
            all_tx_grads = torch.empty([self._batch_size, self._num_txdims]+N_each, dtype=self._dtype, device=self._device)
            all_d_grads = torch.empty([self._batch_size, self._num_ddims]+N_each, dtype=self._dtype, device=self._device)
        for j in range(N):
            param_vec = vec(all_params, N_each)[:,:,j] # shape: (sample_idx, x_idx)
            wx = param_vec[:,:self._num_wxdims]
            tx = param_vec[:,self._num_wxdims:self._num_wxdims+self._num_txdims]
            d = param_vec[:,self._num_wxdims+self._num_txdims:]

            if calc_grad:
                if self._numerical_grad:
                    H, pred_features = self.eval_func(wx, tx, d, R_refpt = self._R_refpt, fname_dict = { (sample_idx*self._num_optim_runs + run_idx): 'rendered_iterations/sample{:02}/optim_run_{:s}/iter{:03}.png'.format(sample_idx, run_name, j+1) for sample_idx in range(self._orig_batch_size) for run_idx, run_name in enumerate(self._optim_runs.keys()) })
                    err_est = pred_features['avg_reproj_err'].squeeze(1)
                    if has_pixel_offset_flag:
                        pixel_offset_est = pred_features['pixel_offset']
                    if has_rel_depth_error_flag:
                        rel_depth_est = pred_features['rel_depth_error']
                    curr_wx_grad = self.eval_func_and_calc_numerical_wx_grad(wx, tx, d, err_est, step_size_wx)
                    curr_tx_grad = self.eval_func_and_calc_numerical_tx_grad(wx, tx, d, err_est, step_size_tx)
                    curr_d_grad = self.eval_func_and_calc_numerical_d_grad(wx, tx, d, err_est, step_size_d)
                else:
                    H, err_est, curr_wx_grad, curr_tx_grad, curr_d_grad = self.eval_func_and_calc_analytical_grad(wx, tx, d, fname_dict = { (sample_idx*self._num_optim_runs + run_idx): 'rendered_iterations/sample{:02}/optim_run_{:s}/iter{:03}.png'.format(sample_idx, run_name, j+1) for sample_idx in range(self._orig_batch_size) for run_idx, run_name in enumerate(self._optim_runs.keys()) })
            else:
                H, pred_features = self.eval_func(wx, tx, d, R_refpt=self._R_refpt, fname_dict = { (sample_idx*self._num_optim_runs + run_idx): 'rendered_iterations/sample{:02}/optim_run_{:s}/iter{:03}.png'.format(sample_idx, run_name, j+1) for sample_idx in range(self._orig_batch_size) for run_idx, run_name in enumerate(self._optim_runs.keys()) })
                err_est = pred_features['avg_reproj_err'].squeeze(1)
                if has_pixel_offset_flag:
                    pixel_offset_est = pred_features['pixel_offset']
                if has_rel_depth_error_flag:
                    rel_depth_est = pred_features['rel_depth_error']
            curr_t_est = self._x2t(tx, d)
            curr_w_est = self._x2w(wx)
            real_rel_depth_error = (curr_t_est[:,2] / self._t_gt[:,2]).squeeze(dim=1)
            t_errmag = (curr_t_est - self._t_gt).norm(dim=1).squeeze(dim=1)
            w_errmag = (curr_w_est - w_gt).norm(dim=1)
            print(
                j,
                err_est.detach().cpu().numpy(),
                pixel_offset_est.detach().cpu().numpy() if has_pixel_offset_flag else None,
                rel_depth_est.detach().cpu().numpy() if has_rel_depth_error_flag else None,
                wx.detach().cpu().numpy(),
                tx.detach().cpu().numpy(),
                d.detach().cpu().numpy(),
                curr_wx_grad.detach().cpu().numpy() if calc_grad else None,
                curr_tx_grad.detach().cpu().numpy() if calc_grad else None,
                curr_d_grad.detach().cpu().numpy() if calc_grad else None,
            )
            print('err_est: {}'.format(err_est.cpu().numpy()))
            print('real_rel_depth_error: {}'.format(real_rel_depth_error.cpu().numpy()))
            print('t_est[2]', curr_t_est[:,2])
            print('t_gt[2]', self._t_gt[:,2])
            print('t_errmag: {}'.format(t_errmag.cpu().numpy()))
            print('w_errmag: {}'.format(w_errmag.cpu().numpy()))

            # Store iterations
            # vec(all_params, N_each)[:,:,j] = x.detach().clone()
            if calc_grad:
                vec(all_wx_grads, N_each)[:,:,j] = curr_wx_grad
                vec(all_tx_grads, N_each)[:,:,j] = curr_tx_grad
                vec(all_d_grads, N_each)[:,:,j] = curr_d_grad

            # Store iterations
            vec(all_err_est, N_each)[:,j] = err_est.detach()
            if has_pixel_offset_flag:
                vec(all_pixel_offset_est, N_each)[:,j] = pixel_offset_est.norm(dim=1).detach()
                vec(all_pixel_offset_x_est, N_each)[:,j] = pixel_offset_est[:,0].detach()
                vec(all_pixel_offset_y_est, N_each)[:,j] = pixel_offset_est[:,1].detach()
            if has_rel_depth_error_flag:
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

        all_params_interp = all_params.clone()
        all_params_interp[:,self._num_wxdims+self._num_txdims:] = torch.exp(all_params_interp[:,self._num_wxdims+self._num_txdims:])

        sample_idx = 0
        # Scalar parameter x.
        if self._num_params == 1:
            nrows = 1
            if has_rel_depth_error_flag:
                nrows += 1
            if has_pixel_offset_flag:
                nrows += 3
            if calc_grad:
                nrows += 1
            ncols = 1
            fig, axes_array = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*2), squeeze=False)
            row_idx = 0
            axes_array[row_idx,0].plot(all_params_interp[sample_idx,:,:].detach().cpu().numpy().T, all_err_est[sample_idx,:].detach().cpu().numpy())
            axes_array[row_idx,0].set_title('all_err_est')
            row_idx += 1
            if has_rel_depth_error_flag:
                axes_array[row_idx,0].plot(all_params_interp[sample_idx,:,:].detach().cpu().numpy().T, all_rel_depth_est[sample_idx,:].detach().cpu().numpy())
                axes_array[row_idx,0].set_title('all_rel_depth_est')
                row_idx += 1
            if has_pixel_offset_flag:
                axes_array[row_idx,0].plot(all_params_interp[sample_idx,:,:].detach().cpu().numpy().T, all_pixel_offset_est[sample_idx,:].detach().cpu().numpy())
                axes_array[row_idx,0].set_title('all_pixel_offset_est')
                row_idx += 1
                axes_array[row_idx,0].plot(all_params_interp[sample_idx,:,:].detach().cpu().numpy().T, all_pixel_offset_x_est[sample_idx,:].detach().cpu().numpy())
                axes_array[row_idx,0].set_title('all_pixel_offset_x_est')
                row_idx += 1
                axes_array[row_idx,0].plot(all_params_interp[sample_idx,:,:].detach().cpu().numpy().T, all_pixel_offset_y_est[sample_idx,:].detach().cpu().numpy())
                axes_array[row_idx,0].set_title('all_pixel_offset_y_est')
                row_idx += 1
            # if calc_grad:
            #     axes_array[row_idx,0].plot(all_params_interp[sample_idx,:,:].detach().cpu().numpy().T, all_wx_grads[sample_idx,:,:].detach().cpu().numpy().T)
            #     # axes_array[row_idx,0].plot(all_params_interp[sample_idx,:,:].detach().cpu().numpy().T, all_tx_grads[sample_idx,:,:].detach().cpu().numpy().T)
            #     # axes_array[row_idx,0].plot(all_params_interp[sample_idx,:,:].detach().cpu().numpy().T, all_d_grads[sample_idx,:,:].detach().cpu().numpy().T)
            #     row_idx += 1
        elif self._num_params == 2:
            nrows = 1
            nrows += 4
            if calc_grad:
                nrows += 1
            ncols = 2
            fig, axes_array = plt.subplots(figsize=(15,6), nrows=nrows, ncols=ncols, squeeze=False)
            # TODO: Plot all_wx, all_tx and all_d separately and neatly.
            plot_surf(axes_array, 0, 0, all_params_interp[sample_idx,:,:,:].detach().cpu().numpy(), all_err_est[sample_idx,:,:].detach().cpu().numpy(), title='all_err_est')
            plot_heatmap(axes_array, 0, 1, all_err_est[sample_idx,:,:].detach().cpu().numpy())
            if has_rel_depth_error_flag:
                plot_surf(axes_array, 1, 0, all_params_interp[sample_idx,:,:,:].detach().cpu().numpy(), all_rel_depth_est[sample_idx,:,:].detach().cpu().numpy(), title='all_rel_depth_est')
                plot_heatmap(axes_array, 1, 1, all_rel_depth_est[sample_idx,:,:].detach().cpu().numpy())
            if has_pixel_offset_flag:
                plot_surf(axes_array, 2, 0, all_params_interp[sample_idx,:,:,:].detach().cpu().numpy(), all_pixel_offset_est[sample_idx,:,:].detach().cpu().numpy(), title='all_pixel_offset_est')
                plot_heatmap(axes_array, 2, 1, all_pixel_offset_est[sample_idx,:,:].detach().cpu().numpy())
                plot_surf(axes_array, 3, 0, all_params_interp[sample_idx,:,:,:].detach().cpu().numpy(), all_pixel_offset_x_est[sample_idx,:,:].detach().cpu().numpy(), title='all_pixel_offset_x_est')
                plot_heatmap(axes_array, 3, 1, all_pixel_offset_x_est[sample_idx,:,:].detach().cpu().numpy())
                plot_surf(axes_array, 4, 0, all_params_interp[sample_idx,:,:,:].detach().cpu().numpy(), all_pixel_offset_y_est[sample_idx,:,:].detach().cpu().numpy(), title='all_pixel_offset_y_est')
                plot_heatmap(axes_array, 4, 1, all_pixel_offset_y_est[sample_idx,:,:].detach().cpu().numpy())
            if calc_grad:
                plot_surf(axes_array, 1, 0, all_params_interp[sample_idx,:,:,:].detach().cpu().numpy(), all_wx_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())
                plot_heatmap(axes_array, 1, 1, all_wx_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())
                # plot_surf(axes_array, 1, 0, all_params_interp[sample_idx,:,:,:].detach().cpu().numpy(), all_tx_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())
                # plot_heatmap(axes_array, 1, 1, all_tx_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())
                # plot_surf(axes_array, 1, 0, all_params_interp[sample_idx,:,:,:].detach().cpu().numpy(), all_d_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())
                # plot_heatmap(axes_array, 1, 1, all_d_grads.norm(dim=1)[sample_idx,:,:].detach().cpu().numpy())

        fig.savefig(os.path.join(self._out_path, '00_func.png'))
        # fig.savefig(os.path.join(self._out_path, '00_func-default_and_90deg-nomax50-train100.png'))
        # fig.savefig(os.path.join(self._out_path, '00_func-default_and_90deg-nomax50-109.png'))
        # fig.savefig(os.path.join(self._out_path, '00_func-default_and_90deg-nomax50-112.png'))

        # assert False
