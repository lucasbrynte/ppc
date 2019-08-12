import os
import numpy as np

from lib.utils import get_rotation_axis_angle, get_translation

def calc_object_pose_on_xy_plane(obj_pose_params, up_dir, bottom_center):
    """
    Given a set of sampled object pose parameters, returns the corresponding
    transformation from model frame to world frame.

    Arguments:
        obj_pose_params  - Size of table on which object position is sampled
        up_dir           - Upwards direction in object reference frame
        bottom_center    - Bottom center location in object reference frame
    """
    # Translate bottom center to origin
    T_bottom2origin = get_translation(-bottom_center)

    # Align upwards direction with z-axis
    up_dir = up_dir / np.linalg.norm(up_dir)
    angle = np.arccos(up_dir[2])
    if np.isclose(angle, 0.):
        T_align_up_dir = np.eye(4)
    else:
        axis = np.cross(up_dir, np.array([0., 0., 1.]))
        T_align_up_dir = get_rotation_axis_angle(axis, angle)

    # Random rotation around z-axis
    T_azimuth = get_rotation_axis_angle(np.array([0., 0., 1.]), np.pi/180.*obj_pose_params['object_azimuth_angle'])

    # Random xy-translation (on table)
    t = np.zeros((3,))
    t[0:2] = -obj_pose_params['xy_transl']
    T_transl_on_table = get_translation(t)

    T_model2world = T_transl_on_table @ T_azimuth @ T_align_up_dir @ T_bottom2origin
    return T_model2world

def calc_camera_pose(cam_pose_params):
    """
    Given a set of sampled camera pose parameters, returns the corresponding
    transformation from world frame to camera frame.
    """
    # ==========================================================================
    # 1)  Sample camera distance & elevation angle. Initialize camera on the
    #     hemisphere above the table (positive z), such that the principal
    #     axis points towards the world origin, and the world's "up" direction
    #     (z-axis) projects to "up" (negative y) in the image.
    # ==========================================================================
    # Rotation around z-axis
    T_azimuth = get_rotation_axis_angle(np.array([0., 0., 1.]), np.pi/180.*cam_pose_params['hemisphere_azimuth_angle'])
    # Rotation around x-axis (might as well have been y-axis)
    angle = np.pi/180. * (180. - cam_pose_params['hemisphere_polar_angle']) # 180 degrees moves camera to bird's eye view. Subtract polar angle from 180.
    T_polar = get_rotation_axis_angle(np.array([1., 0., 0.]), angle)
    # Translation along principal axis
    T_transl_to_hemisphere = get_translation([0., 0., cam_pose_params['hemisphere_radius']])

    # ==========================================================================
    # 2)  A random rotational perturbation around the principal axis is applied.
    # ==========================================================================
    # Rotation around z-axis
    T_inplane_rot = get_rotation_axis_angle(np.array([0., 0., 1.]), np.pi/180.*cam_pose_params['inplane_rot_angle'])

    # ==========================================================================
    # 3)  A random rotational perturbation of the principal axis itself is applied.
    #     A random vector in the principal plane is sampled, defining an axis
    #     around which yet another random rotational perturbation is applied.
    # ==========================================================================
    axis_of_revolution_for_principal_axis_perturbation = np.array([np.cos(np.pi/180.*cam_pose_params['inplane_angle_for_axis_of_revolution_for_paxis_perturb']), np.sin(np.pi/180.*cam_pose_params['inplane_angle_for_axis_of_revolution_for_paxis_perturb']), 0.0])
    T_perturb_principal_axis = get_rotation_axis_angle(axis_of_revolution_for_principal_axis_perturbation, np.pi/180.*cam_pose_params['principal_axis_perturb_angle'])

    # ==========================================================================
    # Everything put together
    # ==========================================================================
    T0 = T_transl_to_hemisphere @ T_polar @ T_azimuth # Initial camera on hemisphere (step 1)
    T_world2cam = T_perturb_principal_axis @ T_inplane_rot @ T0
    return T_world2cam
