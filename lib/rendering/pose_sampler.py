import os
import numpy as np

from lib.utils import get_rotation_axis_angle, get_translation


class PoseSampler():
    def sample_object_pose_on_xy_plane(self, up_dir, bottom_center, table_size):
        """
        Samples an object pose such that the object is placed somewhere on the xy-plane.
        Returns the corresponding transformation from model frame to world frame.

        Arguments:
            up_dir         - Upwards direction in object reference frame
            bottom_center  - Bottom center location in object reference frame
            table_size     - Size of table on which object position is sampled
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
        phi = np.random.uniform(low=0., high=2.*np.pi)
        T_azimuth = get_rotation_axis_angle(np.array([0., 0., 1.]), phi)

        # Random xy-translation (on table)
        cc = np.random.uniform(low=-0.5*table_size, high=0.5*table_size, size=(2,))
        t = np.zeros((3,))
        t[0:2] = -cc
        T_transl_on_table = get_translation(t)

        T_model2world = T_transl_on_table @ T_azimuth @ T_align_up_dir @ T_bottom2origin
        return T_model2world

    def sample_camera_pose(
            self,
            hemisphere_polar_angle_range = [0., np.pi/2],
            hemisphere_radius_range = [0.7, 1.5],
            inplane_rot_angle_range = [-np.pi/6, np.pi/6],
            principal_axis_perturb_angle_range = [-np.pi/6, np.pi/6],
        ):
        """
        Samples a camera pose such that a virtual "table" on the xy-plane, centered
        at the origin, is seen in the camera.

        Returns the corresponding transformation from world frame to camera frame.

        Sampling is done as follows:
            1)  Sample camera distance & elevation angle. Initialize camera on the
                hemisphere above the table (positive z), such that the principal
                axis points towards the world origin, and the world's "up" direction
                (z-axis) projects to "up" (negative y) in the image.
            2)  A random rotational perturbation around the principal axis is applied.
            3)  A random rotational perturbation of the principal axis itself is applied.
                A random vector in the principal plane is sampled, defining an axis
                around which yet another random rotational perturbation is applied.
        """
        hemisphere_polar_angle = np.random.uniform(low=hemisphere_polar_angle_range[0], high=hemisphere_polar_angle_range[1])
        hemisphere_radius = np.random.uniform(low=hemisphere_radius_range[0], high=hemisphere_radius_range[1])
        inplane_rot_angle = np.random.uniform(low=inplane_rot_angle_range[0], high=inplane_rot_angle_range[1])
        principal_axis_perturb_angle = np.random.uniform(low=principal_axis_perturb_angle_range[0], high=principal_axis_perturb_angle_range[1])

        # ==========================================================================
        # 1)  Sample camera distance & elevation angle. Initialize camera on the
        #     hemisphere above the table (positive z), such that the principal
        #     axis points towards the world origin, and the world's "up" direction
        #     (z-axis) projects to "up" (negative y) in the image.
        # ==========================================================================
        # Rotation around z-axis
        hemisphere_azimuth_angle = np.random.uniform(low=0., high=2.*np.pi)
        T_azimuth = get_rotation_axis_angle(np.array([0., 0., 1.]), hemisphere_azimuth_angle)
        # Rotation around x-axis (might as well have been y-axis)
        angle = np.pi - hemisphere_polar_angle # Pi degrees moves camera to bird's eye view. Subtract polar angle from pi.
        T_polar = get_rotation_axis_angle(np.array([1., 0., 0.]), angle)
        # Translation along principal axis
        T_transl_to_hemisphere = get_translation([0., 0., hemisphere_radius])

        # ==========================================================================
        # 2)  A random rotational perturbation around the principal axis is applied.
        # ==========================================================================
        # Rotation around z-axis
        T_inplane_rot = get_rotation_axis_angle(np.array([0., 0., 1.]), inplane_rot_angle)

        # ==========================================================================
        # 3)  A random rotational perturbation of the principal axis itself is applied.
        #     A random vector in the principal plane is sampled, defining an axis
        #     around which yet another random rotational perturbation is applied.
        # No reason to limit these perturbations - all angles allowed
        # ==========================================================================
        random_angle = np.random.uniform(low=0., high=2.*np.pi)
        axis_of_revolution_for_principal_axis_perturbation = np.array([np.cos(random_angle), np.sin(random_angle), 0.0])
        T_perturb_principal_axis = get_rotation_axis_angle(axis_of_revolution_for_principal_axis_perturbation, principal_axis_perturb_angle)

        # ==========================================================================
        # Everything put together
        # ==========================================================================
        T0 = T_transl_to_hemisphere @ T_polar @ T_azimuth # Initial camera on hemisphere (step 1)
        T_world2cam = T_perturb_principal_axis @ T_inplane_rot @ T0
        return T_world2cam
