import math
import numpy as np
import numba as nb
import pyreality as pr
from numba import cuda
from numba.cuda.cudadrv.devicearray import ManagedNDArray

from litar.types import Vector2Int
from litar.pipeline.pointcloud.kernels import cuda_gen_pcd_xyz
from litar.pipeline.pointcloud.kernels import cuda_gen_pcd_rgb

from profilehooks import profile, timecall


class ManagedPointCloudGenerator:
    arg_in_const: ManagedNDArray
    arg_in_cam_mat: ManagedNDArray
    arg_in_depth: ManagedNDArray

    arg_in_y: ManagedNDArray
    arg_in_cbcr: ManagedNDArray

    arg_out_pcd_xyz: ManagedNDArray
    arg_out_pcd_rgb: ManagedNDArray

    __n_points: int
    __c_size: Vector2Int
    __dn_size: Vector2Int


    def __init__(self, k: np.ndarray, dn_size: Vector2Int, c_size: Vector2Int, n_views: int) -> None:
        self.__c_size = c_size
        self.__dn_size = dn_size

        self.arg_in_const = self.arg_in_const = cuda.managed_array(
            (10), dtype=np.float32)
        self.arg_in_const[0] = c_size.x
        self.arg_in_const[1] = c_size.y
        self.arg_in_const[2] = 0  # Offset
        self.arg_in_const[3] = c_size.x / dn_size.x

        self.arg_in_cam_mat = cuda.managed_array(4 + 12, dtype=np.float32)
        self.arg_in_cam_mat[:4] = k

        self.arg_in_depth = cuda.managed_array((dn_size.x * dn_size.y),
                                               dtype=np.float32)

        self.arg_in_y = cuda.managed_array(
            (c_size.y, c_size.x), dtype=np.uint8)
        self.arg_in_cbcr = cuda.managed_array(
            (c_size.y // 2, c_size.x // 2, 2), dtype=np.uint8)

        self.__n_points = c_size.x * c_size.y
        self.arg_out_pcd_xyz = cuda.managed_array(
            (n_views * self.__n_points, 3), dtype=np.float32)
        self.arg_out_pcd_rgb = cuda.managed_array(
            (n_views * self.__n_points, 3), dtype=np.uint8)

    def exec(self, i_view: int, trs: np.ndarray, depth: np.ndarray,
             y: np.ndarray, cbcr: np.ndarray):
        """
        Generate point cloud based on a client uploaded
        near field data package
        """

        self.arg_in_const[2] = float(i_view * self.__n_points)

        # Generate Point Cloud XYZ
        n_thread = min(1024, self.__n_points)
        n_block = (self.__n_points + (n_thread - 1)) // n_thread

        self.arg_in_cam_mat[4:] = trs
        self.arg_in_depth[:] = depth

        cuda_gen_pcd_xyz[n_block, n_thread](
            self.arg_out_pcd_xyz,
            self.arg_in_const,
            self.arg_in_cam_mat,
            self.arg_in_depth)

        cuda.current_context().synchronize()

        # Generate Point Cloud RGB
        n_thread = (min(32, self.__dn_size.x), min(32, self.__dn_size.y))
        n_block = (self.__c_size.x // n_thread[0],
                   self.__c_size.y // n_thread[1])

        self.arg_in_y[::] = y
        self.arg_in_cbcr[::] = cbcr

        cuda_gen_pcd_rgb[n_block, n_thread](
            self.arg_out_pcd_rgb,
            self.arg_in_const,
            self.arg_in_y,
            self.arg_in_cbcr)

        cuda.current_context().synchronize()

    def sample_to_anchor(self, anchor_xyz, downsample_rate=200, filter_surroundings=False):
        """Sparsely sample a point cloud to paint anchor colors.

        Note: This is a CPU-based implementation, the performance might be slow.
        """
        c_offset = int(self.arg_in_const[2])
        l, r = c_offset, c_offset + self.__n_points

        spc_xyz = np.asarray(self.arg_out_pcd_xyz[l:r:downsample_rate, :])
        spc_xyz_mag = np.linalg.norm(spc_xyz, axis=-1, keepdims=True)
        spc_xyz = spc_xyz / spc_xyz_mag

        spc_rgb = np.asarray(self.arg_out_pcd_rgb[l:r:downsample_rate, :])

        t = anchor_xyz @ spc_xyz.T

        # find the distance relationship between anchor sparse points
        i_rgb_to_anchor = np.argmax(t, axis=-1)

        # Select the nearest anchor, and paint
        # this is NOT accurate, but should be good enough
        m = np.max(t, axis=-1) > 0.99

        if filter_surroundings:
            m = m & (np.arange(len(m)) > (len(m) * 3 / 4))

        return m, spc_rgb[i_rgb_to_anchor[m]]

    def sparse_sample(self, downsample_rate=100):
        """Sparsely sample a point cloud to paint anchor colors

        Note: This is a CPU-based implementation, not sure about the performance
        """
        c_offset = int(self.arg_in_const[2])
        l, r = c_offset, c_offset + self.__n_points

        sp_xyz = np.asarray(self.arg_out_pcd_xyz[l:r:downsample_rate, :])
        sp_rgb = np.asarray(self.arg_out_pcd_rgb[l:r:downsample_rate, :])

        sp_xyz = np.ascontiguousarray(sp_xyz)
        sp_rgb = np.ascontiguousarray(sp_rgb)

        return sp_xyz, sp_rgb
