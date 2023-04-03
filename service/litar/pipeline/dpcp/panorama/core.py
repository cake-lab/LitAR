from typing import List

import numpy as np
import numba as nb
from numba import cuda
from numba.cuda.cudadrv.devicearray import ManagedNDArray

from configs import N_ANCHORS
from configs import MLP_SEQUENCE
from configs import NEAR_FIELD_CLIP_DST
from configs import N_ANCHOR_NEIGHBORS

from litar.types import Vector2Int

from .kernel import cuda_update_anchors
from .kernel import dpcp_cuda
from .kernel import draw_far_field
from .kernel import merge_dpcp_cuda
from .static_data import canvas_size, canvas, anchor_xyz, anchor_acc


class ManagedPanoramaRenderer:
    canvas_size: Vector2Int
    prj_sequence = MLP_SEQUENCE  # panorama image heights

    arg_in_const: ManagedNDArray
    arg_inout_mlp_i: ManagedNDArray
    arg_inout_mlp_d: ManagedNDArray

    arg_inout_anchor_rgb: ManagedNDArray
    arg_inout_anchor_depth: ManagedNDArray

    arg_in_anchor_acc_grid: ManagedNDArray
    arg_in_anchor_xyz: ManagedNDArray
    arg_in_anchor_acc: ManagedNDArray

    arg_out_canvas: ManagedNDArray

    def __init__(self, init_ambient_color) -> None:
        self.canvas_size = canvas_size

        self.arg_in_const = cuda.managed_array((10), dtype=np.float32)
        self.arg_in_const[0] = self.canvas_size.x
        self.arg_in_const[1] = self.canvas_size.y
        self.arg_in_const[2] = 0  # offset
        self.arg_in_const[3] = self.prj_sequence[0] * \
            2  # current projection width
        # current projection height
        self.arg_in_const[4] = self.prj_sequence[0]

        # Projection offsets
        self.arg_in_const[5] = len(self.prj_sequence)  # num of projections
        for i in range(len(self.prj_sequence)):
            self.arg_in_const[6 + i] = self.prj_sequence[i]

        mlp_len = sum([h * h * 2 for h in self.prj_sequence])
        self.arg_inout_mlp_i = cuda.managed_array((mlp_len), dtype=np.uint32)
        self.arg_inout_mlp_d = cuda.managed_array((mlp_len), dtype=np.float32)

        # anchor arguments
        t = init_ambient_color[np.newaxis, :]
        t = t.repeat(N_ANCHORS, axis=0)
        self.arg_inout_anchor_rgb = cuda.managed_array(
            (N_ANCHORS, 3), dtype=np.uint8)
        self.arg_inout_anchor_rgb[::] = t

        self.arg_inout_anchor_depth = cuda.managed_array(
            (N_ANCHORS), dtype=np.float32)
        self.arg_inout_anchor_depth.fill(10000)

        self.arg_in_anchor_xyz = cuda.managed_array(
            (N_ANCHORS, 3), dtype=np.float32)
        self.arg_in_anchor_xyz[::] = anchor_xyz

        self.arg_in_anchor_acc_grid = cuda.managed_array(
            (self.canvas_size.y, self.canvas_size.x), dtype=np.int)

        self.arg_in_anchor_acc = cuda.managed_array(
            (canvas_size.y, canvas_size.x, N_ANCHOR_NEIGHBORS),
            dtype=np.uint16)
        self.arg_in_anchor_acc[::] = anchor_acc

        # Result canvas
        self.arg_out_canvas = cuda.managed_array(
            (canvas_size.y, canvas_size.x, 3),
            dtype=np.float32)

    def update(self, in_pcd_xyz: ManagedNDArray, in_pcd_rgb: ManagedNDArray) -> np.ndarray:
        n_thread = 1024
        n_block = (in_pcd_xyz.shape[0] + (n_thread - 1)) // n_thread

        for i, h in enumerate(self.prj_sequence):
            # Run projection kernel on each level

            offset = sum([v * v * 2 for v in self.prj_sequence[:i]])

            self.arg_in_const[2] = offset  # offset
            self.arg_in_const[3] = h * 2  # current projection width
            self.arg_in_const[4] = h  # current projection height

            dpcp_cuda[n_block, n_thread](
                self.arg_inout_mlp_i,
                self.arg_inout_mlp_d,
                self.arg_in_const,
                in_pcd_xyz)

            cuda.current_context().synchronize()

        # Run data merging kernel
        n_thread = (32, 32)
        n_block = (self.canvas_size.x // 32, self.canvas_size.y // 32)
        merge_dpcp_cuda[n_block, n_thread](
            self.arg_out_canvas,
            self.arg_in_const,
            self.arg_inout_mlp_i,
            self.arg_inout_mlp_d,
            in_pcd_rgb)

        cuda.current_context().synchronize()

    def update_anchors(self, in_sp_xyz, in_sp_rgb):
        print('---------------->', in_sp_xyz.shape[0])

        n_thread = in_sp_xyz.shape[0]
        n_block = 1

        cuda_update_anchors[n_block, n_thread](
            self.arg_inout_anchor_rgb,
            self.arg_inout_anchor_depth,
            self.arg_in_const,
            self.arg_in_anchor_acc_grid,
            in_sp_xyz,
            in_sp_rgb)

        cuda.current_context().synchronize()

    def clean_up(self) -> None:
        # Clean up index and depth buffers
        self.arg_inout_mlp_i[::] = 0
        self.arg_inout_mlp_d[::] = NEAR_FIELD_CLIP_DST

        # clean up canvas
        self.arg_out_canvas[::] = canvas

        n_thread = (32, 32)
        n_block = (self.canvas_size.x // 32, self.canvas_size.y // 32)
        draw_far_field[n_block, n_thread](
            self.arg_out_canvas,
            self.arg_in_anchor_acc,
            self.arg_in_anchor_xyz,
            self.arg_inout_anchor_rgb)

        cuda.current_context().synchronize()
