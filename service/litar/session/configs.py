from __future__ import annotations

import numpy as np
from uuid import uuid4, UUID
from numba import cuda
from numba.cuda.cudadrv.devicearray import ManagedNDArray

from litar.types import Vector2Int
from service.schema import SessionInitPackage


class SessionConfigs:
    s_id: UUID

    num_of_views: int
    exp_time_window: int

    depth_native_size: Vector2Int
    depth_upsample_rate: int

    color_dense_sample_size: Vector2Int
    color_sparse_sample_size: Vector2Int

    n_points: int

    k: np.ndarray
    ambient_color: np.ndarray  # uint8, (3)

    def __init__(self, configs: SessionInitPackage) -> None:
        self.s_id = uuid4()

        self.num_of_views = configs.num_of_views
        self.exp_time_window = configs.exp_time_window

        self.depth_native_size = configs.depth_native_size
        self.depth_upsample_rate = configs.color_dense_size.x / configs.depth_native_size.x

        self.color_dense_sample_size = configs.color_dense_size
        self.color_sparse_sample_size = configs.color_sparse_size

        self.n_points = int(self.color_dense_sample_size.x *
                            self.color_dense_sample_size.y)

        self.k = configs.k
        self.ambient_color = configs.ambient_color

        # self.cuda_const = cuda.managed_array((10), dtype=np.float32)
        # self.cuda_const[0] = self.color_dense_sample_size.x
        # self.cuda_const[1] = self.color_dense_sample_size.y
        # self.cuda_const[2] = 0 # Offset
        # self.cuda_const[3] = self.depth_upsample_rate
