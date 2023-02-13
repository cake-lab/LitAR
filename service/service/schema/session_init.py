import numpy as np
from dataclasses import dataclass
from litar.types import Vector2Int
from service.schema.utils import BasePackage


@dataclass
class SessionInitPackage(BasePackage):
    num_of_views: int
    exp_time_window: int
    near_field_size: int

    depth_native_size: Vector2Int
    color_dense_size: Vector2Int
    color_sparse_size: Vector2Int

    k: np.ndarray
    ambient_color: np.ndarray  # uint8, 3

    identifier: int = 0b0000_0000

    def __init__(self, raw_bytes: bytes):
        super(SessionInitPackage, self).__init__(raw_bytes)

        header_len = 1
        config_len = 3 * 4
        ambient_info_len = 2 * 4
        matrix_k_len = 4 * 4
        img_sizes_len = 6 * 4
        offset = header_len

        # Start decoding package
        cfg = np.frombuffer(
            raw_bytes[offset:offset + config_len],
            dtype=np.int32)
        self.num_of_views = cfg[0]
        self.exp_time_window = cfg[1]
        self.near_field_size = cfg[2]
        offset += config_len

        ambient_info = np.frombuffer(
            raw_bytes[offset:offset + ambient_info_len],
            dtype=np.float32)
        self.ambient_color = self.ambient_info_to_rgb(ambient_info)
        offset += ambient_info_len

        self.k = np.frombuffer(
            raw_bytes[offset:offset + matrix_k_len],
            dtype=np.float32)
        offset += matrix_k_len

        sizes = np.frombuffer(
            raw_bytes[offset:offset + img_sizes_len],
            dtype=np.int32)

        self.depth_native_size = Vector2Int(int(sizes[0]), int(sizes[1]))
        self.color_dense_size = Vector2Int(int(sizes[2]), int(sizes[3]))
        self.color_sparse_size = Vector2Int(int(sizes[4]), int(sizes[5]))

    def ambient_info_to_rgb(self, info):
        temperature = info[0]
        brigtheness = info[1]

        rgb = np.zeros(3, dtype=np.uint8)

        t = temperature / 100

        if t <= 66:
            rgb[0] = 255
            rgb[1] = t - 2
            rgb[1] = -155.25485562709179 - 0.44596950469579133 * \
                rgb[1] + 104.49216199393888 * np.log(rgb[1])
            rgb[2] = 0

            if t > 20:
                rgb[2] = t - 10
                rgb[2] = -254.76935184120902 + 0.8274096064007395 * \
                    rgb[2] + 115.67994401066147 * np.log(rgb[2])

        else:
            rgb[0] = t - 55
            rgb[0] = 351.97690566805693 + 0.114206453784165 * \
                rgb[0] - 40.25366309332127 * np.log(rgb[0])
            rgb[1] = t - 50
            rgb[1] = 325.4494125711974 + 0.07943456536662342 * \
                rgb[1] - 28.0852963507957 * np.log(rgb[1])
            rgb[2] = 255

        rgb = np.clip(brigtheness * rgb, 0, 255)

        return rgb

    def __str__(self) -> str:
        return '<SessionInitPackage>\n' +\
            '\n'.join([
                f'num_of_views: {self.num_of_views}',
                f'exp_time_window: {self.exp_time_window}',
                f'near_field_size: {self.near_field_size}',
                f'ambient_color: {self.ambient_color}',
                f'k: {self.k}',
                f'depth_native_size: {self.depth_native_size}',
                f'color_dense_size: {self.color_dense_size}',
                f'color_sparse_size: {self.color_sparse_size}'
            ])
