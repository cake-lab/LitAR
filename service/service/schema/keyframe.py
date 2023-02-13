import numpy as np
from dataclasses import dataclass
from litar.types import Vector2Int
from service.schema.utils import BasePackage


@dataclass
class NearFieldKeyFramePackage(BasePackage):
    view_index: int
    buf_trs: np.ndarray  # float32, 3x4

    buf_y: np.ndarray  # uint8, n_pixels
    buf_cbcr: np.ndarray  # uint8, n_pixels * 0.5
    buf_depth: np.ndarray  # depth, float32

    identifier: int = 0b0001_0000

    def __init__(self, raw_bytes: bytes, c_size: Vector2Int, d_size: Vector2Int) -> None:
        super(NearFieldKeyFramePackage, self).__init__(raw_bytes)

        header_len = 1
        view_index_len = 4
        buf_trs_len = 3 * 4 * 4
        buf_y_len = c_size.x * c_size.y
        buf_cbcr_len = int(c_size.x * c_size.y * 0.5)
        buf_depth_len = d_size.x * d_size.y * 4
        offset = header_len

        self.view_index = np.frombuffer(
            raw_bytes[offset:offset + view_index_len],
            dtype=np.int32)[0]
        offset += view_index_len

        self.buf_trs = np.frombuffer(
            raw_bytes[offset:offset + buf_trs_len],
            dtype=np.float32)
        offset += buf_trs_len

        self.buf_y = np.frombuffer(
            raw_bytes[offset:offset + buf_y_len],
            dtype=np.uint8).reshape((c_size.y, c_size.x))
        offset += buf_y_len

        self.buf_cbcr = np.frombuffer(
            raw_bytes[offset:offset + buf_cbcr_len],
            dtype=np.uint8).reshape((c_size.y // 2, c_size.x // 2, 2))
        offset += buf_cbcr_len

        self.buf_depth = np.frombuffer(
            raw_bytes[offset:offset + buf_depth_len],
            dtype=np.float32)

    def __str__(self) -> str:
        return '\n'.join([
            f'view_index: {self.view_index}',
            f'buf_trs: {self.buf_trs.shape}',
            f'buf_y: {self.buf_y.shape}',
            f'buf_cbcr: {self.buf_cbcr.shape}',
            f'buf_depth: {self.buf_depth.shape}'
        ])


@dataclass
class FarFieldKeyFramePackage(BasePackage):
    buf_r: np.ndarray  # float32, 3x4

    buf_y: np.ndarray  # uint8, n_pixels
    buf_cbcr: np.ndarray  # uint8, n_pixels * 0.5

    identifier: int = 0b0001_0001

    def __init__(self, raw_bytes: bytes, c_size: Vector2Int) -> None:
        super(FarFieldKeyFramePackage, self).__init__(raw_bytes)

        header_len = 1
        buf_r_len = 3 * 3 * 4
        buf_y_len = c_size.x * c_size.y
        buf_cbcr_len = int(c_size.x * c_size.y * 0.5)
        offset = header_len

        self.buf_r = np.frombuffer(
            raw_bytes[offset:offset + buf_r_len],
            dtype=np.float32)
        offset += buf_r_len

        self.buf_y = np.frombuffer(
            raw_bytes[offset:offset + buf_y_len],
            dtype=np.uint8).reshape((c_size.y, c_size.x))
        offset += buf_y_len

        self.buf_cbcr = np.frombuffer(
            raw_bytes[offset:offset + buf_cbcr_len],
            dtype=np.uint8).reshape((c_size.y // 2, c_size.x // 2, 2))

    def __str__(self) -> str:
        return '\n'.join([
            f'buf_r: {self.buf_r.shape}',
            f'buf_y: {self.buf_y.shape}',
            f'buf_cbcr: {self.buf_cbcr.shape}'
        ])
