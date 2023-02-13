import math
import numba as nb
from numba import cuda

from configs import NEAR_FIELD_CLIP_DST


@cuda.jit()
def cuda_gen_pcd_xyz(out_xyz, in_const, in_cam_matrix, in_depth):
    i = cuda.grid(1)

    c_frame_width = int(in_const[0])
    c_frame_height = int(in_const[1])
    c_offset = int(in_const[2])
    c_upsample_rate = in_const[3]

    depth_width = int(c_frame_width / c_upsample_rate)
    depth_height = int(c_frame_height / c_upsample_rate)

    if i >= out_xyz.shape[0]:
        return

    u = nb.float32(i % c_frame_width)
    v = c_frame_height - nb.float32(i // c_frame_width)

    fx, fy, cx, cy = in_cam_matrix[:4]
    ctw = in_cam_matrix[4:]

    # Upsample depth bi-linear
    dx = (i % c_frame_width) / c_upsample_rate
    dy = (i // c_frame_width) / c_upsample_rate

    x0 = math.floor(dx)
    y0 = math.floor(dy)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = max(min(x0, depth_width - 1), 0)
    x1 = max(min(x1, depth_width - 1), 0)
    y0 = max(min(y0, depth_height - 1), 0)
    y1 = max(min(y1, depth_height - 1), 0)

    da = in_depth[int(y0 * depth_width + x0)]
    db = in_depth[int(y1 * depth_width + x0)]
    dc = in_depth[int(y0 * depth_width + x1)]
    dd = in_depth[int(y1 * depth_width + x1)]

    wa = (x1 - dx) * (y1 - dy)
    wb = (x1 - dx) * (dy - y0)
    wc = (dx - x0) * (y1 - dy)
    wd = (dx - x0) * (dy - y0)

    d = wa * da + wb * db + wc * dc + wd * dd

    th = 0.015
    c = math.fabs(da - d) > th\
        or math.fabs(db - d) > th\
        or math.fabs(dc - d) > th\
        or math.fabs(dd - d) > th
    d = NEAR_FIELD_CLIP_DST if c else d

    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d

    # OMG, Numba doesn't support vectorized type
    # And it doesn't support matmul either???
    out_xyz[c_offset + i, 0] = ctw[0] * x + \
        ctw[1] * y + ctw[2] * z + ctw[3] * 1
    out_xyz[c_offset + i, 1] = ctw[4] * x + \
        ctw[5] * y + ctw[6] * z + ctw[7] * 1
    out_xyz[c_offset + i, 2] = ctw[8] * x + \
        ctw[9] * y + ctw[10] * z + ctw[11] * 1


@cuda.jit()
def cuda_gen_pcd_rgb(out_rgb, in_const, in_y, in_cbcr):
    u, v = cuda.grid(2)

    c_frame_width = int(in_const[0])
    c_offset = int(in_const[2])

    if v < in_y.shape[0] and u < in_y.shape[1]:
        y = in_y[v, u]
        cb = in_cbcr[v // 2, u // 2, 0]
        cr = in_cbcr[v // 2, u // 2, 1]

        i = c_offset + v * c_frame_width + u

        out_rgb[i, 0] = nb.uint8(
            min(max(y + 1.40200 * (cr - 0x80), 0), 255))
        out_rgb[i, 1] = nb.uint8(
            min(max(y - 0.34414 * (cb - 0x80) - 0.71414 * (cr - 0x80), 0), 255))
        out_rgb[i, 2] = nb.uint8(
            min(max(y + 1.77200 * (cb - 0x80), 0), 255))
