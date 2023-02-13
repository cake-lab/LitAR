import math
import numba as nb
from numba import cuda

from configs import NEAR_FIELD_CLIP_DST
from configs import NEAR_FIELD_SIZE_HALF


@cuda.jit(device=True)
def cuda_xyz_to_cube_iuv(out_iuv, in_xyz):
    # Code copied from:
    # https://en.wikipedia.org/wiki/Cube_mapping

    x = in_xyz[0]
    y = in_xyz[1]
    z = in_xyz[2]

    abs_x = math.fabs(x)
    abs_y = math.fabs(y)
    abs_z = math.fabs(z)

    is_x_positive = 1 if x > 0 else 0
    is_y_positive = 1 if y > 0 else 0
    is_z_positive = 1 if z > 0 else 0

    maxAxis = 0

    # POSITIVE X
    if is_x_positive and abs_x >= abs_y and abs_x >= abs_z:
        # u (0 to 1) goes from +z to -z
        # v (0 to 1) goes from -y to +y
        maxAxis = abs_x
        uc = -z
        vc = y
        out_iuv[0] = 0

    # NEGATIVE X
    if not is_x_positive and abs_x >= abs_y and abs_x >= abs_z:
        # u (0 to 1) goes from -z to +z
        # v (0 to 1) goes from -y to +y
        maxAxis = abs_x
        uc = z
        vc = y
        out_iuv[0] = 1

    # POSITIVE Y
    if is_y_positive and abs_y >= abs_x and abs_y >= abs_z:
        # u (0 to 1) goes from -x to +x
        # v (0 to 1) goes from +z to -z
        maxAxis = abs_y
        uc = x
        vc = -z
        out_iuv[0] = 2

    # NEGATIVE Y
    if not is_y_positive and abs_y >= abs_x and abs_y >= abs_z:
        # u (0 to 1) goes from -x to +x
        # v (0 to 1) goes from -z to +z
        maxAxis = abs_y
        uc = x
        vc = z
        out_iuv[0] = 3

    # POSITIVE Z
    if is_z_positive and abs_z >= abs_x and abs_z >= abs_y:
        # u (0 to 1) goes from -x to +x
        # v (0 to 1) goes from -y to +y
        maxAxis = abs_z
        uc = x
        vc = y
        out_iuv[0] = 4

    # NEGATIVE Z
    if not is_z_positive and abs_z >= abs_x and abs_z >= abs_y:
        # u (0 to 1) goes from +x to -x
        # v (0 to 1) goes from -y to +y
        maxAxis = abs_z
        uc = -x
        vc = y
        out_iuv[0] = 5

    # Convert range from -1 to 1 to 0 to 1
    out_iuv[1] = 0.5 * (uc / maxAxis + 1.0)
    out_iuv[2] = 0.5 * (vc / maxAxis + 1.0)


@cuda.jit()
def cuda_dpcp_cubemap(out_cubemap, in_xyz, in_rgb):
    iuv = cuda.local.array((3), dtype=nb.float32)

    side = 256

    i = cuda.grid(1)

    if i < in_xyz.shape[0]:
        cuda_xyz_to_cube_iuv(iuv, in_xyz[i])

        i, uf, vf = iuv[0], iuv[1], iuv[2]
        u, v = int(uf * (side - 1)), int(vf * (side - 1))

        out_cubemap[i, v, u, :] = in_rgb[i]


@cuda.jit()
def dpcp_cuda(out_i, out_d, pcd_xyz, height):
    i = cuda.grid(1)

    if i < pcd_xyz.shape[0]:
        x, y, z = pcd_xyz[i]

        dd = NEAR_FIELD_SIZE_HALF
        r = math.hypot(math.hypot(x, y), z)

        r = r if x < dd and x > -dd else NEAR_FIELD_CLIP_DST
        r = r if y < dd and y > -dd else NEAR_FIELD_CLIP_DST
        r = r if z < dd and z > -dd else NEAR_FIELD_CLIP_DST

        v = math.acos(y / r)
        u = math.atan2(x, z)
        r = r

        v = int(v / math.pi * height)
        u = int((u + math.pi) / math.pi * height)

        pi = out_i[v, u]
        pd = out_d[v, u]

        # if r < pd:
        #     out_i[v, u] = i
        #     out_d[v, u] = r if pd >= NEAR_FIELD_CLIP_DST else pd
        # else:
        #     out_i[v, u] = pi
        #     out_d[v, u] = r

        if r < NEAR_FIELD_CLIP_DST:
            if r > pd or pd == NEAR_FIELD_CLIP_DST:
                out_i[v, u] = i
                out_d[v, u] = r
