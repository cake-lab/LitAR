import math
import numba as nb
from numba import cuda

from configs import NEAR_FIELD_CLIP_DST
from configs import NEAR_FIELD_SIZE_HALF


@cuda.jit
def cuda_update_anchors(out_anchor_rgb, out_anchor_depth,
                        in_const, in_acc_grid, in_spc_xyz, in_spc_rgb):
    i = cuda.grid(1)

    prj_w = int(in_const[3])
    prj_h = int(in_const[4])

    x = in_spc_xyz[i, 0]
    y = in_spc_xyz[i, 1]
    z = in_spc_xyz[i, 2]

    r = math.hypot(math.hypot(x, y), z)
    phi = math.atan2(y, x)
    theta = math.acos(z / r)

    cu = round(math.degrees(phi) % 360 / 360.0 * (prj_w - 1))
    cv = round(math.degrees(theta) / 180.0 * (prj_h - 1))
    i_a = in_acc_grid[cv, cu]

    rr = out_anchor_depth[i_a]
    if r < rr:
        out_anchor_depth[i_a] = r
        out_anchor_rgb[i_a, 0] = in_spc_rgb[i, 0]
        out_anchor_rgb[i_a, 1] = in_spc_rgb[i, 1]
        out_anchor_rgb[i_a, 2] = in_spc_rgb[i, 2]


@cuda.jit
def dpcp_cuda(out_mlp_i, out_mlp_d, in_const, in_pcd_xyz):
    i = cuda.grid(1)

    if i >= in_pcd_xyz.shape[0]:
        return

    offset = int(in_const[2])
    prj_w = int(in_const[3])
    prj_h = int(in_const[4])

    x = in_pcd_xyz[i, 0]
    y = in_pcd_xyz[i, 1]
    z = in_pcd_xyz[i, 2]

    dd = NEAR_FIELD_SIZE_HALF

    # Filter near field distances
    r = math.hypot(math.hypot(x, y), z)

    r = r if x < dd and x > -dd else NEAR_FIELD_CLIP_DST
    r = r if y < dd and y > -dd else NEAR_FIELD_CLIP_DST
    r = r if z < dd and z > -dd else NEAR_FIELD_CLIP_DST

    v = math.acos(y / r)
    u = math.atan2(x, z)
    r = r

    v = int(v / math.pi * prj_h)
    u = int((u + math.pi) / (math.pi * 2) * prj_w)

    iuv = offset + v * prj_w + u
    # pi = out_mlp_i[iuv]
    pd = out_mlp_d[iuv]

    if r < NEAR_FIELD_CLIP_DST:
        if r > pd or pd >= NEAR_FIELD_CLIP_DST:
            out_mlp_i[iuv] = i
            out_mlp_d[iuv] = r


@cuda.jit
def merge_dpcp_cuda(out_canvas, in_const, in_mlp_i, in_mlp_d, in_pcd_rgb):
    u, v = cuda.grid(2)

    c_size_y = int(in_const[1])
    n_prj = int(in_const[5])
    im, dm = 0, NEAR_FIELD_CLIP_DST

    for i in range(n_prj):
        prj_h = int(in_const[6 + i])
        prj_w = int(prj_h * 2)

        s = c_size_y // prj_h

        offset = 0
        for j in range(i):
            offset += int(in_const[6 + j] * in_const[6 + j] * 2)

        p_iuv = int(offset + v // s * prj_w + u // s)
        p_d = in_mlp_d[p_iuv]

        if p_d < dm:
            im = p_iuv
            dm = p_d

    if dm < NEAR_FIELD_CLIP_DST:
        ii = in_mlp_i[im]

        out_canvas[v, u, 0] = in_pcd_rgb[ii, 0]
        out_canvas[v, u, 1] = in_pcd_rgb[ii, 1]
        out_canvas[v, u, 2] = in_pcd_rgb[ii, 2]


@cuda.jit()
def draw_far_field(out_canvas_norms, in_anchor_acc, in_anchor_xyz, in_anchor_rgb):
    u, v = cuda.grid(2)

    n = out_canvas_norms[v, u]
    w, r, g, b = 0, 0, 0, 0

    # Should NOT do this, don't know why
    # out_canvas_norms[v, u, 0] = 0
    # out_canvas_norms[v, u, 1] = 0
    # out_canvas_norms[v, u, 2] = 0

    # for i in range(anchor_xyz.shape[0]):
    #     i_anchor = i

    for i in range(in_anchor_acc.shape[2]):
        i_anchor = in_anchor_acc[v, u, i]
        a = in_anchor_xyz[i_anchor]

        # dot canvas pixel norm with anchor vector to get cos value
        c = n[0] * a[0] + n[1] * a[1] + n[2] * a[2]
        c = max(c, 0) ** 128

        w += c
        r += in_anchor_rgb[i_anchor, 0] * c
        g += in_anchor_rgb[i_anchor, 1] * c
        b += in_anchor_rgb[i_anchor, 2] * c

    out_canvas_norms[v, u, 0] = r / w
    out_canvas_norms[v, u, 1] = g / w
    out_canvas_norms[v, u, 2] = b / w
