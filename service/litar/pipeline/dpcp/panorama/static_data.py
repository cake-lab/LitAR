from __future__ import annotations

import os
import math
import numpy as np
import pyreality as pr
from configs import N_ANCHORS, N_ANCHOR_NEIGHBORS
from configs import PANORAMA_WIDTH, PANORAMA_HEIGHT

from litar.types import Vector2Int

# TODO: implement a decorator to wrap these functions


def canvas_equirectangular_panorama(height):
    u = np.arange(height * 2, dtype=int)
    v = np.arange(height, dtype=int)
    uv = np.stack(np.meshgrid(u, v), axis=-1)

    uv_xyz = pr.equirectangular_uv_to_cartesian(uv)
    uv_xyz = pr.euler_rotation_xyz(
        uv_xyz,
        (math.radians(-90), math.radians(0), 0))
    uv_xyz = pr.euler_rotation_xyz(
        uv_xyz,
        (math.radians(0), math.radians(90), 0))

    # right-hand to left-hand
    # UE4 uses left-hand
    uv_xyz = uv_xyz * [-1, 1, -1]

    return uv_xyz.reshape((height, height * 2, 3))


def generate_anchor_acc(p_size: Vector2Int):
    canvas = canvas_equirectangular_panorama(p_size.y)
    anchor_xyz = pr.fibonacci_sphere(N_ANCHORS)

    # if os.path.exists(f'./tmp/anchor_acc_{N_ANCHORS}.npy'):
    #     print('Loading far field acceleration data...')
    #     return canvas, anchor_xyz, np.load(f'./tmp/anchor_acc_{N_ANCHORS}.npy')

    print('Generating far field acceleration data...')
    n_samples = N_ANCHOR_NEIGHBORS
    anchor_acc = np.zeros((p_size.y, p_size.x, n_samples), dtype=np.uint16)

    for v in range(p_size.y):
        for u in range(p_size.x):
            anchor_acc[v, u] = np.argpartition(
                canvas[v, u] @ anchor_xyz.T, -n_samples, axis=-1)[-n_samples:]

    # np.save(f'./tmp/anchor_acc_{N_ANCHORS}.npy', anchor_acc)

    return canvas, anchor_xyz, anchor_acc


def make_acc_grid(n_anchors, v_acc_grid):
    if os.path.exists(f'./tmp/acc_grid_{n_anchors}.npy'):
        print('Loading acceleration grid...')
        return np.load(f'./tmp/acc_grid_{n_anchors}.npy')

    anchors = pr.fibonacci_sphere(n_anchors)

    v = np.arange(v_acc_grid)
    v = v / np.max(v) * np.pi
    u = np.arange(v_acc_grid * 2)
    u = u / np.max(u) * np.pi * 2

    uv = np.stack(np.meshgrid(u, v), axis=-1)
    uv = np.stack((uv[:, :, 1], uv[:, :, 0]), axis=-1)
    r = np.ones_like(uv).mean(axis=-1, keepdims=True)

    uvr = np.concatenate((uv, r), axis=-1)
    uvr_flat = uvr.reshape((-1, 3))
    uvr_cart = pr.spherical_to_cartesian(uvr_flat)

    # Build cache grid
    uvr_neighbors = np.array([np.argmax(anchors @ uvr_cart[i])
                             for i in range(uvr_cart.shape[0])])
    acc_grid = uvr_neighbors.reshape(v_acc_grid, v_acc_grid * 2)

    np.save(f'./tmp/acc_grid_{n_anchors}.npy', acc_grid)

    return acc_grid


acc_grid = make_acc_grid(N_ANCHORS, PANORAMA_HEIGHT)


canvas_size = Vector2Int(PANORAMA_WIDTH, PANORAMA_HEIGHT)
canvas, anchor_xyz, anchor_acc = generate_anchor_acc(canvas_size)
