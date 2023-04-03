from __future__ import annotations

import imageio
import numpy as np
from numba import cuda

from litar.session.configs import SessionConfigs
from litar.pipeline.dpcp import ManagedPanoramaRenderer
from litar.pipeline.pointcloud import ManagedPointCloudGenerator

from service.schema.keyframe import FarFieldKeyFramePackage
from service.schema.keyframe import NearFieldKeyFramePackage

from profilehooks import profile, timecall


class LightingReconstructionSession:
    configs: SessionConfigs

    renderer: ManagedPanoramaRenderer
    far_field_pc_generator: ManagedPointCloudGenerator
    near_filed_pc_generator: ManagedPointCloudGenerator

    def __init__(self, configs: SessionConfigs) -> None:
        self.configs = configs

        # Near field point cloud generator
        self.near_filed_pc_generator = ManagedPointCloudGenerator(
            configs.k,
            configs.depth_native_size,
            configs.color_dense_sample_size,
            configs.num_of_views)

        # Far field point cloud generator
        sparse_to_dense_ratio = configs.color_sparse_sample_size.x / \
            configs.color_dense_sample_size.x
        self.far_field_pc_generator = ManagedPointCloudGenerator(
            configs.k * sparse_to_dense_ratio,
            configs.color_sparse_sample_size,
            configs.color_sparse_sample_size,
            1)

        self.far_field_static_depth = cuda.managed_array(
            (configs.color_sparse_sample_size.y *
             configs.color_sparse_sample_size.x), dtype=np.float32)
        self.far_field_static_depth = np.ones(
            (configs.color_sparse_sample_size.y *
             configs.color_sparse_sample_size.x), dtype=np.float32)

        # Environment map renderer, using panorama for now
        self.renderer = ManagedPanoramaRenderer(configs.ambient_color)
        self.renderer.clean_up()

    # @timecall(immediate=True)
    def reconstruct_near_field_pcd(self, pkg: NearFieldKeyFramePackage):
        self.near_filed_pc_generator.exec(
            pkg.view_index, pkg.buf_trs,
            pkg.buf_depth, pkg.buf_y, pkg.buf_cbcr)

        ds = self.configs.color_dense_sample_size.x // self.configs.color_sparse_sample_size.x
        m, spc_rgb = self.near_filed_pc_generator.sample_to_anchor(
            self.renderer.arg_in_anchor_xyz,
            downsample_rate=ds * ds,
            filter_surroundings=True)
        self.renderer.arg_inout_anchor_rgb[m, :] = spc_rgb

        # sp_xyz, sp_rgb = self.near_filed_pc_generator.sparse_sample(
        #     downsample_rate=ds * ds)
        # self.renderer.update_anchors(sp_xyz, sp_rgb)
        self.renderer.clean_up()

    # @timecall(immediate=True)
    def reconstruct_far_field_pcd(self, pkg: FarFieldKeyFramePackage):
        m_r = pkg.buf_r.reshape((3, 3))
        m_t = np.zeros((3, 1), dtype=np.float32)
        m_trs = np.concatenate((m_r, m_t), axis=1)
        m_trs = m_trs.reshape(-1)

        self.far_field_pc_generator.exec(
            0, m_trs,
            self.far_field_static_depth,
            pkg.buf_y, pkg.buf_cbcr)

        m, spc_rgb = self.far_field_pc_generator.sample_to_anchor(
            self.renderer.arg_in_anchor_xyz, downsample_rate=1)
        self.renderer.arg_inout_anchor_rgb[m, :] = spc_rgb

        # sp_xyz, sp_rgb = self.near_filed_pc_generator.sparse_sample()
        # self.renderer.update_anchors(sp_xyz, sp_rgb)
        self.renderer.clean_up()

    # @timecall(immediate=True)
    def run_direct_point_cloud_projection(self):
        self.renderer.update(self.near_filed_pc_generator.arg_out_pcd_xyz,
                             self.near_filed_pc_generator.arg_out_pcd_rgb)

    def clean_up_renderer(self):
        self.renderer.clean_up()

    def dump_point_cloud(self):
        c_offset = int(self.near_filed_pc_generator.arg_in_const[2])
        np.save('./tmp/pcd.npy', np.concatenate(
            (self.near_filed_pc_generator.arg_out_pcd_xyz[c_offset:c_offset + self.configs.n_points],
             self.near_filed_pc_generator.arg_out_pcd_rgb[c_offset:c_offset + self.configs.n_points] / 255),
            axis=1).astype(np.float32))

    def dump_anchors(self):
        np.save('./tmp/anchors.npy', np.concatenate((
            self.renderer.arg_in_anchor_xyz,
            self.renderer.arg_inout_anchor_rgb / 255
        ), axis=-1).astype(np.float32))

    def dump_env_map(self, name='envmap'):
        imageio.imsave(f'./tmp/{name}.png',
                       self.renderer.arg_out_canvas.astype(np.uint8))
