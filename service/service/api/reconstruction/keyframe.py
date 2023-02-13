"""Service API handler for reconstruction keyframes."""
import io
import os
import time
import shutil
import datetime
import imageio
import numpy as np
import open3d as o3d
import tornado.websocket
from uuid import UUID
from PIL import Image
from profilehooks import profile, timecall

from configs import PANORAMA_WIDTH
from litar.session import SessionConfigs
from litar.session import LightingReconstructionSession

from service.schema import SessionInitPackage
from service.schema import NearFieldKeyFramePackage
from service.schema.keyframe import FarFieldKeyFramePackage


class ReconKeyframeWSHandler(tornado.websocket.WebSocketHandler):
    __timestamp: str
    __debug: bool = False
    __n_frame: int = 0

    session: LightingReconstructionSession

    def open(self):
        """Handling socket opening."""
        print('! WebSocket Opened')

        self._t_start = time.time()
        self._t_stamp = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
        os.makedirs(f'./tmp/session_recording/{self._t_stamp}')

        # shutil.rmtree('./tmp/session')
        # os.mkdir('./tmp/session')

    # @profile(immediate=True)
    def on_message(self, message):
        """Called when new message received
        message: bytes | str
        """
        if type(message) is str:
            self.on_string_received(message)
            return

        # Debugging purpose
        if self.__debug:
            self.dump_message(message)

        # Dispatch message
        if message[0] == 0b0000_0000:  # Session Init
            pkg = SessionInitPackage(message)
            self.on_session_init(pkg)

        # Near field keyframe
        elif message[0] == 0b0001_0000:
            pkg = NearFieldKeyFramePackage(
                message,
                self.session.configs.color_dense_sample_size,
                self.session.configs.depth_native_size)

            self.on_near_field_keyframe_received(pkg)

        # Far field keyframe
        elif message[0] == 0b0001_0001:
            pkg = FarFieldKeyFramePackage(
                message,
                self.session.configs.color_sparse_sample_size)

            self.on_far_field_keyframe_received(pkg)

        else:
            print(f'! Unrecognized package with header {message[0]}')
            return

        self.__n_frame += 1

    def on_close(self):
        """Handling socket closes."""
        print('! WebSocket Closed')

    def on_string_received(self, message: str):
        print(f'< {message}')

    def on_session_init(self, pkg: SessionInitPackage):
        s_configs = SessionConfigs(pkg)
        self.session = LightingReconstructionSession(s_configs)

        print(f'! New session initialized: {s_configs.s_id}\n\n{pkg}\n')
        self.write_message(b'\x01' + str(s_configs.s_id).encode(), binary=True)

    # @profile(immediate=True)
    def on_near_field_keyframe_received(self, pkg: NearFieldKeyFramePackage):
        print(f'! New near field keyframe: {pkg}')

        self.session.reconstruct_near_field_pcd(pkg)
        self.session.run_direct_point_cloud_projection()

        env_map = self.convert_env_map_for_unity()
        self.write_message(b'\x10' + env_map, binary=True)

        if self.__debug:
            self.session.dump_point_cloud()

    def on_far_field_keyframe_received(self, pkg: FarFieldKeyFramePackage):
        print(f'! New far field keyframe: {pkg}')

        self.session.reconstruct_far_field_pcd(pkg)
        self.session.run_direct_point_cloud_projection()

        env_map = self.convert_env_map_for_unity()
        self.write_message(b'\x10' + env_map, binary=True)

        if self.__debug:
            self.session.dump_anchors()

    def convert_env_map_for_unity(self, encode_jpg=True):
        env_map = np.copy(self.session.renderer.arg_out_canvas)
        env_map = env_map.astype(np.uint8)
        # env_map = np.flipud(env_map)

        shift = PANORAMA_WIDTH // 4
        env_map = np.concatenate(
            (env_map[:, shift:, :], env_map[:, :shift, :]), axis=1)

        t_now = time.time()
        t_elapsed = t_now - self._t_start

        # Save the envmap updates for comparisons
        if self.__debug:
            np.save(
                f'./tmp/session_recording/{self._t_stamp}/{t_elapsed}_envmap.npy', env_map)

        # Other operations
        if self.__debug:
            imageio.imsave('./tmp/envmap.png', env_map)

        if encode_jpg:
            return self.encode_env_map_jpg(env_map)
        else:
            return env_map.tobytes()

    @staticmethod
    def encode_env_map_jpg(env_map: np.ndarray):
        """Convert input environment map array to JPEG encoded bytes."""
        img = Image.fromarray(env_map)

        with io.BytesIO() as f_output:
            img.save(f_output, format='JPEG')
            contents = f_output.getvalue()

        return contents

    def dump_message(self, message: bytes):
        t = np.frombuffer(message, dtype=np.uint8)

        np.save(
            f'./tmp/session_recording/{self._t_stamp}/{self.__n_frame}.npy', t)


keyframe_index_ws_routes = [
    (r"keyframe/", ReconKeyframeWSHandler)
]

__all__ = ['keyframe_index_ws_routes']
