import os

import imageio
import numpy as np


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            # frame = env.render(
            #     mode="rgb_array",
            #     height=self.height,
            #     width=self.width,
            #     camera_id=self.camera_id,
            # )
            frame = env.env.env._get_viewer()._read_pixels_as_in_window((256, 256))

            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            print(path)
            imageio.mimsave(path, self.frames, fps=self.fps)
