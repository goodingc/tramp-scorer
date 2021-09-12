from __future__ import annotations

import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator

from Logger import Logger
from Frame import Frame

from typing import List, Optional, Tuple, Callable


class Batch:
    frames: List[Frame]
    backplate: np.ndarray

    def __init__(self, index: int, size: int, frame_width: int, frame_height: int, bed_y: int, focus_width: int, focus_height: int, logger: Logger):
        self.index = index
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.bed_y = bed_y
        self.focus_width = focus_width
        self.focus_height = focus_height
        self.frames = [Frame(index, frame_index, frame_width, frame_height, bed_y, focus_width, focus_height, logger) for frame_index in
                       range(size)]

        self.prev_batch: Optional[Batch] = None
        self.next_batch: Optional[Batch] = None

        self.logger = logger
        self.phase: str

    def set_adjacent_batches(self, prev_batch: Optional[Batch], next_batch: Optional[Batch]):
        self.prev_batch = prev_batch
        self.next_batch = next_batch

        for index in range(self.size):
            self.frames[index].set_adjacent_frames(
                self.get_relative_frame(index - 1),
                self.get_relative_frame(index + 1)
            )

    def get_relative_frame(self, index: int) -> Optional[Frame]:
        if index < 0:
            if self.prev_batch is None:
                return None
            return self.prev_batch.get_relative_frame(index + self.prev_batch.size)
        elif index < self.size:
            return self.frames[index]

        if self.next_batch is None:
            return None
        return self.next_batch.get_relative_frame(index - self.size)

    def load_frames(self, video_reader: cv2.VideoCapture, left_cutoff: int, right_cutoff: int):
        for batch_frame_index in range(self.size):
            success, image = video_reader.read()
            if not success:
                raise Exception("Error reading batch frame %d" % batch_frame_index)

            self.frames[batch_frame_index].set_image(image, left_cutoff, right_cutoff)

        self.logger.success("Loaded %d frames" % self.size)

    def find_backplate(self, frame_gap: int, cooldown: int):
        backplate_frame_count = max(1, int((self.size - cooldown) / frame_gap))
        self.logger.info("Finding %d frame backplate" % backplate_frame_count)

        backplate_buffer = np.zeros((backplate_frame_count, self.frame_height, self.frame_width, 3), np.uint8)

        for backplate_frame_index in range(backplate_frame_count):
            backplate_buffer[backplate_frame_index, :, :, :] = self.frames[backplate_frame_index * frame_gap].image

        self.backplate = np.median(backplate_buffer, 0).astype(np.uint8)
        self.logger.success("Found backplate")

    def find_performer(self):
        timer = self.logger.start_timer("Find performer")
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        for index in range(self.size):
            self.frames[index].find_performer(self.backplate, opening_kernel)

        timer.end()

    def smooth_positions(self, smoothing_factor: float):
        timer = self.logger.start_timer("Smooth positions")

        for index in range(self.size):
            self.frames[index].smooth_position(smoothing_factor)


        timer.end()

    def differentiate_ys(self):
        timer = self.logger.start_timer("Find y'")

        norm_start = 0
        norm_end = self.size
        if self.prev_batch is None:
            norm_start = 2
            self.frames[0].find_dy(self.frames[0], self.frames[1], 1)
            self.frames[1].find_dy(self.frames[0], self.frames[2], 2)
        if self.next_batch is None:
            norm_end -= 2
            self.frames[-1].find_dy(self.frames[-2], self.frames[-1], 1)
            self.frames[-2].find_dy(self.frames[-3], self.frames[1], 2)

        for index in range(norm_start, norm_end):
            self.frames[index].find_dy(self.get_relative_frame(index - 2), self.get_relative_frame(index + 2), 4)

        timer.end()

    def index_skills(self, carry_index: int, carry_ascending: bool) -> Tuple[int, bool]:
        timer = self.logger.start_timer("Index skills")

        for index in range(self.size):
            carry_index, carry_ascending = self.frames[index].find_skill_index(carry_index, carry_ascending)

            # if index % 50 == 0:
            #     report(index / self.size)

        timer.end()
        return carry_index, carry_ascending

    def find_skill_angles(self, skill_angles: np.ndarray) -> np.ndarray:
        timer = self.logger.start_timer("Skill angles")

        for frame in self.frames:
            skill_angles = frame.find_skill_angle(skill_angles)

        timer.end()
        return skill_angles

    def focus_images(self):
        timer = self.logger.start_timer("Focus images")

        for index in range(self.size):
            self.frames[index].find_focus()

        timer.end()

    def find_poses(self, estimator: TfPoseEstimator):
        timer = self.logger.start_timer("Find poses")

        for index in range(self.size):
            self.frames[index].find_pose(estimator)
            self.logger.info("Detected pose in %.2f%% of frames" % (index / self.size * 100))

        timer.end()

    def find_shapes(self):
        timer = self.logger.start_timer("Find shapes")

        for frame in self.frames:
            frame.find_shape()

        timer.end()
