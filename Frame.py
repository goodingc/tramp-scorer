from __future__ import annotations

from typing import Optional, Callable, Tuple, List

import cv2
import numpy as np
from tf_pose.common import CocoPart
from tf_pose.estimator import TfPoseEstimator, Human

from Logger import Logger


class Frame:
    image: np.ndarray

    def __init__(self, batch_index: int, batch_frame_index: int, width: int, height: int, bed_y: int, focus_width: int,
                 focus_height: int, logger: Logger):
        self.batch_index = batch_index
        self.batch_frame_index = batch_frame_index
        self.width = width
        self.height = height
        self.bed_y = bed_y
        self.focus_width = focus_width
        self.focus_height = focus_height

        self.position: Optional[np.ndarray] = None
        self.smoothed_position: Optional[np.ndarray] = None
        self.extremities: Optional[np.ndarray] = None
        self.position_confidence: Optional[float] = None
        self.extremities_confidence: Optional[float] = None
        self.angle: Optional[float] = None
        self.y: Optional[float] = None
        self.dy: Optional[float] = None
        self.skill_index: Optional[int] = None
        self.skill_angle: Optional[float] = None
        self.focus_image: Optional[np.ndarray] = None
        self.poses: Optional[List[Human]] = None
        self.closest_pose: Optional[Human] = None
        self.knee_angle: Optional[float] = None
        self.hip_angle: Optional[float] = None

        self.prev_frame: Optional[Frame] = None
        self.next_frame: Optional[Frame] = None

        self.logger = logger

    def set_adjacent_frames(self, prev_frame: Optional[Frame], next_frame: Optional[Frame]):
        self.prev_frame = prev_frame
        self.next_frame = next_frame

    def get_next_valid_frame(self, test: Callable[[Frame], bool], prev_frame=False) -> Optional[Frame]:
        return self.get_next_valid_frame_with_distance(test, prev_frame)[0]

    def get_next_valid_frame_with_distance(self, test: Callable[[Frame], bool], prev_frame=False) -> Tuple[
        Optional[Frame], int]:
        candidate_frame = self.prev_frame if prev_frame else self.next_frame
        distance = -1 if prev_frame else 1
        if candidate_frame is None:
            return None, distance
        while not test(candidate_frame):
            candidate_frame = candidate_frame.prev_frame if prev_frame else candidate_frame.next_frame
            distance += -1 if prev_frame else 1
            if candidate_frame is None:
                return None, distance
        return candidate_frame, distance

    def set_image(self, image: np.ndarray, left_cutoff: int, right_cutoff: int):
        self.image = image[:, left_cutoff:right_cutoff, :]

    @staticmethod
    def get_contour_centroid(contour):
        moments = cv2.moments(contour)
        return np.array([int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])])

    def find_performer(self, backplate: np.ndarray, opening_kernel: np.ndarray):
        self.extremities_confidence = 0
        self.position_confidence = 0

        diff = cv2.subtract(self.image, backplate)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        ret, threshold = cv2.threshold(diff_gray, 7, 255, 0)

        clean_threshold = cv2.erode(threshold, opening_kernel)
        clean_threshold = cv2.dilate(clean_threshold, opening_kernel)

        contours, hierarchy = cv2.findContours(clean_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        biggest_contour = None
        biggest_contour_area = 0

        for candidate_contour in contours:
            contour_area = cv2.contourArea(candidate_contour)
            if contour_area > biggest_contour_area:
                biggest_contour = candidate_contour
                biggest_contour_area = contour_area

        if biggest_contour is not None:
            included_contours = []

            weighted_centroid = np.array([0, 0])
            area_sum = 0

            extremities_distance = 0
            extremities = None

            for candidate_contour in contours:
                candidate_extremities_distance = extremities_distance
                candidate_extremities = []

                candidate_accepted = False
                for biggest_contour_point in biggest_contour[::15]:
                    for candidate_contour_point in candidate_contour[::15]:
                        contour_point_distance = np.linalg.norm(
                            candidate_contour_point[0] - biggest_contour_point[0])

                        if contour_point_distance > candidate_extremities_distance:
                            candidate_extremities_distance = contour_point_distance
                            candidate_extremities = [candidate_contour_point[0], biggest_contour_point[0]]

                        if contour_point_distance < 40:
                            candidate_accepted = True

                if candidate_accepted:
                    if candidate_extremities_distance > extremities_distance:
                        extremities_distance = candidate_extremities_distance
                        extremities = candidate_extremities

                    candidate_contour_area = int(cv2.contourArea(candidate_contour))
                    weighted_centroid += Frame.get_contour_centroid(
                        candidate_contour) * candidate_contour_area
                    area_sum += candidate_contour_area
                    included_contours.append(candidate_contour)




            self.position = (weighted_centroid / area_sum).astype(np.int32)
            self.position_confidence = min(area_sum / 5000.0, 1) ** 0.5

            self.extremities = extremities

            if self.extremities is None:
                return


            extremities_average = (self.extremities[0] + self.extremities[1]) / 2
            extremities_average_dist = np.linalg.norm(extremities_average - self.position)

            if extremities_average_dist < 100:
                self.extremities_confidence = (1 - extremities_average_dist / 100) * self.position_confidence

                a = self.extremities[0] - self.extremities[1]
                self.angle = np.rad2deg(np.arctan2(a[0], a[1]))

    def smooth_position(self, smoothing_factor: float):
        weighted_positions = np.array([[0, 0]])
        weights_sum = 0

        prev_frame = self.get_next_valid_frame(lambda f: f.position is not None and f.position_confidence > 0.1, True)
        next_frame = self.get_next_valid_frame(lambda f: f.position is not None and f.position_confidence > 0.1)

        if prev_frame is not None:
            weight = prev_frame.position_confidence * smoothing_factor
            weighted_positions = np.append(weighted_positions, [prev_frame.position * weight], 0)
            weights_sum += weight
        if next_frame is not None:
            weight = next_frame.position_confidence * smoothing_factor
            weighted_positions = np.append(weighted_positions, [next_frame.position * weight], 0)
            weights_sum += weight
        if self.position is not None and self.position_confidence > 0.1:
            weighted_positions = np.append(weighted_positions, [self.position * self.position_confidence], 0)
            weights_sum += self.position_confidence

        self.smoothed_position = weighted_positions.sum(0) / weights_sum
        self.smoothed_position[1] = min(self.smoothed_position[1], self.bed_y)
        self.y = self.bed_y - self.smoothed_position[1]

        if self.angle is None or self.extremities_confidence < 0.1:
            prev_frame, prev_frame_dist = self.get_next_valid_frame_with_distance(lambda f: f.angle is not None, True)
            next_frame, next_frame_dist = self.get_next_valid_frame_with_distance(lambda f: f.angle is not None)

            if prev_frame is None:
                self.angle = next_frame.angle
            elif next_frame is None:
                self.angle = prev_frame.angle
            else:
                self.angle = (prev_frame.angle * next_frame_dist + next_frame.angle * -prev_frame_dist) / \
                             (next_frame_dist - prev_frame_dist)

    def find_dy(self, prev_frame: Optional[Frame], next_frame: Optional[Frame], frame_dist: int):
        self.dy = (next_frame.y - prev_frame.y) / frame_dist

    def find_skill_index(self, carry_index: int, carry_ascending: bool) -> Tuple[int, bool]:
        if self.prev_frame is None:
            carry_ascending = self.y < self.next_frame.y
        elif self.next_frame is not None:
            if self.prev_frame.dy < 0 and self.next_frame.dy >= 0 and not carry_ascending and self.y < 100:
                carry_ascending = True
                carry_index += 1
            elif self.prev_frame.dy > 0 and self.next_frame.dy <= 0 and carry_ascending and self.y > 100:
                carry_ascending = False

        self.skill_index = carry_index
        return carry_index, carry_ascending

    def find_skill_angle(self, skill_angles: np.ndarray) -> np.ndarray:
        if self.prev_frame is None:
            self.skill_angle = 0
            return skill_angles

        if self.extremities is None:
            if self.prev_frame.skill_index < self.skill_index:
                skill_angles = np.append(skill_angles, 0)
            self.skill_angle = skill_angles[-1]
            return skill_angles

        delta_angle = self.angle - self.prev_frame.angle
        if (delta_angle < -90 or delta_angle > 90) and not (delta_angle < -270 or delta_angle > 270):
            extremities = [self.extremities[1], self.extremities[0].copy()]
            a = extremities[0] - extremities[1]
            self.angle = np.rad2deg(np.arctan2(a[0], a[1]))

        delta_angle = self.angle - self.prev_frame.angle

        if -300 < delta_angle < 300:
            skill_angles[-1] += delta_angle

        if self.prev_frame.skill_index < self.skill_index:
            skill_angles = np.append(skill_angles, 0)

        self.skill_angle = skill_angles[-1]

        return skill_angles

    def find_focus(self):
        rot_mat = cv2.getRotationMatrix2D((self.smoothed_position[0], self.smoothed_position[1]), -self.angle, 1.0)
        result = cv2.warpAffine(self.image, rot_mat, self.image.shape[1::-1], flags=cv2.INTER_LINEAR)

        focus_margin = ((self.focus_width / 2) ** 2 + (self.focus_height / 2) ** 2) ** 0.5 + 20

        left_border = int(max(focus_margin - self.smoothed_position[0], 0))
        right_border = int(max(self.smoothed_position[0] - (self.width - focus_margin), 0))
        top_border = int(max(focus_margin - self.smoothed_position[1], 0))
        bottom_border = int(max(self.smoothed_position[1] - (self.height - focus_margin), 0))
        if sum([left_border, right_border, top_border, bottom_border]) > 0:
            result = cv2.copyMakeBorder(result, top_border, bottom_border, left_border, right_border,
                                        cv2.BORDER_CONSTANT, None, [0, 0, 0])
        self.focus_image = result[int(self.smoothed_position[1] - 150) + top_border: int(self.smoothed_position[1] + 150) + top_border,
                           int(self.smoothed_position[0] - 150) + left_border: int(self.smoothed_position[0] + 150) + left_border].copy()

        # cv2.imshow("x", self.focus_image)
        # cv2.waitKey(0)

    @staticmethod
    def pose_part_pos(pose: Human, part: CocoPart) -> Optional[np.ndarray]:
        pose_part = pose.body_parts.get(part.value)
        if pose_part is None: return None
        return np.array([pose_part.x, pose_part.y])

    def find_pose(self, estimator: TfPoseEstimator):
        if self.poses is None:
            self.poses = estimator.inference(self.focus_image, resize_to_default=True, upsample_size=4)

        closest_pose = None
        closest_pose_distance = 0.2

        frame_center = np.array([0.5, 0.5])

        for pose in self.poses:
            neck_pos = Frame.pose_part_pos(pose, CocoPart.Neck)
            left_hip_pos = Frame.pose_part_pos(pose, CocoPart.LHip)
            right_hip_pos = Frame.pose_part_pos(pose, CocoPart.RHip)

            if neck_pos is not None and \
                    left_hip_pos is not None and \
                    right_hip_pos is not None:
                center = ((left_hip_pos + right_hip_pos) / 2 + neck_pos) / 2

                pose_distance = np.linalg.norm(center - frame_center)
                if pose_distance < closest_pose_distance:
                    closest_pose = pose
                    closest_pose_distance = pose_distance

        self.closest_pose = closest_pose

    def closest_pose_part_pos(self, part: CocoPart) -> Optional[np.ndarray]:
        if self.closest_pose is None: return None
        return Frame.pose_part_pos(self.closest_pose, part)

    def pose_joint_angle(self, external_part_a: CocoPart, external_part_b: CocoPart, internal_part: CocoPart) -> \
            Optional[np.ndarray]:
        external_a_coords = self.closest_pose_part_pos(external_part_a)
        external_b_coords = self.closest_pose_part_pos(external_part_b)
        internal_coords = self.closest_pose_part_pos(internal_part)

        if external_a_coords is None or external_b_coords is None or internal_coords is None:
            return None

        ba = external_a_coords - external_b_coords
        bi = internal_coords - external_b_coords

        return np.degrees(np.arccos(np.dot(ba, bi) / (np.linalg.norm(ba) * np.linalg.norm(bi))))

    def find_shape(self):
        if self.closest_pose is None:
            return None

        left_knee_angle = self.pose_joint_angle(CocoPart.LHip, CocoPart.LAnkle, CocoPart.LKnee) or np.nan
        right_knee_angle = self.pose_joint_angle(CocoPart.RHip, CocoPart.RAnkle, CocoPart.RKnee) or np.nan

        left_hip_angle = self.pose_joint_angle(CocoPart.LKnee, CocoPart.Neck, CocoPart.LHip) or np.nan
        right_hip_angle = self.pose_joint_angle(CocoPart.RKnee, CocoPart.Neck, CocoPart.RHip) or np.nan

        if left_knee_angle is not np.nan or right_knee_angle is not np.nan:
            self.knee_angle = np.nanmean([left_knee_angle, right_knee_angle])

        if left_hip_angle is not np.nan or right_hip_angle is not np.nan:
            self.hip_angle = np.nanmean([left_hip_angle, right_hip_angle])
