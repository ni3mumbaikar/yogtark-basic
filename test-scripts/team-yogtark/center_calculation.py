import numpy as np
from data import BodyPart

MIN_CROP_KEYPOINT_SCORE = 0.5


def calculate_center(originalkp: np.ndarray):
    keypoints = []
    print(originalkp)

    return keypoints


def _torso_visible(self, keypoints: np.ndarray) -> bool:
    """Checks whether there are enough torso keypoints.

    This function checks whether the model is confident at predicting one of
    the shoulders/hips which is required to determine a good crop region.

    Args:
      keypoints: Detection result of Movenet model.

    Returns:
      True/False
    """
    left_hip_score = keypoints[BodyPart.LEFT_HIP.value, 2]
    right_hip_score = keypoints[BodyPart.RIGHT_HIP.value, 2]
    left_shoulder_score = keypoints[BodyPart.LEFT_SHOULDER.value, 2]
    right_shoulder_score = keypoints[BodyPart.RIGHT_SHOULDER.value, 2]

    left_hip_visible = left_hip_score > MIN_CROP_KEYPOINT_SCORE
    right_hip_visible = right_hip_score > MIN_CROP_KEYPOINT_SCORE
    left_shoulder_visible = left_shoulder_score > MIN_CROP_KEYPOINT_SCORE
    right_shoulder_visible = right_shoulder_score > MIN_CROP_KEYPOINT_SCORE

    return ((left_hip_visible or right_hip_visible) and
            (left_shoulder_visible or right_shoulder_visible))
