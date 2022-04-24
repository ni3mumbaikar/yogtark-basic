import numpy
import numpy as np


def normalize_zero_one(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def normalize_2d(data):
    # Only this is changed to use 2-norm put 2 instead of 1
    # norm = np.linalg.norm(matrix, 2)
    # normalized matrix
    # matrix = matrix / norm
    return (data - np.min(data)) / (np.max(data) - np.min(data))

    # return matrix


def processWithCenterOfBody(keypoints: np.ndarray, width, height, center_threshold=0.5):
    np.set_printoptions(suppress=True)

    # To scale up coordinates
    normalize(keypoints, width, height)
    # To calculate center of the body
    centerBody = getBodyCentre(keypoints, center_threshold)
    # To get distance of each kp from center
    distance_list = get_distance_from_centre(keypoints, centerBody)
    distance_list = normalize_2d(distance_list)
    return [centerBody, distance_list]


def get_distance_from_centre(keypoints, centerBody):
    distanceList = []
    # print(keypoints)
    for keypoint in keypoints:
        kpnp = np.array([keypoint[1], keypoint[0]])
        centernp = np.array(centerBody)
        distance = np.subtract(kpnp, centernp)
        distanceList.append(distance.tolist())

    # print('Distance', np.array(distanceList))

    return distanceList


def normalize(keypoints, width, height):
    for keypoint in keypoints:
        keypoint[1] = int(keypoint[1] * height)
        keypoint[0] = int(keypoint[0] * width)


def getBodyCentre(keypoints, center_threshold=0.5):
    if keypoints.shape != (17, 3):
        raise Exception("Keypoints array is not in required shape is (17,3) Current shape is ", keypoints.shape)
    else:
        if keypoints[11][2] > center_threshold and keypoints[12][2] > center_threshold:
            centerBody = [(keypoints[11][1] + keypoints[12][1]) / 2, (keypoints[11][0] + keypoints[12][0]) / 2]
            print('Center of the body is', centerBody[0], centerBody[1])
        else:
            raise Exception("Important keypoints are not available with threshold confidence value or above")
    return centerBody
