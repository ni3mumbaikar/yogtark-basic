import numpy
import numpy as np


def processWithCenterOfBody(keypoints: np.ndarray, width, height, center_threshold=0.5):
    np.set_printoptions(suppress=True)
    # print('Keypoints', np.array(keypoints))

    # To scale up coordinates
    normalize(keypoints, width, height)
    # To calculate center of the body
    centerBody = getBodyCentre(keypoints, center_threshold)
    # To get distance of each kp from center
    distance_list = get_distance_from_centre(keypoints, centerBody)

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
