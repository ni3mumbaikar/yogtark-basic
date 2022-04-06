import tensorflow as tf
import numpy as np
import cv2
from Preprocessor import processWithCenterOfBody
import threading
import pyttsx3

from speaker import *

"""
This script is intended to be used for running yogtark classifier with camera interface of machine
Kindly follow the comments and do not edit any code unless you are sure what you are doing

This script deals with

1. Leveraging the yogtark classfication model to detect the current pose

"""

np.set_printoptions(suppress=True)

# s = speaker()

eng = pyttsx3.init()
eng.say("Please take your position in 5 seconds")
eng.runAndWait()

poses = ['downdog', 'plank', 'tree', 'goddess', 'warrior2', 'no_pose']
# poses = ['downdog', 'plank', 'tree', 'goddess', 'warrior2', 'no_pose']
# model_path = "../lite-model_movenet_singlepose_thunder_3.tflite"
model_path = "../lite-model_movenet_singlepose_lightning_3.tflite"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
# Setup input and output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    # for edge, color in edges.items():
    for edge in edges:
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)),
                     (int(x2), int(y2)), (0, 0, 255), 2)


# EDGES = {
#     (0, 1): 'm',
#     (0, 2): 'c',
#     (1, 3): 'm',
#     (2, 4): 'c',
#     (0, 5): 'm',
#     (0, 6): 'c',
#     (5, 7): 'm',
#     (7, 9): 'm',
#     (6, 8): 'c',
#     (8, 10): 'c',
#     (5, 6): 'y',
#     (5, 11): 'm',
#     (6, 12): 'c',
#     (11, 12): 'y',
#     (11, 13): 'm',
#     (13, 15): 'm',
#     (12, 14): 'c',
#     (14, 16): 'c'
# }

EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7),
         (7, 9), (6, 8), (8, 10), (5, 6), (5,
                                           11), (6, 12), (11, 12), (11, 13),
         (13, 15), (12, 14), (14, 16)]


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    # print(keypoints)
    # print(shaped)

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (255, 255, 255), -1)


def implementYg(keypoints_with_scores):
    # Create model interpreter and allocate tensors to it
    yg_model_path = "./output/tf_lite_model_yogtark.tflite"
    yg_interpreter = tf.lite.Interpreter(yg_model_path)
    yg_interpreter.allocate_tensors()

    # Setup input and output
    yg_input_details = yg_interpreter.get_input_details()
    yg_output_details = yg_interpreter.get_output_details()
    # print(yg_output_details)

    # newkp = np.squeeze(keypoints_with_scores[0][0])

    width = 640
    height = 640

    # try:
    [centerBody, distanceList] = processWithCenterOfBody(
        keypoints_with_scores[0][0], width, height, 0.7)
    # except:
    #     print('Skipping image due to low confidence score')
    #     continue

    featureVector = []

    for distance in distanceList:
        # x_row = pd.Series(distance[0], index=df.columns)
        # y_row = pd.Series(distance[1], index=df.columns)
        featureVector.append(distance[0])
        featureVector.append(distance[1])
        # df = df.append(x_row, ignore_index=True)
        # df = df.append(y_row, ignore_index=True)

    # for kp in newkp:
    #     # if kp[2] > 0.5:
    #     featureVector.append(kp[1])
    #     featureVector.append(kp[0])
    #     # else:
    #     #     featureVector.append(0)
    #     #     featureVector.append(0)

    featureVector = np.array(featureVector, dtype=np.float32).reshape((1, 34))

    # Pose classification
    yg_interpreter.set_tensor(
        yg_input_details[0]['index'], np.array(featureVector))
    yg_interpreter.invoke()
    yg_pose = yg_interpreter.get_tensor(yg_output_details[0]['index'])
    # print(yg_pose)
    a = yg_pose[0]
    print(a)

    pose_prediction = np.interp(a, (a.min(), a.max()), (0, 1)).tolist()
    # pose_prediction = np.array(pose_prediction, dtype=np.float16)
    maximum = np.max(pose_prediction)
    # print(maximum)
    index_of_maximum = np.where(pose_prediction == maximum)
    maxpos = pose_prediction.index(max(pose_prediction))
    print(poses[maxpos])


cap = cv2.VideoCapture(0)

isFirstFrame = True


def first_frame_logic():
    global isFirstFrame

    s = speaker()

    t1 = threading.Thread(
        target=s.speak(), daemon=True)

    t2 = threading.Thread(
        target=render(isFirstFrame), daemon=True)

    t2.start()
    t1.start()

    isFirstFrame = False


def render(flag):
    ret, frame = cap.read()

    # Reshape image
    # img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
    # img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 256, 256)
    input_image = tf.cast(img, dtype=tf.float32)

    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Rendering
    draw_connections(frame, keypoints_with_scores[0][0], EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores[0][0], 0.4)

    if keypoints_with_scores[0][0][11][2] < 0.7 or keypoints_with_scores[0][0][12][2] < 0.7:
        print('Please be in frame properly')
        cv2.imshow('MoveNet Lightning', frame)
    else:
        implementYg(keypoints_with_scores)
        cv2.imshow('MoveNet Lightning', frame)


while cap.isOpened():
    if isFirstFrame:
        first_frame_logic()
    else:
        render(isFirstFrame)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
