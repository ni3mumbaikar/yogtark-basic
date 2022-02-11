import tensorflow as tf
import numpy as np
import cv2

poses = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
model_path = "../lite-model_movenet_singlepose_lightning_3.tflite"
# model_path = "../lite-model_movenet_singlepose_thunder_3.tflite"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()


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

EDGES = [(0, 1), (0, 2), (1, 3), (2, 4),  (5, 7),
         (7, 9), (6, 8), (8, 10), (5, 6), (5,
                                           11), (6, 12), (11, 12), (11, 13),
         (13, 15), (12, 14), (14, 16)]


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    # img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    input_image = tf.cast(img, dtype=tf.float32)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Rendering
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    # Create model interpreter and allocate tensors to it
    yg_model_path = "./output/tf_lite_model_yogtark.tflite"
    yg_interpreter = tf.lite.Interpreter(yg_model_path)
    yg_interpreter.allocate_tensors()

    # Setup input and output
    yg_input_details = yg_interpreter.get_input_details()
    yg_output_details = yg_interpreter.get_output_details()
    # print(yg_output_details)

    newkp = np.squeeze(keypoints_with_scores)
    featureVector = []
    for kp in newkp:
        if(kp[2] > 0.5):
            featureVector.append(kp[1])
            featureVector.append(kp[0])
        else:
            featureVector.append(0)
            featureVector.append(0)
    featureVector = np.array(featureVector, dtype=np.float32).reshape((1, 34))

    # Pose classification
    yg_interpreter.set_tensor(
        yg_input_details[0]['index'], np.array(featureVector))
    yg_interpreter.invoke()
    yg_pose = yg_interpreter.get_tensor(yg_output_details[0]['index'])
    # print(yg_pose)
    a = yg_pose[0]
    pose_prediction = np.interp(a, (a.min(), a.max()), (0, 1)).tolist()
    # pose_prediction = np.array(pose_prediction, dtype=np.float16)
    # print(pose_prediction)
    # maximum = np.max(pose_prediction)
    # print(maximum)
    # index_of_maximum = np.where(pose_prediction == maximum)
    maxpos = pose_prediction.index(max(pose_prediction))
    print(poses[maxpos])

    cv2.imshow('MoveNet Lightning', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
