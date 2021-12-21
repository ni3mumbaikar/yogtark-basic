import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, False)

# Setup camera
cap = cv2.VideoCapture(0)

# Setup interpreter
# model_path = "../lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite"
model_path = "../lite-model_movenet_singlepose_lightning_3.tflite"
# model_path = "../lite-model_movenet_singlepose_thunder_3.tflite"

interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# While loop
while cap.isOpened():
    width  = cap.get(3)  # float `width`
    height = cap.get(4)  # float `height`

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Reshape image
    input_image = frame.copy()
    # input_image = np.expand_dims(input_image, axis=0)
    input_image = tf.image.resize_with_pad(
        np.expand_dims(input_image, axis=0), 192, 192)
        # np.expand_dims(input_image, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.float32)
    # input_image = tf.cast(input_image, dtype=tf.uint8)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print('Input', input_details)
    # print('Output', output_details)

    # keypoints prediction
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    shaped_keypoints = np.squeeze(np.multiply(keypoints,[height,width,1]))

    for keypoint in shaped_keypoints:
        # x = int(keypoint[1] * width)
        # y = int(keypoint[0] * height)
        if(keypoint[2] > 0.5):
            cv2.circle(frame, (int(keypoint[1]),int(keypoint[0])), 4, (0, 0, 255), -1)

    KEYPOINT_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7),(7, 9), (6, 8), (8, 10), (5, 6), (5,11),
    (6, 12), (11, 12), (11, 13),(13, 15), (12, 14), (14, 16)]

    for edge in KEYPOINT_EDGES:
        print(edge)

        x1 = int(shaped_keypoints[edge[0]][1])# * width)
        y1 = int(shaped_keypoints[edge[0]][0])# * height)

        x2 = int(shaped_keypoints[edge[1]][1])# * width)
        y2 = int(shaped_keypoints[edge[1]][0])# * height)

        if(shaped_keypoints[edge[0]][2] >= 0.5 and shaped_keypoints[edge[1]][2] >= 0.5):
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the captured image / rendering the frame
    cv2.imshow('YogTark', frame)

    # wait for the key and come out of the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Quit cv2
# cv2.release()
cv2.destroyAllWindows()
