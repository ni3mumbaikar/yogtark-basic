import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import pandas as pd
import os


# This module provides csv generation functionality for diffrent classes


class csv_generator:

    """
    This is parameterized constructor for csv generator classs
    @__self : current instance context
    @classpath : path to be passed of a specific class
    """

    def __init__(self, classname, classpath):
        print('Loaded new class : ', classpath)
        self.classpath = classpath
        self.classname = classname

    # Actual method to generate csv files for each sample of present at classpath

    def generate_csv(self):

        # if dataset_csv and its child directories are not present in the system then this block creates new directory structure for storing csv
        if not os.path.exists('./dataset_csv'):
            os.mkdir('./dataset_csv')
        csv_path = './dataset_csv/'+self.classname
        if not os.path.exists(csv_path):
            os.mkdir(csv_path)

        filenames = [img for img in glob.glob(self.classpath+'/*.jpeg')]
        filenames.sort()
        images = []
        count = 0
        for img in filenames:
            df = pd.DataFrame({'Y': [], 'X': [], 'Z': []})
            # if count > 1:
            # break
            print(self.classname, ':', 'Generating csv for', 'image number',
                  str(count), 'out of', str(len(filenames)))
            image = tf.io.read_file(img)
            image = tf.image.decode_jpeg(image)
            input_image = tf.expand_dims(image, axis=0)
            input_image = tf.image.resize_with_pad(input_image, 256, 256)

            # model_path = "lite-model_movenet_singlepose_thunder_3.tflite"
            model_path = "lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite"
            interpreter = tf.lite.Interpreter(model_path)
            interpreter.allocate_tensors()

            input_image = tf.cast(input_image, dtype=tf.uint8)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(
                input_details[0]['index'], input_image.numpy())
            interpreter.invoke()
            # print(output_details[0])
            keypoints = interpreter.get_tensor(output_details[0]['index'])

            keypoint_string = ['nose ', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                               'left_wrist',  'right_wrist',  'left_hip', 'right_hip', 'left_knee', 'right_knee',  'left_ankle', 'right_ankle']

            # print(keypoints[0][0])

            width = 640
            height = 640

            KEYPOINT_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4),  (5, 7),
                              (7, 9), (6, 8), (8, 10), (5, 6), (5,
                                                                11), (6, 12), (11, 12), (11, 13),
                              (13, 15), (12, 14), (14, 16)]

            input_image = tf.expand_dims(image, axis=0)
            input_image = tf.image.resize_with_pad(input_image, width, height)
            input_image = tf.cast(input_image, dtype=tf.uint8)

            image_np = np.squeeze(input_image.numpy(), axis=0)
            image_np = cv2.resize(image_np, (width, height))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            for keypoint in keypoints[0][0]:

                s_row = pd.Series(keypoint, index=df.columns)
                df = df.append(s_row, ignore_index=True)
                x = int(keypoint[1] * width)
                y = int(keypoint[0] * height)

                cv2.circle(image_np, (x, y), 4, (0, 0, 255), -1)

            for edge in KEYPOINT_EDGES:

                x1 = int(keypoints[0][0][edge[0]][1] * width)
                y1 = int(keypoints[0][0][edge[0]][0] * height)

                x2 = int(keypoints[0][0][edge[1]][1] * width)
                y2 = int(keypoints[0][0][edge[1]][0] * height)

                cv2.line(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # cv2.imshow("pose estimation", image_np)
            # cv2.waitKey()
            df['Keypoints'] = pd.Series(keypoint_string)
            df.to_csv('./dataset_csv/'+self.classname +
                      '/'+self.classname+'_'+str(count)+'.csv')
            count = count + 1
