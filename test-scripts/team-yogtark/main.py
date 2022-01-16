import os
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

"""
This script is intended to be used for data generation with image files stored in ../../dataset_images /['TEST' and 'TRAIN']
Kindly follow the comments and do not edit any code unless you are sure what you are doing
"""
poses = ['downdog','goddess','plank','tree','warrior2']


def generateIndividualCsv(classname,isTest):
    folder_path = '../../dataset_images/'
    if isTest:
        folder_path= folder_path +'TEST/'+classname
    else:
        folder_path=folder_path+'TRAIN/'+classname

    if not os.path.exists('../../dataset_csv'):
        os.mkdir('../../dataset_csv')
        os.mkdir('../../dataset_csv/TRAIN')
        os.mkdir('../../dataset_csv/TEST')

    pathcheck(classname,isTest)

    filenames = [img for img in glob.glob(folder_path+'/*.jpeg')]
    filenames.sort()
    images = []
    count = 0
    for img in filenames:
        df = pd.DataFrame({'Value': []})
        print(classname, ':', 'Generating csv for', 'image number',
              str(count), 'out of', str(len(filenames)))
        image = tf.io.read_file(img)
        image = tf.image.decode_jpeg(image)
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, 256, 256)

        # model_path = "lite-model_movenet_singlepose_thunder_3.tflite"
        model_path = "../../lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite"
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

        keypoint_string = [
            'noseX', 'noseY',
            'left_eyeX', 'left_eyeY',
            'right_eyeX', 'right_eyeY',
            'left_earX', 'left_earY',
            'right_earX', 'right_earY',
            'left_shoulderX', 'left_shoulderY',
            'right_shoulderX', 'right_shoulderY',
            'left_elbowX', 'left_elbowY',
            'right_elbowX', 'right_elbowY',
            'left_wristX', 'left_wristY',
            'right_wristX', 'right_wristY',
            'left_hipX', 'left_hipY',
            'right_hipX', 'right_hipY',
            'left_kneeX', 'left_kneeY',
            'right_kneeX', 'right_kneeY',
            'left_ankleX', 'left_ankleY',
            'right_ankleX', 'right_ankleY']



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

            x_row = pd.Series(keypoint[1], index=df.columns)
            y_row = pd.Series(keypoint[0], index=df.columns)
            df = df.append(x_row, ignore_index=True)
            df = df.append(y_row, ignore_index=True)
            x = int(keypoint[1] * width)
            y = int(keypoint[0] * height)

            cv2.circle(image_np, (x, y), 4, (0, 0, 255), -1)

        for edge in KEYPOINT_EDGES:

            x1 = int(keypoints[0][0][edge[0]][1] * width)
            y1 = int(keypoints[0][0][edge[0]][0] * height)

            x2 = int(keypoints[0][0][edge[1]][1] * width)
            y2 = int(keypoints[0][0][edge[1]][0] * height)

            cv2.line(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # comment next two lines if you don't want to see output of detected images directly use the output
        # cv2.imshow("pose estimation", image_np)
        # cv2.waitKey()

        df['Keypoints'] = pd.Series(keypoint_string)
        if not isTest:
            df.to_csv(folder_path.replace('dataset_images','dataset_csv')+'/'+classname+'_'+str(count)+'.csv')
        else:
            df.to_csv(folder_path.replace('dataset_images','dataset_csv')+'/'+classname+'_'+str(count)+'.csv')
        count = count + 1


def pathcheck(classname,isTest):

    if not isTest:
        csv_path = '../../dataset_csv/TRAIN/'+classname
    else:
        csv_path = '../../dataset_csv/TEST/'+classname
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)

# pose_label=0
for pose in poses:
    generateIndividualCsv(pose,True)
    generateIndividualCsv(pose,False)
    # pose_label+=1
