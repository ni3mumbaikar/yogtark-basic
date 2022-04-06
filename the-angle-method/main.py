import os
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import csv
from class_yogatark import yogtark
from Preprocessor import processWithCenterOfBody

"""
This script is intended to be used for data generation with image files stored in ../../dataset_images /['TEST' and
'TRAIN'] Kindly follow the comments and do not edit any code unless you are sure what you are doing 

This script deals with

1. Reading individual image and generating a csv file for each image
2. Merge individual csv files that were generated for each image in first point and produce final training_set.csv
   and testing_set.csv
3. Now the Model Start processing the csv i.e. actual training and evaluation phase begins here using the model
   skeleton defined in class_csv file
4. Running a camera interface while utilizing the yogtark classifier which was generated as output of third point

"""
poses = ['downdog', 'plank', 'tree', 'goddess', 'warrior2', 'no_pose']

# Toggle this to generate csv for individual images
generateCSV = True


# noinspection PyUnresolvedReferences


def generateIndividualCsv(classname, isTest):
    folder_path = '../dataset_images/'
    if isTest:
        folder_path = folder_path + 'TEST/' + classname
    else:
        folder_path = folder_path + 'TRAIN/' + classname

    if not os.path.exists('../dataset_csv'):
        os.mkdir('../dataset_csv')
        os.mkdir('../dataset_csv/TRAIN')
        os.mkdir('../dataset_csv/TEST')

    path_check(classname, isTest)

    filenames = [img for img in glob.glob(folder_path + '/*.jpeg')]
    filenames.sort()
    count = 0
    for img in filenames:
        df = pd.DataFrame({'Value': []})
        print(classname, ':', 'Generating csv for', 'image number',
              str(count), 'out of', str(len(filenames)))
        image = tf.io.read_file(img)
        image = tf.image.decode_jpeg(image)
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, 256, 256)

        model_path = "../lite-model_movenet_singlepose_thunder_3.tflite"
        # model_path = "../lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite"
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()

        input_image = tf.cast(input_image, dtype=tf.float32)

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

        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, width, height)
        input_image = tf.cast(input_image, dtype=tf.uint8)

        image_np = np.squeeze(input_image.numpy(), axis=0)
        image_np = cv2.resize(image_np, (width, height))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # print(type(keypoints[0][0]))
        try:
            [centerBody, distanceList] = processWithCenterOfBody(
                keypoints[0][0], width, height)
        except:
            print('Skipping image due to low confidence score')
            continue

        cv2.circle(image_np, (int(centerBody[0]), int(
            centerBody[1])), 10, (0, 255, 255), -1)

        for distance in distanceList:
            x_row = pd.Series(distance[0], index=df.columns)
            y_row = pd.Series(distance[1], index=df.columns)
            df = df.append(x_row, ignore_index=True)
            df = df.append(y_row, ignore_index=True)

        drawEdges(keypoints, image_np)
        for keypoint in keypoints[0][0]:
            cv2.circle(image_np, (int(keypoint[1]), int(
                keypoint[0])), 4, (0, 0, 255), -1)

        # comment next two lines if you don't want to see output of detected images directly use the output
        cv2.imshow("pose estimation", image_np)
        cv2.waitKey()

        df['Keypoints'] = pd.Series(keypoint_string)
        if not isTest:
            df.to_csv(folder_path.replace('dataset_images',
                                          'dataset_csv') + '/' + classname + '_' + str(count) + '.csv')
        else:
            df.to_csv(folder_path.replace('dataset_images',
                                          'dataset_csv') + '/' + classname + '_' + str(count) + '.csv')
        count = count + 1


def path_check(classname, is_test):
    if not is_test:
        csv_path = '../dataset_csv/TRAIN/' + classname
    else:
        csv_path = '../dataset_csv/TEST/' + classname
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)


# noinspection PyUnresolvedReferences
def drawEdges(keypoints, image_np):
    KEYPOINT_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7),
                      (7, 9), (6, 8), (8, 10), (5, 6), (5,
                                                        11), (6, 12), (11, 12), (11, 13),
                      (13, 15), (12, 14), (14, 16)]
    for edge in KEYPOINT_EDGES:
        x1 = int(keypoints[0][0][edge[0]][1])
        y1 = int(keypoints[0][0][edge[0]][0])
        x2 = int(keypoints[0][0][edge[1]][1])
        y2 = int(keypoints[0][0][edge[1]][0])
        cv2.line(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)


# Generate final test and training set using this method


def generateTrainingSet():
    # new column pose added for final csv
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
        'right_ankleX', 'right_ankleY', 'pose']

    # check if the training Data is Created or not
    if not (os.path.exists('../training_set.csv') and os.path.exists('../testing_set.csv')):

        training_rows = []
        testing_rows = []

        for pose_instance in poses:
            print('Loading class', pose_instance)
            testpath = '../dataset_csv/TEST/' + pose_instance
            trainpath = testpath.replace('/TEST/', '/TRAIN/')
            paths = [testpath, trainpath]
            print(paths)
            for resourcepath in paths:
                filenames = [csvfile for csvfile in glob.glob(
                    resourcepath + '/*.csv')]

                for filename in filenames:
                    filerow = []
                    with open(filename, 'r') as csvfile:
                        # creating a csv reader object
                        csvreader = csv.reader(csvfile)

                        # extracting field names through first row
                        fields = next(csvreader)

                        # extracting each data row one by one
                        for row in csvreader:
                            # removing name of keypoint and index colounn
                            row.pop(2)
                            row.pop(0)
                            # appending value of x/y only instead of [54.242] it is now 54.242
                            filerow.append(row[0])
                        # labeling for supervised dataset
                        filerow.append(poses.index(pose_instance))
                        if '/TRAIN/' in resourcepath:
                            training_rows.append(filerow)
                        else:
                            testing_rows.append(filerow)

                DF1 = pd.DataFrame(training_rows, columns=keypoint_string)
                DF2 = pd.DataFrame(testing_rows, columns=keypoint_string)
                # save the dataframe as a csv file
                DF1.to_csv("../training_set.csv")
                DF2.to_csv("../testing_set.csv")

    else:
        DF1 = pd.read_csv('../training_set.csv')
        DF2 = pd.read_csv('../testing_set.csv')
        DF1.info()
        DF2.info()
        print('Training set is already generated')


def modelinit():
    model = yogtark()
    model.train(500)
    model.test()
    model.save()


if generateCSV:
    for pose in poses:
        generateIndividualCsv(pose, True)
        generateIndividualCsv(pose, False)

generateTrainingSet()
modelinit()
