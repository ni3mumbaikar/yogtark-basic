import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import pandas as pd
from class_csv import csv_generator as csvgen
from class_yogatark import yogtark
import csv
import matplotlib
import time

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

getCsv = False
classes = []


def generateIndividualCsv():

    classes.append(
        csvgen('downdog', "./dataset_images/TRAIN/downdog", True, 0))
    classes.append(csvgen('tree', "./dataset_images/TRAIN/tree", True, 1))
    classes.append(csvgen('plank', "./dataset_images/TRAIN/plank", True, 2))
    classes.append(
        csvgen('goddess', "./dataset_images/TRAIN/goddess", True, 3))
    classes.append(
        csvgen('warrior2', "./dataset_images/TRAIN/warrior2", True, 4))

    classes.append(
        csvgen('downdog', "./dataset_images/TEST/downdog", False, 0))
    classes.append(csvgen('tree', "./dataset_images/TEST/tree", False, 1))
    classes.append(csvgen('plank', "./dataset_images/TEST/plank", False, 2))
    classes.append(
        csvgen('goddess', "./dataset_images/TEST/goddess", False, 3))
    classes.append(
        csvgen('warrior2', "./dataset_images/TEST/warrior2", False, 4))

    if getCsv:
        for classitem in classes:
            classitem.generate_csv()


# def generateTestingSet():


def generateTrainingSet():

    # new coloumn pose added for final csv
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
    if not (os.path.exists('./training_set.csv') and os.path.exists('./testing_set.csv')):

        trainrows = []
        testingrows = []

        for classitem in classes:
            print('Loading class', classitem.classname)
            if classitem.isTrainset:
                filenames = [csvfile for csvfile in glob.glob(
                    classitem.classpath.replace('dataset_images', 'dataset_csv')+'/*.csv')]
            else:
                filenames = [csvfile for csvfile in glob.glob(
                    classitem.classpath.replace('dataset_images', 'dataset_csv')+'/*.csv')]
            # print(filenames)

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
                    # appending a number for each class
                    filerow.append(classitem.classvalue)
                    if classitem.isTrainset:
                        trainrows.append(filerow)
                    else:
                        testingrows.append(filerow)

            DF1 = pd.DataFrame(trainrows, columns=keypoint_string)
            DF2 = pd.DataFrame(testingrows, columns=keypoint_string)
            # save the dataframe as a csv file
            DF1.to_csv("./training_set.csv")
            DF2.to_csv("./testing_set.csv")

    else:
        DF1 = pd.read_csv('./training_set.csv')
        DF2 = pd.read_csv('./testing_set.csv')
        DF1.info()
        DF2.info()
        print('Training set is already generated')


def modelinit():
    yg = yogtark()
    yg.train()


start = time.time()
generateIndividualCsv()
end = time.time()
print('Time taken : is', str(end-start) + 's')
generateTrainingSet()
# modelinit()
