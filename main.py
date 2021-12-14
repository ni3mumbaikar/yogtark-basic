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

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

getCsv = True
classes = []

# TODO: Change csv layouts to individual X and Y parameter of everykeypoint in both individual and final_training_set


def generateIndividualCsv():

    classes.append(csvgen('downdog', "./dataset_images/downdog"))
    classes.append(csvgen('tree', "./dataset_images/tree"))
    classes.append(csvgen('warrior2', "./dataset_images/warrior2"))

    if getCsv:
        for classitem in classes:
            classitem.generate_csv()


def generateTrainingSet():
    # check if the training Data is Created or not
    if not os.path.exists('./training_set.csv'):

        rows = []

        for classitem in classes:
            print('Loading class', classitem.classname)
            filenames = [csvfile for csvfile in glob.glob(
                classitem.classpath.replace('dataset_images', 'dataset_csv')+'/*.csv')]

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
                    filerow.append(classitem.classname[0])
                    rows.append(filerow)

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

        DF = pd.DataFrame(rows, columns=keypoint_string)

        # save the dataframe as a csv file
        DF.to_csv("./training_set.csv")

    else:
        DF = pd.read_csv('./training_set.csv')
        DF.info()
        print('Training set is already generated')


def modelinit():
    yg = yogtark()
    yg.model.print()


generateIndividualCsv()
generateTrainingSet()


# modelinit()

# model = yogtark()
# optensor = model.getresults(t1)
# tf.print(optensor)
# print(optensor)
