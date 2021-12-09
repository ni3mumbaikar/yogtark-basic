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

getCsv = False
classes = []


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
                        row.pop(4)
                        row.pop(3)
                        row.pop(0)
                        filerow.append(row)
                    filerow.append(classitem.classname[0])
                    rows.append(filerow)

            # print(len(rows))

        x = np.array(rows)
        # print(x)
        keypoint_string = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                           'left_wrist',  'right_wrist',  'left_hip', 'right_hip', 'left_knee', 'right_knee',  'left_ankle', 'right_ankle', 'pose']

        DF = pd.DataFrame(x, columns=keypoint_string)

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
