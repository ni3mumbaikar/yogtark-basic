import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import pandas as pd
from class_csv import csv_generator as csvgen
from class_yogatark import yogtark
import csv

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

getCsv = False


classes = []
classes.append(csvgen('downdog', "./dataset_images/downdog"))
classes.append(csvgen('tree', "./dataset_images/tree"))
classes.append(csvgen('warrior2', "./dataset_images/warrior2"))

if getCsv:
    for classitem in classes:
        classitem.generate_csv()

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
            rows.append(filerow)

    # print(len(rows))


x = np.array(rows, dtype=float)
print(x)
y = ['downdog', 'downdog', 'tree', 'tree' 'warrior', 'warrior']
y = np.array(y, dtype=str)
DF = pd.DataFrame(x)

# save the dataframe as a csv file
DF.to_csv("./dataX.csv")

# model = yogtark()
# optensor = model.getresults(t1)
# tf.print(optensor)
# print(optensor)
