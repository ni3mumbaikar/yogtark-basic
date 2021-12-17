import glob
import os

filenames = []

# script to reaname all the files present in the given directory with new extension
filenames.extend([img for img in glob.glob(
    "./dataset_images/TEST/downdog/*.png")])
filenames.extend([img for img in glob.glob(
    "./dataset_images/TEST/goddess/*.png")])
filenames.extend([img for img in glob.glob(
    "./dataset_images/TEST/plank/*.png")])
filenames.extend([img for img in glob.glob(
    "./dataset_images/TEST/tree/*.png")])
filenames.extend([img for img in glob.glob(
    "./dataset_images/TEST/warrior2/*.png")])
filenames.extend([img for img in glob.glob(
    "./dataset_images/TRAIN/downdog/*.png")])
filenames.extend([img for img in glob.glob(
    "./dataset_images/TRAIN/goddess/*.png")])
filenames.extend([img for img in glob.glob(
    "./dataset_images/TRAIN/plank/*.png")])
filenames.extend([img for img in glob.glob(
    "./dataset_images/TRAIN/tree/*.png")])
filenames.extend([img for img in glob.glob(
    "./dataset_images/TRAIN/warrior2/*.png")])


# print(filenames)

for file in filenames:
    pre, ext = os.path.splitext(file)
    os.rename(file, pre + '.jpeg')
