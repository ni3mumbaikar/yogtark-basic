import glob
import os


# script to reaname all the files present in the given directory with new extension
filenames = [img for img in glob.glob("./dataset_images/warrior2/*.jpg")]

for file in filenames:
    pre, ext = os.path.splitext(file)
    os.rename(file, pre + '.jpeg')
