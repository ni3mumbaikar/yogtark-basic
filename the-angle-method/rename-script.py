import glob
import os
"""
This is file rename util script to reaname class files please mention class name and follow directory
structure as specified

class here means asna or the output possibilities

-YogTark
    -dataset_images
        -TEST
            -downdog
            -class2
            -classN
        -TRAIN
            -downdog
            -class2
            -classN
"""
def extend(classname):
    list1 = [img for img in glob.glob(
        "../../dataset_images/TEST/"+classname+"/*.jpeg")]
    # print(list1)
    renamefiles(list1, classname,True)
    list2 = [img for img in glob.glob(
        "../../dataset_images/TRAIN/"+classname+"/*.jpeg")]
    renamefiles(list2, classname, False)

# Change filename with _ or - if file already exists so that all files will override new name
def renamefiles(filenames,classname, isTest):
    i = 0;
    for file in filenames:
        pre, ext = os.path.splitext(file)
        if isTest:
            os.rename(pre+ext,'../../dataset_images/TEST/'+classname+'/'+classname+"-"+str(i)+ '.jpeg')
        else:
            os.rename(pre+ext,'../../dataset_images/TRAIN/'+classname+'/'+classname+"-"+str(i)+ '.jpeg')
        i+=1

poses = ['downdog','goddess','plank','tree','warrior2']

for pose in poses:
    extend(pose)

print('Program complete')
