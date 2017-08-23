import os
import re
import glob
import fnmatch
import numpy as np

# Path to labels directory on the VM
dirpath = "/datasets/aic540/train/labels"
output_dir = "/home/aicg2/vgg_aic/aic540_vgg_train_labels"
nb_files = len(fnmatch.filter(os.listdir(dirpath), '*.txt'))
nb_targets = 15
other_objects = "Others"

target = {"Car" : 0, "SUV" : 1, "SmallTruck" : 2,
"MediumTruck" : 3, "LargeTruck" : 4, "Pedestrian" : 5, "Bus" : 6, "Van" : 7,
"GroupOfPeople" : 8, "Bicycle" : 9, "Motorcycle" : 10, "TrafficSignal-Green" : 11,
"TrafficSignal-Yellow" : 12, "TrafficSignal-Red" : 13, other_objects : 14}
if os.path.isdir(output_dir) is False:
    os.mkdir(output_dir)

result = np.zeros((nb_files, nb_targets))
i = 0
npy_file_regex = re.compile("(.*?)(\.txt)")
for filepath in glob.glob(os.path.join(dirpath, '*.txt')):
    filename = filepath.split("/")[-1]
    output_file =  output_dir +"/"+ npy_file_regex.match(filename).groups()[0] + ".npy"
    f= open(filepath,"r")
    for line in f:
        _object = line.split(" ")[0]
        if _object in target:
            _index = target[_object]
        else:
            _index = target[other_objects]
        result[i][_index] = result[i][_index] + 1

    np.save(output_file,result[i])
    i = i+1
    if i == nb_files:
        break

print 'generated number of files', nb_files, ' at', output_dir
