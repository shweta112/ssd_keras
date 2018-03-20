import numpy as np
import h5py

weights_filename='../trained_weights/SSD/VGG_coco_SSD_300x300.h5'
weights_file = h5py.File(weights_filename, 'r')

print(weights_file['assign_49']['assign_49']['kernel:0'].shape)
