import subprocess
import ast
import cv2
import glob
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetections2 import DecodeDetections2
from keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd_box_utils.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from keras_loss_function.keras_ssd_loss import SSDLoss

# Set the image size.
img_height = 300
img_width = 480

def load_test_data(path, folder_name='Sempach-7'):
    box_output_format = ['class_id', 'xmin', 'ymin', 'xmax', 'ymax']
    img_path = folder_name + '/16bit/'

    files = glob.glob(img_path)
    orig_images = np.array([cv2.imread(img_file, -1 | 0) for img_file in files]) # Store the images here.
    input_images = np.array([cv2.resize(img, (img_width, img_height),
                                 interpolation=cv2.INTER_AREA) for img in orig_images])  # Store resized versions of the images here.
    labels = []

    pipe = subprocess.Popen(["perl", "vbb.pl", folder_name], stdout=subprocess.PIPE)

    result = ast.literal_eval(pipe.stdout.read().decode("utf-8"))
    no_of_people = len(result)

    for p in range(no_of_people):
        data = result[p]
        start = data['firstFrame']
        end = data['lastFrame']
        frame = start
        boxes = []
        for pos in data['pos']:
            pos = [max(0, round(x)-1) for x in pos]
            print(pos)
            # Since the dataset only contains one class, the class ID is always 1 (i.e. 'Person')
            class_id = 1
            xmin = pos[1]
            ymin = pos[0]
            xmax = pos[1] + pos[3]
            ymax = pos[0] + pos[2]
            item_dict = {'image_id': path + frame + '.png',
                         'class_id': class_id,
                         'xmin': xmin,
                         'ymin': ymin,
                         'xmax': xmax,
                         'ymax': ymax}
            box = []
            for item in box_output_format:
                box.append(item_dict[item])
            boxes.append(box)
            frame += 1
        labels.append(boxes)

    return input_images, labels


model_path = 'ssd_weights_epoch-15_val_acc-0.9322_val_loss-7.5056.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})