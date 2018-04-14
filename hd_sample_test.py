#!/usr/bin/env python
'''
Author: Shweta Mahajan
'''

from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
import cv2
import numpy as np
import glob
import os
from os import listdir
import sys
import time
from matplotlib import pyplot as plt

# TODO: Specify the directory that contains the `pycocotools` here.
pycocotools_dir = '../../cocoapi-master/PythonAPI/'
if pycocotools_dir not in sys.path:
    sys.path.insert(0, pycocotools_dir)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetections2 import DecodeDetections2
from keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd_box_utils.ssd_box_encode_decode_utils import decode_y, decode_y2
from data_generator.ssd_batch_generator import BatchGenerator

from eval_utils.custom_utils import predict_all_to_json

def display_plt(y_pred, output_images):
    for i in range(len(y_pred)):
        # Set the colors for the bounding boxes
        color = 'g'
        fig = plt.figure(figsize=(20,12))
        plt.imshow(output_images[i])

        current_axis = plt.gca()

        for box in y_pred[i]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[-4] * output_images[0].shape[1] / img_width
            ymin = box[-3] * output_images[0].shape[0] / img_height
            xmax = box[-2] * output_images[0].shape[1] / img_width
            ymax = box[-1] * output_images[0].shape[0] / img_height

            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

        #plt.savefig(base_path + folder + '%03d'%int(filenames[i].split('.')[0]), format='png', dpi=fig.dpi)
        plt.savefig(base_path + folder + filenames[i].split('.')[0], format='png', dpi=fig.dpi)
        plt.close(fig)


def display_cv(y_pred, output_images):
    for i in range(len(y_pred)):
        for box in y_pred[i]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = int(box[-4] * output_images[i].shape[1] / img_width)
            ymin = int(box[-3] * output_images[i].shape[0] / img_height)
            xmax = int(box[-2] * output_images[i].shape[1] / img_width)
            ymax = int(box[-1] * output_images[i].shape[0] / img_height)

            cv2.rectangle(output_images[i], (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])

            cv2.putText(output_images[i], label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imwrite(base_path + folder + '%03d'%int(filenames[i].split('.')[0]) + '.png', output_images[i])

def save_pred():
    test_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])
    test_dataset.parse_custom(images_dirs=[img_paths.split('*')[0]],
                               annotations_filenames=[base_path + 'annotations.json'],
                               ret=False)
    predict_all_to_json('pred_test_full.json', model, img_height, img_width, test_dataset, 32, 'pad', 'inference')


def infer(confidence_threshold):
    y_pred = model.predict(input_images)

    y_pred = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    return y_pred

def evaluate():
    coco_gt = COCO(base_path + 'annotations_COCO.json')
    coco_dt = coco_gt.loadRes('pred.json')
    image_ids = sorted(coco_gt.getImgIds())

    cocoEval = COCOeval(cocoGt=coco_gt,
                        cocoDt=coco_dt,
                        iouType='bbox')
    cocoEval.params.imgIds = image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


# Set the image size.
img_height = 480
img_width = 640

# Display the image and draw the predicted boxes onto it.
classes = ['background', 'Person']

scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=1,
                mode='inference',
                l2_regularization=0.0005,
                scales=scales_coco, #[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                limit_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                coords='centroids',
                normalize_coords=True,
                subtract_mean=None, #[123, 117, 104],
                swap_channels=False,
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.

# weights_path = 'weights_COCO4_17-03/ssd_weights_epoch-05_val_acc-0.9237_val_loss-5.9265.h5'
# weights_path = 'weights_COCO4_27-03/ssd_weights_epoch-12_val_acc-0.9457_val_loss-6.5008.h5'
weights_path = 'weights_COCO4_13-04_full/ssd_weights_epoch-50_acc-0.9770_loss-0.3436.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# base_path = '../../rosbags_meppen/AutoAuge_RosbagsUAV2/meppen/meppen_2017-10-17-17-45-23_filtered/'
base_path = 'sample_test/'

img_paths = base_path + '16bit/*.png'
img8_paths = base_path + '8bit/*.png'
filenames = listdir(img_paths.split('*')[0])

folder = 'pred/' if len(sys.argv)==1 else sys.argv[1]

output_images = np.array([cv2.imread(img_file) for img_file in glob.glob(img8_paths)])
orig_images = np.array([cv2.imread(img_file, -1 | 0) for img_file in glob.glob(img_paths)])
input_images = np.array([cv2.resize(img, (img_width, img_height),
                                 interpolation=cv2.INTER_AREA) for img in orig_images])
input_images = np.stack([input_images] * 3, axis=-1)

tic = time.time()

y_pred = infer(confidence_threshold=0.7)
# save_pred()
# evaluate()

dur = time.time() - tic
print('DONE (t={:0.4f}s)'.format(dur))
print('Time per frame: {:0.4f}s'.format(dur / len(filenames)))
print('FPS: {:0.4f}s'.format(len(filenames) / dur))

# np.set_printoptions(precision=2, suppress=True, linewidth=90)
# print("Predicted boxes:\n")
# print('    class    conf  xmin    ymin    xmax    ymax')
# print(y_pred[0])

display_plt(y_pred, output_images)


