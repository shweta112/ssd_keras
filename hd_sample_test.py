#!/usr/bin/env python

from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
import cv2
import numpy as np
import glob
import os
from os import listdir
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetections2 import DecodeDetections2
from keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd_box_utils.ssd_box_encode_decode_utils import decode_y, decode_y2
from data_generator.ssd_batch_generator import BatchGenerator

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

        plt.savefig('sample_test/pred_27-03_e12/' + '%03d'%int(filenames[i].split('.')[0]), format='png', dpi=fig.dpi)
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

        cv2.imwrite('sample_test/pred_27-03_e12/' + '%03d'%int(filenames[i].split('.')[0]), output_images[i])


# Set the image size.
img_height = 300
img_width = 480

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
weights_path = 'weights_COCO4_27-03/ssd_weights_epoch-12_val_acc-0.9457_val_loss-6.5008.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

img_paths = 'sample_test/16bit/*.png'
img8_paths = 'sample_test/8bit/*.png'
filenames = listdir('sample_test/16bit/')


output_images = np.array([cv2.imread(img_file) for img_file in glob.glob(img8_paths)])
orig_images = np.array([cv2.imread(img_file, -1 | 0) for img_file in glob.glob(img_paths)])
input_images = np.array([cv2.resize(img, (img_width, img_height),
                                 interpolation=cv2.INTER_AREA) for img in orig_images])
input_images = np.stack([input_images] * 3, axis=-1)

y_pred = model.predict(input_images)

confidence_threshold = 0.5

y_pred = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

# np.set_printoptions(precision=2, suppress=True, linewidth=90)
# print("Predicted boxes:\n")
# print('    class    conf  xmin    ymin    xmax    ymax')
# print(y_pred[0])

# Display the image and draw the predicted boxes onto it.
classes = ['background', 'Person']

display_plt(y_pred, output_images)



