'''
Author: Shweta Mahajan
'''

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras import backend as K

from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_box_utils.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from data_generator.ssd_batch_generator import BatchGenerator


# Define a learning rate schedule.
def lr_schedule(epoch):
    if epoch <= 100: return 0.0001
    elif epoch <= 160: return 0.00001
    else: return 0.000001


img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
img_channels = 3 # Number of color channels of the input images
subtract_mean = None # [123, 117, 104] # The per-channel mean of the images in the dataset
swap_channels = False # True # The color channel order in the original SSD is BGR
n_classes = 1 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_coco
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids' # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
normalize_coords = True

## MODEL
# 1: Build the Keras model
K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                limit_boxes=limit_boxes,
                variances=variances,
                coords=coords,
                normalize_coords=normalize_coords,
                subtract_mean=subtract_mean,
                divide_by_stddev=None,
                swap_channels=swap_channels)

# 2: Load the trained VGG-16 weights into the model.
weights_path = 'trained_weights/SSD/VGG_coco_SSD_300x300_iter_400000_subsampled_hd.h5'

model.load_weights(weights_path, by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss, metrics=['accuracy'])


## Data generators
# 1: Instantiate to `BatchGenerator` objects: One for training, one for validation.
train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])
val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.
# The directories that contain the images.
HD_val_images_dir = '../../Human_Detection/images/2017-08-16/16bit'
HD_08_29_images_dir = '../../Human_Detection/images/2017-08-29/16bit'
HD_08_30_images_dir = '../../Human_Detection/images/2017-08-30/16bit'

# The filenames that contain the annotations.
HD_val_annotation_filename = '../rnd-human-detection/annotations/hd_20170816.json'
HD_08_29_annotation_filename = '../rnd-human-detection/annotations/hd_20170829.json'
HD_08_30_annotation_filename = '../rnd-human-detection/annotations/hd_20170830.json'

train_dataset.parse_custom(images_dirs=[HD_08_29_images_dir,
                                        HD_08_30_images_dir],
                           annotations_filenames=[HD_08_29_annotation_filename,
                                                  HD_08_30_annotation_filename],
                           ret=False)

val_dataset.parse_custom(images_dirs=[HD_val_images_dir],
                         annotations_filenames=[HD_val_annotation_filename],
                         ret=False)

# 3: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=None,
                                max_scale=None,
                                scales=scales,
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                steps=steps,
                                offsets=offsets,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)

# 4: Set the batch size.
batch_size = 16 # Change the batch size if you like, or if you run into memory issues with your GPU.

# 5: Set the image processing / data augmentation options and create generator handles.
train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         convert_to_3_channels=True,
                                         equalize=True,
                                         brightness=(0.1, 5.0, 0.5),
                                         flip=0.5,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=(img_height, img_width, 1, 3), # This one is important because the Pascal VOC images vary in size
                                         random_pad_and_resize=(img_height, img_width, 1, 3, 0.5), # This one is important because the Pascal VOC images vary in size
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True, # While the anchor boxes are not being clipped, the ground truth boxes should be
                                         include_thresh=0.4,
                                         keep_images_without_gt=True)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=True,
                                     train=True,
                                     ssd_box_encoder=ssd_box_encoder,
                                     convert_to_3_channels=True,
                                     equalize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     max_crop_and_resize=(img_height, img_width, 1, 3), # This one is important because the Pascal VOC images vary in size
                                     random_pad_and_resize=(img_height, img_width, 1, 3, 0.5), # This one is important because the Pascal VOC images vary in size
                                     random_crop=False,
                                     crop=False,
                                     resize=False,
                                     gray=False,
                                     limit_boxes=True,
                                     include_thresh=0.4,
                                     keep_images_without_gt=True)

# Get the number of samples in the training and validations datasets to compute the epoch lengths below.
n_train_samples = train_dataset.get_n_samples()
n_val_samples   = val_dataset.get_n_samples()

# print(n_train_samples)
# print(n_val_samples)

epochs = 20

history = model.fit_generator(generator = train_generator,
                              steps_per_epoch = ceil(n_train_samples/batch_size),
                              epochs = epochs,
                              verbose=1,
                              callbacks = [ModelCheckpoint('weights_COCO4_27-03/ssd_weights_epoch-{epoch:02d}_val_acc-{val_acc:.4f}_val_loss-{val_loss:.4f}.h5',
                                                           monitor='val_acc',
                                                           verbose=1,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='auto',
                                                           period=1),
                                           EarlyStopping(monitor='val_acc',
                                                         min_delta=0.001,
                                                         patience=201),
                                           LearningRateScheduler(lr_schedule),
                                           CSVLogger('training.log', append=True),
                                           ReduceLROnPlateau(monitor='val_acc', factor=0.8,
                                                             patience=200, min_lr=0, verbose=1),
                                           TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32,
                                                       write_graph=True, write_grads=False, write_images=True,
                                                       embeddings_freq=0, embeddings_layer_names=None,
                                                       embeddings_metadata=None)],
                              validation_data = val_generator,
                              validation_steps = ceil(n_val_samples/batch_size))


#       Do the same in the `ModelCheckpoint` callback above.
# model_name = 'ssd300_hd_v1'
# model.save('{}.h5'.format(model_name))
# model.save_weights('{}_weights.h5'.format(model_name))
#
# print()
# print("Model saved under {}.h5".format(model_name))
# print("Weights also saved separately under {}_weights.h5".format(model_name))
# print()

plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.legend(loc='upper right', prop={'size': 24})
plt.show(block=True)