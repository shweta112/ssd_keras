import numpy as np
#import h5py
import cv2

def test_wts():
    weights_filename='../trained_weights/SSD/VGG_coco_SSD_300x300.h5'
    weights_file = h5py.File(weights_filename, 'r')
    print(weights_file['assign_49']['assign_49']['kernel:0'].shape)

def test_brightness():
    path = '/media/shweta.mahajan/Daten/Human_Detection/images/2017-08-29/8bit/frame00860.png'
    path16 = '/media/shweta.mahajan/Daten/Human_Detection/images/2017-08-29/16bit/frame00860.png'

    img = cv2.imread(path)
    img16 = np.stack([cv2.imread(path16, -1|0)] * 3, axis=-1)

    cv2.imshow('test', img)
    print(img16.dtype)
    ycrb = cv2.cvtColor(img16, cv2.COLOR_RGB2YCrCb)
    print(ycrb.dtype)

    max = 2**16 - 1
    random_br = np.random.uniform(0.1, 5.0)
    mask = ycrb[0, :, :] * random_br > max
    y_channel = np.where(mask, max, ycrb[0, :, :] * random_br)
    ycrb[0, :, :] = y_channel
    img_br = cv2.cvtColor(ycrb, cv2.COLOR_YCrCb2RGB)

    cv2.imshow('test_br', img_br)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_brightness()
