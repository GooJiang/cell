import numpy as np
import scipy.misc as misc
import os
import glob
import matplotlib.pyplot as plt


ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'aug')

def load_data(dataset, type, reshape_size=None, det=True, cls=True, preprocss_num=128.):
    """
    Load dataset from files
    :param type: either train, test or validation.
    :param reshape_size: reshape to (512, 512) if cropping images are using.
    :param det: True if detection masks needed.
    :param cls: True if classification masks needed.
    :param preprocss_num: number to subtract and divide in normalization step.
    """
    def _image_normalization(image):
        """
        preprocessing on image.
        """
        image = image - preprocss_num
        image = image / preprocss_num
        return image

    def _torch_image_transpose(images):
        """
        change image to channel first.
        """
        images = np.array(images)
        images = np.transpose(images, (0, 3, 1, 2))
        return images

    path = os.path.join(dataset, type)
    imgs, det_masks, cls_masks = [], [], []
    for i, file in enumerate(os.listdir(path)):
        for j, img_file in enumerate(os.listdir(os.path.join(path, file))):
            if 'original.bmp' in img_file:
                img_path = os.path.join(path, file, img_file)
                img = misc.imread(img_path)
                if reshape_size is not None:
                    img = misc.imresize(img, reshape_size, interp='nearest')
                img = _image_normalization(img)
                imgs.append(img)

            elif 'detection.bmp' in img_file and det == True:
                det_mask_path = os.path.join(path, file, img_file)
                det_mask = misc.imread(det_mask_path, mode='L')
                if reshape_size is not None:
                    det_mask = misc.imresize(det_mask, reshape_size, interp='nearest')
                det_masks.append(det_mask)

            elif 'classification.bmp' in img_file and cls == True:
                if cls == True:
                    cls_mask_path = os.path.join(path, file, img_file)
                    cls_mask = misc.imread(cls_mask_path, mode='L')
                    if reshape_size != None:
                        cls_mask = misc.imresize(cls_mask, reshape_size, interp='nearest')
                    cls_masks.append(cls_mask)

            imgs = _torch_image_transpose(imgs)
            det_masks = _torch_image_transpose(det_masks)
            cls_masks = _torch_image_transpose(cls_masks)
    return imgs, det_masks, cls_masks


class CRC_joint:
    def __init__(self,imgdir,target_size=256):
        self.imgdir = imgdir
        self.target_size = target_size

    def load_train(self, preprocess=True):
        x_train = []
        y_train_det = []
        y_train_cls = []
        for i in range(1,71):
            for k in range(1,7):
                img_file_name = 'img' + str(i) + '_' + str(k) + '.bmp'
                detection_file_name = 'img' + str(i) + '_' + str(k) + '_detectionL.bmp'
                classification_file_name = 'img' + str(i) + '_' + str(k) + '_ClassificationL.bmp'
                img_path = os.path.join(self.imgdir, img_file_name)
                detection_path = os.path.join(self.imgdir, detection_file_name)
                classification_path = os.path.join(self.imgdir, classification_file_name)
                if os.path.exists(img_path):
                    img = misc.imread(img_path)
                    img = misc.imresize(img,(self.target_size,self.target_size),interp='nearest')
                    if preprocess:
                        # img = img/255.
                        #img -= np.mean(img, keepdims=True)
                        #img /= (np.std(img, keepdims=True) + 1e-7)
                        img = img - 128.
                        img = img / 128.
                    x_train.append(img)

                    detection = misc.imread(detection_path, mode='L')
                    detection = misc.imresize(detection, (self.target_size, self.target_size), interp='nearest')

                    classification = misc.imread(classification_path, mode='L')
                    classification = misc.imresize(classification, (self.target_size, self.target_size), interp='nearest')

                    detection = detection.reshape(detection.shape[0], detection.shape[1], 1)
                    classification = classification.reshape(classification.shape[0], classification.shape[1], 1)

                    y_train_det.append(detection)
                    y_train_cls.append(classification)

            print('finsihed:img',i)
        x_train = np.array(x_train)
        x_train = np.transpose(x_train, (0, 3, 1, 2))
        y_train_det = np.array(y_train_det)
        y_train_det = np.transpose(y_train_det, (0, 3, 1, 2))
        y_train_cls = np.array(y_train_cls)
        y_train_cls = np.transpose(y_train_cls, (0, 3, 1, 2))

        return x_train, y_train_det, y_train_cls

    def load_val(self, preprocess=True):
        x_train = []
        y_train_det = []
        y_train_cls = []
        for i in range(71, 81):
            for k in range(1, 7):
                img_file_name = 'img' + str(i) + '_' + str(k) + '.bmp'
                detection_file_name = 'img' + str(i) + '_' + str(k) + '_detectionL.bmp'
                classification_file_name = 'img' + str(i) + '_' + str(k) + '_ClassificationL.bmp'
                img_path = os.path.join(self.imgdir, img_file_name)
                detection_path = os.path.join(self.imgdir, detection_file_name)
                classification_path = os.path.join(self.imgdir, classification_file_name)
                if os.path.exists(img_path):
                    img = misc.imread(img_path)
                    img = misc.imresize(img, (self.target_size, self.target_size), interp='nearest')
                    if preprocess:
                        # img = img/255.
                        # img -= np.mean(img, keepdims=True)
                        # img /= (np.std(img, keepdims=True) + 1e-7)
                        img = img - 128.
                        img = img / 128.
                    x_train.append(img)

                    detection = misc.imread(detection_path, mode='L')
                    detection = misc.imresize(detection, (self.target_size, self.target_size), interp='nearest')

                    classification = misc.imread(classification_path, mode='L')
                    classification = misc.imresize(classification, (self.target_size, self.target_size),
                                                   interp='nearest')

                    detection = detection.reshape(detection.shape[0], detection.shape[1], 1)
                    classification = classification.reshape(classification.shape[0], classification.shape[1], 1)

                    y_train_det.append(detection)
                    y_train_cls.append(classification)

            print('finsihed:img', i)
        x_train = np.array(x_train)
        x_train = np.transpose(x_train, (0, 3, 1, 2))
        y_train_det = np.array(y_train_det)
        y_train_det = np.transpose(y_train_det, (0, 3, 1, 2))
        y_train_cls = np.array(y_train_cls)
        y_train_cls = np.transpose(y_train_cls, (0, 3, 1, 2))

        return x_train, y_train_det, y_train_cls


