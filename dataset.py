import numpy as np
import scipy.misc as misc
import os
import glob
import matplotlib.pyplot as plt
import torch
import scipy.io as sio

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'aug')


def validation_metric(dataset, model, reshape_size = (256, 256), preprocss_num=128.):
    path= os.path.join(dataset, 'validation')
    tp_num = [0, 0]
    gt_num = [0, 0]
    pred_num = [0, 0]
    precision = [0, 0]
    recall = [0, 0]
    f1_score = [0, 0]
    cell_type_group = ['epithelial', 'fibroblast', 'inflammatory', 'others']
    gt_type = ['Detection', 'Classification']
    for i, file in enumerate(os.listdir(path)):
        for j, img_file in enumerate(os.listdir(os.path.join(path, file))):
            curr_file = os.path.join(path, file, img_file)
            if 'original.bmp' in img_file:
                img = misc.imread(curr_file)
                if reshape_size is not None:
                    img = misc.imresize(img, reshape_size, interp='nearest')
                img = _image_normalization(img, preprocss_num)
                img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
                img = _torch_image_transpose(img)
                img = torch.Tensor(img).cuda()


                img_result = model(img)
            if 'detection_mat' in img_file:
                det_mat = sio.loadmat(curr_file)['detection']
            if 'epithelial.mat' in img_file:
                epi_mat = sio.loadmat(curr_file)['detection']
            if 'inflammatory.mat' in img_file:
                inf_mat = sio.loadmat(curr_file)['detection']
            if 'others.mat' in img_file:
                other_mat = sio.loadmat(curr_file)['detection']
            if 'fibroblast.mat' in img_file:
                fib_mat = sio.loadmat(curr_file)['detection']

        result = img_result.cpu().detach().numpy()
        result = np.transpose(result, (0, 2, 3, 1))[0]
        result = np.exp(result)



def _image_normalization(image, preprocss_num):
    """
    preprocessing on image.
    """
    image = image - preprocss_num
    image = image / preprocss_num
    return image


def _torch_image_transpose(images, type='image'):
    """
    change image to channel first.
    """
    images = np.array(images)
    images = np.transpose(images, (0, 3, 1, 2))

    return images


def load_data(dataset, type, reshape_size=None, det=True, cls=True, preprocss_num=128.):
    """
    Load dataset from files
    :param type: either train, test or validation.
    :param reshape_size: reshape to (512, 512) if cropping images are using.
    :param det: True if detection masks needed.
    :param cls: True if classification masks needed.
    :param preprocss_num: number to subtract and divide in normalization step.
    """
    path = os.path.join(dataset, type)
    imgs, det_masks, cls_masks = [], [], []
    for i, file in enumerate(os.listdir(path)):
        for j, img_file in enumerate(os.listdir(os.path.join(path, file))):
            if 'original.bmp' in img_file:
                img_path = os.path.join(path, file, img_file)
                img = misc.imread(img_path)
                if reshape_size is not None:
                    img = misc.imresize(img, reshape_size, interp='nearest')
                img = _image_normalization(img, preprocss_num)
                imgs.append(img)
            if 'detection.bmp' in img_file and det is True and 'verifiy' not in img_file:
                det_mask_path = os.path.join(path, file, img_file)
                det_mask = misc.imread(det_mask_path, mode='L')
                if reshape_size is not None:
                    det_mask = misc.imresize(det_mask, reshape_size, interp='nearest')
                det_mask = det_mask.reshape(det_mask.shape[0], det_mask.shape[1], 1)
                det_masks.append(det_mask)

            if 'classification.bmp' in img_file and cls is True and 'verifiy' not in img_file:
                cls_mask_path = os.path.join(path, file, img_file)
                cls_mask = misc.imread(cls_mask_path, mode='L')
                if reshape_size != None:
                    cls_mask = misc.imresize(cls_mask, reshape_size, interp='nearest')
                cls_mask = cls_mask.reshape(cls_mask.shape[0], cls_mask.shape[1], 1)
                cls_masks.append(cls_mask)

    print(len(imgs), len(det_masks), len(cls_masks))
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


