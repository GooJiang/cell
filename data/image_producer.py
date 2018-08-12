from torch.utils.data import Dataset
import os
import scipy.misc as misc
import numpy as np
from random import shuffle

class CellImageDataset(Dataset):

    def __init__(self, data_path, image_size=None,
                 det=True, cls=True):
        self.image_size = image_size
        self.data_path = data_path
        self.cls = cls
        self.det = det
        self._preprocess()

    def _preprocess(self):
        self.files = []
        for i, file in enumerate(os.listdir(self.data_path)):
            self.files.append(file)
        shuffle(self.files)

    def __getitem__(self, index):
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
            if type == 'image':
                images = np.transpose(images, (2, 1, 0))
            return images

        file = self.files[index]
        img = np.zeros((1, 3, 500, 500))
        det_mask = np.zeros((1, 2, 500, 500))
        cls_mask = np.zeros((1, 2, 500, 500))
        imgs, det_masks, cls_masks = [], [], []
        print(file)
        for j, img_file in enumerate(os.listdir(os.path.join(self.data_path, file))):
            if 'original.bmp' in img_file:
                img_path = os.path.join(self.data_path, file, img_file)
                img = misc.imread(img_path)
                if self.image_size is not None and self.image_size != img.shape[1]:
                    img = misc.imresize(img, self.image_size, interp='nearest')
                img = _image_normalization(img, 128.0)
                imgs.append(img)
            if 'detection.bmp' in img_file and self.det is True and 'verifiy' not in img_file:
                det_mask_path = os.path.join(self.data_path, file, img_file)
                det_mask = misc.imread(det_mask_path, mode='L')
                if self.image_size is not None and self.image_size != img.shape[1]:
                    det_mask = misc.imresize(det_mask, self.image_size, interp='nearest')
                #det_mask = det_mask.reshape(det_mask.shape[0], det_mask.shape[1], 1)
                det_masks.append(det_mask)
            if 'classification.bmp' in img_file and self.cls is True and 'verifiy' not in img_file:
                cls_mask_path = os.path.join(self.data_path, file, img_file)
                cls_mask = misc.imread(cls_mask_path, mode='L')
                if self.image_size != None and self.image_size != img.shape[1]:
                    cls_mask = misc.imresize(cls_mask, self.image_size, interp='nearest')
                #cls_mask = cls_mask.reshape(cls_mask.shape[0], cls_mask.shape[1], 1)
                cls_masks.append(cls_mask)
        #print(len(imgs), len(det_masks), len(cls_masks))
        #print('before _torch_image: ', img.shape, det_mask.shape, cls_mask.shape)
        img = _torch_image_transpose(img)
        #det_mask = _torch_image_transpose(det_mask)
        #cls_mask = _torch_image_transpose(cls_mask)
        #print('after _torch_image: ', img.shape, det_mask.shape, cls_mask.shape)
        return (img, det_mask, cls_mask)

    def __len__(self):
        return len(self.files)




        """
        counter = 0
        batch_features = np.zeros((self.batch_per_step, 3, self.image_size, self.image_size))
        batch_det_labels = np.zeros((self.batch_per_step, 1, self.image_size, self.image_size))
        batch_cls_labels = np.zeros((self.batch_per_step, 1, self.image_size, self.image_size))
        for i in range(self.batch_per_step):
            index = np.random.choice(self.features.shape[0], 1)
            feature, det_label, cls_label = self.features[index], self.det_labels[index], self.cls_labels[index]
            batch_features[counter] = feature[index]
            batch_det_labels[counter] = det_label[index]
            batch_cls_labels[counter] = cls_label[index]
            counter += 1
        print('batch_features.shape: {}, batch_det_labels: {}, batch_cls_labels: {}'.format(batch_features.shape,
                                                                                            batch_det_labels.shape,
                                                                                            batch_cls_labels.shape))
        return batch_features, batch_det_labels, batch_cls_labels
    """