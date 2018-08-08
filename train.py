from sfcn_model import Attention_Net_Global
from tensorboardX import SummaryWriter
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time, os
import dataset
from config import Config
from dataset import load_data

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'aug')

BATCH_SIZE = Config.image_per_gpu * Config.gpu_count

def data_prepare():
    x_train, y_train_det, y_train_cls = load_data(dataset=DATA_DIR, type='train')
    x_val, y_val_det, y_val_cls = load_data(DATA_DIR, type='validation')

    train_count = len(x_train)
    val_count = len(x_val)
    val_steps = int(val_count / BATCH_SIZE)
    print('training imgs:', train_count)
    print('val imgs:', val_count)

    trainset = np.concatenate([x_train, y_train_det, y_train_cls], axis=1)
    trainset = torch.Tensor(trainset)

    valset = np.concatenate([x_val, y_val_det, y_val_cls], axis=1)
    valset = torch.Tensor(valset)

    trainset = trainset.cuda()
    valset = valset.cuda()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader


def tune_weights(weight_det=None, weight_cls=None):
    if weight_det is None:
        weight_det = torch.Tensor([1, 1])
    else:
        weight_det = torch.Tensor(weight_det)
    if weight_cls is None:
        weight_cls = torch.Tensor([1, 1, 1, 1, 1])
    else:
        weight_cls = torch.Tensor(weight_cls)
    weight_det = weight_det.cuda()
    weight_cls = weight_cls.cuda()
    return weight_det, weight_cls


def train(model, weight_det=None, weight_cls=None,data_dir='',
          preprocess=True, gpu=True, num_epochs=Config.epoch, target_size=256):

    weight_det, weight_cls = tune_weights(weight_det, weight_cls)
    writer = SummaryWriter()

    data = dataset.CRC_joint(data_dir, target_size=target_size)
    x_train, y_train_det, y_train_cls = data.load_train(preprocess=preprocess)
    train_count = len(x_train)

    x_val, y_val_det, y_val_cls = data.load_val(preprocess=preprocess)
    val_count = len(x_val)
    val_steps = int(val_count / BATCH_SIZE)
    print('training imgs:', train_count)
    print('val imgs:', val_count)

    trainset = np.concatenate([x_train, y_train_det, y_train_cls], axis=1)
    trainset = torch.Tensor(trainset)

    valset = np.concatenate([x_val, y_val_det, y_val_cls], axis=1)
    valset = torch.Tensor(valset)

    if gpu:
        model = model.cuda()
        trainset = trainset.cuda()
        valset = valset.cuda()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    NLLLoss_det = nn.NLLLoss(weight=weight_det)
    NLLLoss_cls = nn.NLLLoss(weight=weight_cls)
    best_loss = 99999.0

    for epoch in range(num_epochs):

        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        for i, datapack in enumerate(train_loader, 0):
            train_imgs = datapack[:, 0:3]
            train_det_masks = datapack[:, 3:4]
            train_cls_masks = datapack[:, 4:]

            train_det_masks = train_det_masks.long()
            train_cls_masks = train_cls_masks.long()

            train_det_masks = train_det_masks.view(
                train_det_masks.size()[0],
                train_det_masks.size()[2],
                train_det_masks.size()[3]
            )

            train_cls_masks = train_cls_masks.view(
                train_cls_masks.size()[0],
                train_cls_masks.size()[2],
                train_cls_masks.size()[3]
            )

            optimizer.zero_grad()
            train_det_out, train_cls_out = model(train_imgs)
            t_det_loss = NLLLoss_det(train_det_out, train_det_masks)
            t_cls_loss = NLLLoss_cls(train_cls_out, train_cls_masks)
            t_loss = t_det_loss + t_cls_loss
            t_loss.backward()
            optimizer.step()
            train_loss += t_loss.item()

            if i % 10 == 9:
                print('epoch: %3d, step: %3d loss: %.5f' % (epoch + 1, i + 1, train_loss / 10))
                writer.add_scalar('train_loss', train_loss, (210 * epoch + i + 1))
                train_loss = 0.0

        for i, datapack in enumerate(val_loader, 0):
            val_imgs = datapack[:, 0:3]
            val_det_masks = datapack[:, 3:4]
            val_cls_masks = datapack[:, 4:]

            val_det_masks = val_det_masks.long()
            val_det_masks = val_det_masks.view(
                val_det_masks.size()[0],
                val_det_masks.size()[2],
                val_det_masks.size()[3]
            )

            val_cls_masks = val_cls_masks.long()
            val_cls_masks = val_cls_masks.view(
                val_cls_masks.size()[0],
                val_cls_masks.size()[2],
                val_cls_masks.size()[3]
            )

            # optimizer.zero_grad()
            val_det_out, val_cls_out = model(val_imgs)
            v_det_loss = NLLLoss_det(val_det_out, val_det_masks)
            v_cls_loss = NLLLoss_cls(val_cls_out, val_cls_masks)
            v_loss = v_det_loss + v_cls_loss
            val_loss += v_loss.item()

            if i % val_steps == val_steps - 1:
                val_loss = val_loss / val_steps
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), './ckpt/test_att_global.pkl')
                end = time.time()
                time_spent = end - start
                writer.add_scalar('val_loss', val_loss, epoch)
                print('epoch: %3d, time: %.5f val_loss: %.5f' % (epoch + 1, time_spent, val_loss))
                val_loss = 0.0
                #p, r, f = MyMetrics(model)
                #writer.add_scalar('precision', p, epoch)
                #writer.add_scalar('recall', r, epoch)
                #writer.add_scalar('f1_score', f, epoch)
                #print('p:', p)
                #print('r:', r)
                #print('f:', f)
                print('******************************************************************************')

    # writer.export_scalars_to_json('./loss.json')
    writer.close()

if __name__ == '__main__':
    net = Attention_Net_Global()
    train(net, weight_det=[0.1, 2], weight_cls=[0.1, 4, 3, 6, 10], data_dir='./aug', target_size=64)