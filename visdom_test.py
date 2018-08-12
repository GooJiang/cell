from sfcn_model import Attention_Net_Global,Net
from tensorboardX import SummaryWriter
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import dataset
from config import Config
import visdom
import os
import scipy.misc as misc
import scipy.io as sio
from util import non_max_suppression, get_metrics
from torch.utils.data import DataLoader
from torch.autograd import Variable
from train import data_unpack
from image_producer import CellImageDataset

BATCH_SIZE = Config.image_per_gpu * Config.gpu_count
epsilon = 1e-7
TARGET_SIZE = 64

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'aug')
CKPT_DIR = os.path.join(ROOT_DIR, 'ckpt')

def MyMetrics(model,target_size=256):
    path = './aug'
    tp_num = [0, 0]
    gt_num = [0, 0]
    pred_num = [0, 0]
    precision = [0, 0]
    recall = [0, 0]
    f1_score = [0, 0]
    cell_type_group = ['epithelial', 'fibroblast', 'inflammatory', 'others']
    gt_type = ['Detection', 'Classification']

    for i in range(81, 101):
        filename = os.path.join(path, 'img' + str(i) + '_1.bmp')
        imgname = 'img' + str(i)
        if os.path.exists(filename):
            img = misc.imread(filename)
            img = misc.imresize(img, (target_size, target_size))
            img = img - 128.
            img = img / 128.
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = np.transpose(img, (0, 3, 1, 2))
            img = torch.Tensor(img)
            img_result = model(img)

            for index, ground_truth in enumerate(gt_type, 0):
                gtpath = os.path.join('F:/CRCHistoPhenotypes_2016_04_28',ground_truth)
                result = img_result[index]
                result = result.cpu().detach().numpy()
                result = np.transpose(result, (0, 2, 3, 1))[0]
                result = np.exp(result)
                for j in range(1, result.shape[-1]):
                    result_type = result[:, :, j]
                    result_type = misc.imresize(result_type, (500, 500))
                    result_type = result_type / 255.
                    boxes = non_max_suppression(result_type)
                    if ground_truth == 'Detection':
                        matname = imgname + '_detection.mat'
                    else:
                        matname = imgname + '_' + cell_type_group[j-1] + '.mat'
                    matpath = os.path.join(gtpath, imgname, matname)
                    gt = sio.loadmat(matpath)['detection']
                    pred = []
                    for k in range(boxes.shape[0]):
                        x1 = boxes[k, 0]
                        y1 = boxes[k, 1]
                        x2 = boxes[k, 2]
                        y2 = boxes[k, 3]
                        cx = int(x1 + (x2 - x1) / 2)
                        cy = int(y1 + (y2 - y1) / 2)
                        pred.append([cx, cy])
                    p, r, f1, tp = get_metrics(gt, pred)
                    tp_num[index] += tp
                    gt_num[index] += gt.shape[0]
                    pred_num[index] += np.array(pred).shape[0]

    for index, ground_truth in enumerate(gt_type, 0):
        precision[index] = tp_num[index] / (pred_num[index] + epsilon)
        recall[index] = tp_num[index] / (gt_num[index] + epsilon)
        f1_score[index] = 2 * (precision[index] * recall[index] / (precision[index] + recall[index] + epsilon))

    return precision, recall, f1_score


def train_epoch(dataloader_train, optimizer, model, batch_size, det_loss_fn, cls_loss_fn, type=None):
    steps = int(len(dataloader_train)/batch_size)
    total_loss = 0.0
    time_now = time.time()
    dataiter_train = iter(dataloader_train)
    print('total steps: {}'.format(steps))
    for i, (data_train, det_mask, cls_mask) in enumerate(dataloader_train):
        data_train, det_mask, cls_mask = data_train.type(torch.FloatTensor), det_mask.type(torch.LongTensor), cls_mask.type(torch.LongTensor)
        data_train = data_train.cuda()
        data_det_mask = det_mask.cuda()
        data_cls_mask = cls_mask.cuda()
        train_det_out, train_cls_out = model(data_train)
        det_loss = det_loss_fn(train_det_out, data_det_mask)
        optimizer.zero_grad()
        det_loss.backward()
        optimizer.step()
        print('loss {} finishes training'.format(det_loss.item()))


        """
    for step in range(steps):
        data_train, det_mask, cls_mask = next(dataiter_train)
        #print('size ', data_train.shape, det_mask.shape, cls_mask.shape)
        data_train, det_mask, cls_mask = data_train.type(torch.FloatTensor), det_mask.type(torch.LongTensor), cls_mask.type(torch.LongTensor)
        data_train = Variable(data_train.cuda())
        data_det_mask = Variable(det_mask.cuda())
        data_cls_mask = Variable(cls_mask.cuda())

        train_det_out, train_cls_out = model(data_train)
        #print('train_det_out: {}, train_cls_out: {}'.format(train_det_out.shape, train_cls_out.shape))
        det_loss = det_loss_fn(train_det_out, data_det_mask)
        cls_loss = cls_loss_fn(train_cls_out, data_cls_mask)
        optimizer.zero_grad()
        det_loss.backward()
        optimizer.step()
        total_loss += det_loss.item()
        print('step {} at loss {} finishes training'.format(step, det_loss.item()))
"""

    return total_loss



def train(model, batch_size, weight_det=None, weight_cls=None, data_dir='',
          preprocess=True, gpu=True, num_epochs=Config.epoch, target_size=256):
    from train import tune_weights, data_prepare, data_unpack
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    weight_det, weight_cls = tune_weights(weight_det, weight_cls)
    weight_det, weight_cls = weight_det.cuda(), weight_cls.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    batch_size_train = batch_size * Config.gpu_count
    batch_size_valid = batch_size * Config.gpu_count * 2
    num_workers =  Config.gpu_count
    #model = torch.nn.DataParallel(model, device_ids=None)
    model = model.cuda()
    NLLLoss_det = nn.NLLLoss(weight=weight_det).cuda()
    NLLLoss_cls = nn.NLLLoss(weight=weight_cls).cuda()
    #optimizer = optim.Adam(model.parameters(), weight_decay=0.01)


    dataset_train = CellImageDataset(os.path.join(DATA_DIR, 'train'), image_size=(256, 256))
    dataloader_train = DataLoader(dataset_train)

    best_loss = 99999.0
    train_loss_list = []
    train_step_list = []
    val_loss_list = []
    val_det_p = []
    val_det_r = []
    val_det_f = []
    val_cls_p = []
    val_cls_r = []
    val_cls_f = []
    epoch_list = []
    #vis = visdom.Visdom(env=u'test1')

    for epoch in range(num_epochs):
        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        print('epoch {} start to train'.format(epoch))
        loss = train_epoch(dataloader_train, optimizer=optimizer, model=model, batch_size=batch_size,
                               det_loss_fn=NLLLoss_det, cls_loss_fn=NLLLoss_cls)

        if i % 10 == 9:
            print('epoch: %3d, loss: %.5f' % (epoch + 1, loss / 10))
            train_loss_list.append(loss/10)
            #train_step_list.append((train_steps * epoch + i + 1))

            trace_p = dict(x=train_step_list, y=train_loss_list, mode="lines", type='custom', name='train_loss')
            #layout = dict(title="train loss", xaxis={'title': 'step'}, yaxis={'title': 'loss'})

            #vis._send({'data': [trace_p], 'layout': layout, 'win': 'trainloss'})
            train_loss = 0.0

        """
        for i, datapack in enumerate(val_loader, 0):
            val_imgs, val_det_masks, val_cls_masks = data_unpack(datapack)

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
                print('epoch: %3d, time: %.5f val_loss: %.5f' % (epoch + 1, time_spent, val_loss))
                val_loss_list.append(val_loss)
                epoch_list.append(epoch + 1)

                trace_p = dict(x=epoch_list, y=val_loss_list, mode='lines', type='custom', name='val_loss')
                layout = dict(title='val loss', xaxis={'title': 'epoch'}, yaxis={'title': 'loss'})
                vis._send({'data': [trace_p], 'layout': layout, 'win': 'valloss'})

                val_loss = 0.0
                p, r, f = MyMetrics(model, target_size=target_size)
                print('p:', p)
                print('r:', r)
                print('f:', f)
                val_det_p.append(p[0])
                val_det_r.append(r[0])
                val_det_f.append(f[0])
                val_cls_p.append(p[1])
                val_cls_r.append(r[1])
                val_cls_f.append(f[1])
                trace_det_f = dict(x=epoch_list, y=val_det_f, mode='lines', type='custom', name='val_det_f')
                layout = dict(title='val det f', xaxis={'title': 'epoch'}, yaxis={'title': 'F'})
                vis._send({'data': [trace_det_f], 'layout': layout, 'win': 'valdetf'})
                trace_det_p = dict(x=epoch_list, y=val_det_p, mode='lines', type='custom', name='val_det_p')
                trace_det_r = dict(x=epoch_list, y=val_det_r, mode='lines', type='custom', name='val_det_r')
                layout = dict(title='val det pr', xaxis={'title': 'epoch'}, yaxis={'title': 'PR'})
                vis._send({'data': [trace_det_p,trace_det_r], 'layout': layout, 'win': 'valdetpr'})
                trace_p = dict(x=epoch_list, y=val_cls_f, mode='lines', type='custom', name='val_cls_f')
                layout = dict(title='val cls f', xaxis={'title': 'epoch'}, yaxis={'title': 'F'})
                vis._send({'data': [trace_p], 'layout': layout, 'win': 'valclsf'})
                trace_cls_p = dict(x=epoch_list, y=val_cls_p, mode='lines', type='custom', name='val_cls_p')
                trace_cls_r = dict(x=epoch_list, y=val_cls_r, mode='lines', type='custom', name='val_cls_r')
                layout = dict(title='val cls pr', xaxis={'title': 'epoch'}, yaxis={'title': 'PR'})
                vis._send({'data': [trace_cls_p, trace_cls_r], 'layout': layout, 'win': 'valclspr'})
                print('******************************************************************************')
"""

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    net = Net()
    train(net, weight_det=[0.1, 2], weight_cls=[0.1, 4, 3, 6, 10], data_dir='./aug', target_size=TARGET_SIZE, batch_size=2)
