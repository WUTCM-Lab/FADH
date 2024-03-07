import mindspore
import mindspore as ms
import mindspore.communication as comm
from mindspore import context
import mindspore.nn as nn

import os
import time
import argparse
import yaml
import copy
import numpy as np

from data import get_loaders
from tools.logger import get_logger
from model_utils.device_adapter import get_device_id
from model_utils.local_adapter import set_device
from tools.lr_scheduler import get_lr
from tools.utils import cpu_affinity, AverageMeter
from model_utils.cell import cast_amp
from layers.FADH import factory,TotalLoss,MyWithLossCell
from mindspore import ops
from tqdm import tqdm
from mindspore.profiler import Profiler

ms.set_seed(1)

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSICD_FADH.yaml', type=str,
                         help='path to a yaml options file')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle,Loader=yaml.FullLoader)

    return options

def compute_result(dataloader, net, device):
    bi,ba, clses = [], [],[]
    net.set_train(False)
    for img, aud,cls in tqdm(dataloader):
        clses.append(cls)
        image,audio=net(img,aud)
        bi.append(image)
        ba.append(audio)
    return ops.cat(bi).sign(), ops.cat(ba).sign(),ops.cat(clses)
def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH
def CalcTopMap(qB, rB, queryL, retrievalL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        if topk==0:
            tgnd = gnd[:]
        else:
            tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap



if __name__ == '__main__':

    options = parser_options()
    device = options['device_target']

    logger = get_logger(options['logs']['logger_name'], options['rank'])
    logger.save_args(options)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    loss_meter = AverageMeter('loss')

    model=factory(options)

    train_loader, test_loader = get_loaders(options)
    logger.info('Finish loading dataset')
    best_acc = 0


    steps_per_epoch = train_loader.get_dataset_size()
    lr = get_lr(options, steps_per_epoch)
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=lr)
    crit = TotalLoss()
    net_with_crit = MyWithLossCell(model, crit)
    train_net = nn.TrainOneStepCell(net_with_crit, optimizer)
    train_net.set_train()

    first_step = True
    t_end = time.time()
    for epoch_idx in range(options['max_epoch']):
        train_net.set_train(True)
        for step_idx, data in enumerate(train_loader):
            img = data[0]
            aud = data[1]
            label = data[2]
            loss = train_net(img, aud, label)
            loss_meter.update(loss.asnumpy())

            # it is used for loss, performance output per config.log_interval steps.
            if (epoch_idx * steps_per_epoch + step_idx) % options['log_interval'] == 0:
                time_used = time.time() - t_end
                if first_step:
                    fps = options['per_batch_size'] * options['group_size'] / time_used
                    per_step_time = time_used * 1000
                    first_step = False
                else:
                    fps = options['per_batch_size'] * options['log_interval'] * options['group_size'] / time_used
                    per_step_time = time_used / options['log_interval'] * 1000
                logger.info('epoch[{}], iter[{}], {}, fps:{:.2f} imgs/sec, '
                                   'lr:{}, per step time: {}ms'.format(epoch_idx + 1, step_idx + 1,
                                                                       loss_meter, fps, lr[step_idx], per_step_time))
                t_end = time.time()
                loss_meter.reset()
        tst_img_binary, tst_aud_binary, tst_label = compute_result(test_loader, model, device=device)

        trn_img_binary, trn_aud_binary, trn_label = compute_result(train_loader, model, device=device)
        mAP = CalcTopMap(tst_img_binary.asnumpy(), trn_aud_binary.asnumpy(), tst_label.asnumpy(), trn_label.asnumpy(),
                         0)
        mAP1 = CalcTopMap(tst_img_binary.asnumpy(), trn_aud_binary.asnumpy(), tst_label.asnumpy(), trn_label.asnumpy(),
                          1)
        mAP5 = CalcTopMap(tst_img_binary.asnumpy(), trn_aud_binary.asnumpy(), tst_label.asnumpy(), trn_label.asnumpy(),
                          5)
        mAP10 = CalcTopMap(tst_img_binary.asnumpy(), trn_aud_binary.asnumpy(), tst_label.asnumpy(), trn_label.asnumpy(),
                           10)

        print("img to aud mAP:%f mAP1:%f mAP5:%f mAP10:%f" % (mAP, mAP1, mAP5, mAP10))
        mAP = CalcTopMap(tst_aud_binary.asnumpy(), trn_img_binary.asnumpy(), tst_label.asnumpy(), trn_label.asnumpy(),
                         0)
        mAP1 = CalcTopMap(tst_aud_binary.asnumpy(), trn_img_binary.asnumpy(), tst_label.asnumpy(), trn_label.asnumpy(),
                          1)
        mAP5 = CalcTopMap(tst_aud_binary.asnumpy(), trn_img_binary.asnumpy(), tst_label.asnumpy(), trn_label.asnumpy(),
                          5)
        mAP10 = CalcTopMap(tst_aud_binary.asnumpy(), trn_img_binary.asnumpy(), tst_label.asnumpy(), trn_label.asnumpy(),
                           10)

        print("aud to img mAP:%f mAP1:%f mAP5:%f mAP10:%f" % (mAP, mAP1, mAP5, mAP10))
        train_acc = mAP

        if train_acc > best_acc:
            best_acc = train_acc
            best_net = copy.deepcopy(model)
            ckpt_name = os.path.join(options['output_dir'], "FADH_{}_{}.ckpt".format(epoch_idx + 1, steps_per_epoch))
            mindspore.save_checkpoint(best_net, ckpt_name)
    print("-----------------------------------------------------")
    print("best_acc = %.6f" % (best_acc))
    logger.info('==========end training=============')