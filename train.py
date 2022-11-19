import argparse
import logging
import os
import pprint
import random

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml

from dataset.dual_dg_dataset import DG_Dataset
from model import DeepLabV3Plus
from util.dcc_loss import BalancedSoftmaxCE
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed, setup_seed

parser = argparse.ArgumentParser(description='DG-Dual Consistency Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=400.0, help='consistency_rampup')


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img).argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIOU = np.mean(iou_class) * 100.0
    model.train()
    return mIOU, iou_class

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    
    setup_seed(rank, cfg)

    if cfg['head'] == 'deeplabv3plus':
        model = DeepLabV3Plus(cfg)
        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                         {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                          'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    else:
        raise NotImplementedError('Unsupported Segmentation Head {}'.format(cfg['head']))
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    # if not cfg['freeze_bn']:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'DCCLoss':
        criterion = BalancedSoftmaxCE(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    consistency_criterion = softmax_mse_loss
    
    with open(os.path.join(cfg['train_split'], cfg['dataset'], 'train.txt'), 'r') as f:
        train_ids = f.read().splitlines()
        random.shuffle(train_ids)
    
    with open(os.path.join(cfg['val_split'], '{}_val.txt'.format(cfg['val_dataset'])), 'r') as f:
        val_ids = f.read().splitlines()

    trainset = DG_Dataset('train', train_ids, cfg['crop_size'], True)
    valset = DG_Dataset('val', val_ids)
    
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)
    
    iters = 0
    total_epochs = cfg['max_iters'] // len(trainloader) + 1
    total_iters = cfg['max_iters']
    previous_best = 0.0

    for epoch in range(total_epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.8f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))
        
        model.train()
        total_loss = 0.0

        trainsampler.set_epoch(epoch)

        for i, (img_s1, img_s2, mask) in enumerate(trainloader):

            b, _, h, w = img_s1.shape

            img_s1, img_s2, mask = img_s1.cuda(), img_s2.cuda(), mask.cuda()

            img = torch.cat((img_s1, img_s2), dim=0)
            mask = torch.cat((mask, mask), dim=0)

            pred = model(img)

            ## calculate the loss
            seg_loss = criterion(pred, mask)

            soft_pred = torch.softmax(pred, dim=1)
            soft_pred = soft_pred.reshape(2, b, cfg['nclass'], h, w)
            soft_pred = torch.mean(soft_pred, dim=0) #(b, c, h, w)
            uncertainty = -1.0 * torch.sum(soft_pred * torch.log(soft_pred + 1e-6), dim=1, keepdim=True) # (b, 1, h, w)
            consistency_weight = get_current_consistency_weight(iters//1500, args)
            consistency_dist = consistency_criterion(pred[:b], pred[b:]) # (b, c, h, w)
            threshold = (0.75 + 0.25 * sigmoid_rampup(iters, total_iters)) * np.log(2)
            mask = (uncertainty >= threshold).float()
            consistency_dist = torch.sum(mask * consistency_dist)/(2 * torch.sum(mask) + 1e-16)
            consistency_loss = consistency_weight * consistency_dist

            loss = seg_loss + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if (iters % cfg['log_step'] == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(iters, total_loss / (i+1)))
            
            if (iters % cfg['interval'] == 0):
                if cfg['dataset'] == 'cityscapes':
                    eval_mode = 'original'
                else:
                    eval_mode = 'original'
                mIOU, iou_class = evaluate(model, valloader, eval_mode, cfg)

                if rank == 0:
                    logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU))
                    torch.save(model.module.state_dict(),
                            os.path.join(args.save_path, 'iter_%d_%.2f.pth' % (iters, mIOU)))
                
                if mIOU > previous_best and rank == 0:
                    if previous_best != 0:
                        os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], previous_best)))
                    previous_best = mIOU
                    torch.save(model.module.state_dict(),
                            os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], mIOU)))
            
            if iters >= total_iters:
                if rank == 0:
                    torch.save(model.module.state_dict(),
                        os.path.join(args.save_path, 'iter_%d.pth' % (iters)))
                    return rank, 'Finish Train.'
                else:
                    return rank, 'Finish Train.'
        

if __name__ == '__main__':
    rank, finish = main()
    if rank == 0:
        print(finish)