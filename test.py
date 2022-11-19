import argparse
import logging
import os
import pprint

import torch
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from dataset.dg_dataset import DG_Dataset, category_list
from model import DeepLabV3Plus
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log, color_map, colorize
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(description='DG-Baseline Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--vis', action='store_true', help='Save Vis Results')
parser.add_argument('--vis_mask', action='store_true', help='Save Masks')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--selected_classes', default=[0,10,2,1,8], help="poly_power")


def evaluate(model, loader, mode, cfg, args, rank):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window', 'pooling']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    tbar = tqdm(loader, ncols=70)

    with torch.no_grad():
        for i, (img, mask, id) in enumerate(tbar):
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
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)
            
            elif mode == 'pooling':
                pooling_size = cfg['pooling_size']
                b, _, h, w = img.shape
                img = F.interpolate(img, size=pooling_size, mode='bilinear', align_corners=True)
                pred = model(img)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True).argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = pred.argmax(dim=1)
            
            if args.vis:
                colormap = color_map('cityscapes')
                for j, (name) in enumerate(id):
                    gray = np.uint8(pred[j].cpu().numpy())
                    color_path = os.path.join(os.path.join(args.save_path, 'vis', name.split(' ')[1].split('/')[-1]))
                    color = colorize(gray, colormap)
                    color.save(color_path)
            
            if args.vis_mask:
                colormap = color_map('cityscapes')
                for j, (name) in enumerate(id):
                    gray = np.uint8(mask[j].numpy())
                    color_path = os.path.join(os.path.join(args.save_path, '../../mask', name.split(' ')[1].split('/')[-1]))
                    color = colorize(gray, colormap)
                    color.save(color_path)

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

    return mIOU, iou_class

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))
        logger.info('{}\n'.format(pprint.pformat(args)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        if args.vis:
            os.makedirs(os.path.join(args.save_path, 'vis'), exist_ok=True)
        if args.vis_mask:
            os.makedirs(os.path.join(args.save_path, '../../mask'), exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    if cfg['head'] == 'deeplabv3plus':
        model = DeepLabV3Plus(cfg)
    else:
        raise NotImplementedError('Unsupported Segmentation Head {}'.format(cfg['head']))
    
    model.load_state_dict(torch.load(cfg['load_from']))
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)
    
    with open(os.path.join(cfg['val_split'], '{}_val.txt'.format(cfg['val_dataset'])), 'r') as f:
        val_ids = f.read().splitlines()

    valset = DG_Dataset('val', val_ids, dataset=cfg['val_dataset'])
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=2, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)

    eval_mode = cfg['eval_mode']
    mIOU, iou_class = evaluate(model, valloader, eval_mode, cfg, args, rank)

    if rank == 0:
        logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU))
        with open(os.path.join(args.save_path, 'eval_result.txt'), 'w') as f:
            f.write('***** Evaluation {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU))
            for i, cate_name in enumerate(category_list):
                logger.info('IOU.{}: {:.2f}\n'.format(cate_name, iou_class[i] * 100))
                f.write('IOU.{}: {:.2f}\n'.format(cate_name, iou_class[i] * 100))
                f.flush()
        f.close()
        

if __name__ == '__main__':
    main()