import argparse
import os
import random
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import D, ConcatDataset
from model import M
from adv import EMA, compute_kl_loss, AWP

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--work_dir', type=str, default='work_dirs/exp1')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--pretrained', type=str, default="")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

args = parse_args()
setup_seed(args.seed)

world_size = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=world_size)

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
print_freq = args.print_freq
work_dir = args.work_dir
warmup_ratio = args.warmup_ratio
nlayers = args.nlayers

if args.local_rank == 0:
    os.system("mkdir -p %s"%work_dir)
    os.system("mkdir -p %s/checkpoints"%work_dir)
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
            
    fh = logging.FileHandler(work_dir + '/log.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

if args.debug:
    train_dataset = D('../val.npy', training=True)
else:
    if world_size == 8:
        train_dataset = D('../train_full_%d.npy'%args.local_rank, training=True)
    elif world_size == 4:
        train_dataset = ConcatDataset([
            D('../train_full_%d.npy'%(args.local_rank*2), training=True),
            D('../train_full_%d.npy'%(args.local_rank*2+1), training=True),
        ])

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

if args.debug:
    val_dataset = D('../val.npy')
else:
    val_dataset = D('../val.npy')
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, sampler=val_sampler)


model = M(nlayers)
model.cuda(args.local_rank)

ema = EMA(model, 0.9998)
ema.register()


opt = AdamW(model.parameters(), lr=lr)
stepsize = len(train_dataset) // batch_size + 1
total_steps = stepsize * epochs
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_ratio*total_steps, num_training_steps=total_steps)

awp = AWP(model=model, optimizer=opt, adv_lr=3e-5,
              adv_eps=1e-2, start_epoch=1)



#awp = AWP(model=model, optimizer=opt, adv_lr=1e-4, adv_eps=1e-2, start_epoch=1)

model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

#if args.pretrained != "":
#    ckpt = torch.load(args.pretrained, map_location='cpu')
#    
#    model.load_state_dict(ckpt['state_dict'], strict=False) 

maxacc = 0.
for e in range(epochs):

    model.train()
    for b, batch in enumerate(train_loader):
        # get input batch
        data, label, data2 = [x.cuda(args.local_rank) for x in batch]

        bs = data.shape[0]
        # since we want to obtain sample-level rdrop, we concatenate the data and label into a batch
        data = torch.cat([data, data2], dim=0)
        label = torch.cat([label, label], dim=0)

        res = model(data)
        res_softmax = F.softmax(res, dim=1)
        loss = F.cross_entropy(res, label, label_smoothing=0.5) # cross entropy loss with label smoothing 0.5
        label_onehot = F.one_hot(label, num_classes=250)
        loss_poly = 1. - (label_onehot * res_softmax).sum(dim=1).mean() # poly loss
        loss += loss_poly
        loss_kl = compute_kl_loss(res[:bs], res[bs:]) # rdrop loss
        loss += loss_kl
        loss.backward()


        awp.attack_backward(e) # AWP attach, and then re-compute the loss again
        res = model(data)
        res_softmax = F.softmax(res, dim=1)
        loss = F.cross_entropy(res, label, label_smoothing=0.5)
        loss_poly = 1. - (label_onehot * res_softmax).sum(dim=1).mean()
        loss += loss_poly
        loss_kl = compute_kl_loss(res[:bs], res[bs:])
        loss += loss_kl
        loss.backward()
        awp.restore()


        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) # clip the gradient  
        opt.step()
        ema.update()
        scheduler.step()
        opt.zero_grad()

        if args.local_rank == 0 and b % print_freq == 0:
            logger.info("Epoch %d Batch %d Loss %f"%(e, b, loss.item()))

    model.eval()
    ema.apply_shadow()

    pred_all = 0.
    pred_correct = 0.
    for b, batch in enumerate(val_loader):
        data, label = [x.cuda(args.local_rank) for x in batch]
        with torch.no_grad():
            res = model(data)
        res = res.argmax(dim=-1)

        res = concat_all_gather(res)
        label = concat_all_gather(label)

        tmp = (res == label).float()
        pred_all += tmp.shape[0]
        pred_correct += tmp.sum()
    acc = pred_correct / pred_all
    if args.local_rank == 0:
        logger.info("Val Epoch %d Acc %f"%(e, acc))


    if args.local_rank == 0:
        maxacc = acc
        ckpt = {'state_dict': model.state_dict()}
        torch.save(ckpt, work_dir + '/checkpoints/epoch_%d.pth'%e)

    ema.restore()

