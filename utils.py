import os
import random
from threading import Thread

import numpy as np
import torch
import torch.distributed as dist
import wandb


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed=3407):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_grads(grads, params):
    for g, p in zip(grads, params):
        p.grad = g


def Threaded(fn):                                   # annotation wrapper to launch a function as a thread
    def wrapper(*args, **kwargs):
        t = Thread(target=fn, args=args, kwargs=kwargs)
        t.start()
        return t
    return wrapper


@Threaded
def log_imgs_wandb(**kwargs):
    im_dict = {}
    for im, arr in kwargs:
        im_dict[im] = [wandb.Image(im) for im in tensors2im(arr)]
    wandb.log(im_dict)  


def tensors2im(x):
    x = x.detach()
    mx = x.amax(dim=(1, 2, 3), keepdim=True)
    mn = x.amin(dim=(1, 2, 3), keepdim=True)
    x = x.sub_(mn).div_(mx - mn)
    x = x.mul_(255.).to(torch.uint8)
    x = x.permute(0, 2, 3, 1)
    return x.cpu().numpy()


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss

    with torch.no_grad():
        dist.reduce(loss, dst=0, async_op=True)
        if dist.get_rank() == 0:
            loss /= world_size

    return loss


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    if dist.get_world_size() == 1:
        return
    dist.barrier()


def cleanup(distributed=True):
    if distributed:
        dist.destroy_process_group()
