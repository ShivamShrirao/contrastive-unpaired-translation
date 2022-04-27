import os
from turtle import forward
import numpy as np
from tqdm import tqdm
import json

from zmq import device
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autograd

from options.train_options import TrainOptions
from utils import AverageMeter, tensors2im, reduce_loss, synchronize, cleanup, seed_everything, set_grads
from data import CreateDataLoader
from data.unaligned_dataset import UnAlignedDataset
from models.custom_unet import Unet, NLayerDiscriminator, PatchSampleF, GANLoss


class TrainModel:
    def __init__(self, args):
        self.device = torch.device('cuda', args.local_rank)
        self.unetG = Unet(self_attn=True).to(self.device)
        self.netD = NLayerDiscriminator(3).to(self.device)
        self.unetG.eval()
        with torch.inference_mode():
            feats = self.unetG(torch.randn(1, 3, 256, 256, device=self.unetG.device), encode_only=True)
        self.netF = PatchSampleF(use_mlp=True)
        self.netF.create_mlp(feats)
        del feats
        dist.init_process_group(backend="nccl")
        if args.sync_bn:
            self.unetG = nn.SyncBatchNorm.convert_sync_batchnorm(self.unetG)
            self.netD = nn.SyncBatchNorm.convert_sync_batchnorm(self.netD)
            self.netF = nn.SyncBatchNorm.convert_sync_batchnorm(self.netF)
        self.unetG = DDP(self.unetG, device_ids=[args.local_rank], output_device=args.local_rank,
                         broadcast_buffers=False)
        self.netD = DDP(self.netD, device_ids=[args.local_rank], output_device=args.local_rank,
                        broadcast_buffers=False)
        self.netF = DDP(self.netF, device_ids=[args.local_rank], output_device=args.local_rank,
                        broadcast_buffers=False)

        self.criterion_gan = GANLoss()
        dataset = UnAlignedDataset(args.dataset_dir, 512, args.phase)
        self.dataloader = CreateDataLoader(dataset, args.batch_size, workers=args.workers)
        dataset = UnAlignedDataset(args.dataset_dir, 512, args.phase)
        if args.local_rank == 0:
            val_dataset = UnAlignedDataset(args.dataset_dir, 1024, phase="test")
            val_dataset.img_names = val_dataset.img_names[:20]
            self.val_loader = CreateDataLoader(val_dataset, 2, workers=args.workers, shuffle=False, distributed=False)

        self.optG = optim.AdamW(self.unetG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wd)
        self.optD = optim.AdamW(self.unetD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wd)
        self.optF = optim.AdamW(self.unetF.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wd)
        self.scaler = amp.GradScaler(enabled=not args.no_amp)


    def calculate_NCE_loss(self, args, feat_k, feat_q):
        feat_k_pool, sample_ids = self.netF(feat_k, args.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, args.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
            total_nce_loss += crit(f_q, f_k) * args.lambda_NCE

        return total_nce_loss / len(feat_k)


    def forward(self, args, real_A, real_B):
        with amp.autocast(enabled=not args.no_amp):
            real = torch.cat((real_A, real_B), dim=0)
            pred, feats = self.unetG(real, get_feat=True)
            batch_size = real_A.size(0)
            fake_B = pred[:batch_size]
            idt_B = pred[batch_size:]

            fake_out = self.netD(fake_B)
            real_out = self.netD(real_B)

            _, feat_q = self.unetG(fake_B, get_feat=True, encode_only=True)
            feat_k = [ft[:batch_size] for ft in feats]
            nce_loss_A = self.calculate_NCE_loss(args, feat_k, feat_q)

            _, feat_q = self.unetG(idt_B, get_feat=True, encode_only=True)
            feat_k = [ft[batch_size:] for ft in feats]
            nce_loss_B = self.calculate_NCE_loss(args, feat_k, feat_q)

            lossG = self.criterion_gan(fake_out, True) * args.lambda_GAN + (nce_loss_A + nce_loss_B) * 0.5
            lossD = (self.criterion_gan(fake_out, False)
                     + self.criterion_gan(real_out, True)) * 0.5

        gradsD = autograd.grad(self.scaler.scale(lossD), self.netD.parameters(), retain_graph=True)
        GF_params = list(self.unetG.parameters()) + list(self.netF.parameters())
        gradsG = autograd.grad(self.scaler.scale(lossG), GF_params)
        set_grads(gradsD, self.netD.parameters())
        set_grads(gradsG, GF_params)
        del gradsG, gradsD

        self.scaler.step(self.optG)
        self.optG.zero_grad(set_to_none=True)
        self.scaler.step(self.optF)
        self.optF.zero_grad(set_to_none=True)
        # if lossD>=args.min_lossD:
        self.scaler.step(self.optD)
        self.optD.zero_grad(set_to_none=True)


    def train_epoch(self, args, epoch):
        mae_losses = AverageMeter()
        feat_losses = AverageMeter()

        info = {}
        with tqdm(self.dataloader, desc=f"Epoch {epoch:>2}", disable=args.local_rank != 0) as pbar:
            for step, (real_A, real_B) in enumerate(pbar):
                real_A = real_A.to(self.device, non_blocking=True)
                real_B = real_B.to(self.device, non_blocking=True)
                mae_loss, feat_loss = self.forward(args, real_A, real_B)

                mae_losses.update(reduce_loss(mae_loss.detach()), real_A.size(0))
                feat_losses.update(reduce_loss(feat_loss.detach()), real_A.size(0))

                if args.local_rank == 0:
                    if not step % args.log_interval:
                        info = {
                            'MAE_Loss': float(mae_losses.avg),
                            'Feat_Loss': float(feat_losses.avg),
                        }
                        pbar.set_postfix(info)
                        if args.use_wandb:
                            wandb.log(info)
                            if not step % args.img_log_interval:
                                im_dict = {}
                                im_dict["inp"] = [wandb.Image(im) for im in tensors2im(real_A)]
                                im_dict["pred"] = [wandb.Image(im) for im in tensors2im(pred)]
                                im_dict["label"] = [wandb.Image(im) for im in tensors2im(real_B)]
                                wandb.log(im_dict)
                if args.lr_schedule:
                    self.lr_scheduler.step()
                self.scaler.update()
        return info

    def train_loop(self, args):
        im_size_dict = {6: 1024}
        keys = np.array(list(im_size_dict.keys()))
        comp = args.init_epoch >= keys
        if comp.any():
            kidx = max(np.argmax(np.logical_not(comp)) - 1, 0)
            key = keys[kidx]
            dataset = UnAlignedDataset(args.dataset_dir, im_size_dict[key], args.phase)
            self.dataloader = CreateDataLoader(dataset, args.batch_size)
            if args.local_rank == 0:
                print(f"[*] Increasing image size to {im_size_dict[key]}px and batch size set to {args.batch_size}.")
        self.validate(args)
        self.unetG.train()
        for epoch in range(args.init_epoch, args.epochs):
            if epoch in im_size_dict:
                if args.use_wandb and args.local_rank == 0:
                    wandb.config.update(dict(batch_size=args.batch_size // 4), allow_val_change=True)
                    print(f"[*] Increasing image size to {im_size_dict[epoch]}px and batch size set to {args.batch_size}.")
                else:
                    args.batch_size //= 4
                dataset = UnAlignedDataset(args.dataset_dir, im_size_dict[epoch], args.phase)
                self.dataloader = CreateDataLoader(dataset, args.batch_size)
            self.dataloader.sampler.set_epoch(epoch)
            info = self.train_epoch(args, epoch)
            if args.local_rank == 0:
                if args.use_wandb:
                    wandb.log({'epoch': epoch})
            self.save_models(args, 'latest', info)
            if not epoch % 2:
                self.save_models(args, epoch, info)
                self.validate(args)

    @torch.no_grad()
    def validate(self, args):
        if args.local_rank != 0 or not args.use_wandb:
            return
        self.unetG.eval()
        im_dict = {"val_inp": [], "val_pred": [], "val_lbl": []}
        with tqdm(self.val_loader, desc=f"Validating") as pbar:
            for step, (img, lbl) in enumerate(pbar):
                img = img.to(self.device, non_blocking=True)
                with amp.autocast(enabled=not args.no_amp):
                    pred = self.unetG(img)
                im_dict["val_inp"] += [wandb.Image(im) for im in tensors2im(img)]
                im_dict["val_lbl"] += [wandb.Image(im) for im in tensors2im(lbl)]
                im_dict["val_pred"] += [wandb.Image(im) for im in tensors2im(pred)]
        wandb.log(im_dict)
        self.unetG.train()

    def save_models(self, args, epoch='latest', info={}):
        if args.local_rank == 0:
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            torch.save(self.unetG.state_dict(), os.path.join(args.checkpoint_dir, f"unetG_{epoch}.pth"))
            torch.save(self.optG.state_dict(), os.path.join(args.checkpoint_dir, f"optG_{epoch}.pth"))
            torch.save(info, os.path.join(args.checkpoint_dir, f"info_{epoch}.pth"))
            print("[+] Weights saved.")

    def load_models(self, args, epoch='latest'):
        synchronize()
        map_location = {'cuda:0': f'cuda:{args.local_rank}'}
        try:
            self.unetG.load_state_dict(torch.load(os.path.join(
                args.checkpoint_dir, f"unetG_{epoch}.pth"), map_location=map_location))
            if args.phase == 'train':
                self.optG.load_state_dict(torch.load(os.path.join(
                    args.checkpoint_dir, f"optG_{epoch}.pth"), map_location=map_location))
            if args.local_rank == 0:
                print(f"[+] Weights loaded for {epoch} epoch.")
        except FileNotFoundError as e:
            if args.local_rank == 0:
                print(f"[!] {e}, skipping weights loading.")


def main():
    args = TrainOptions().parse()
    if args.local_rank == 0:
        print(json.dumps(dict(args), indent=4))
    torch.cuda.set_device(args.local_rank)
    seed_everything(args.seed)
    try:
        tm = TrainModel(args)
        tm.load_models(args)
        tm.train_loop(args)
        tm.save_models(args)
    except KeyboardInterrupt:
        print("[!] Keyboard Interrupt! Cleaning up and shutting down.")
    finally:
        cleanup()


if __name__ == '__main__':
    main()
