import os
import numpy as np
from tqdm import tqdm

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autograd
from torch.optim import lr_scheduler
from torchinfo import summary

from options.train_options import TrainOptions
from utils import AverageMeter, reduce_loss, synchronize, cleanup, seed_everything, set_grads, log_imgs_wandb
from data import CreateDataLoader
from data.unaligned_dataset import UnAlignedDataset
from models.custom_unet import Unet, NLayerDiscriminator, PatchSampleF, GANLoss, PatchNCELoss, get_norm_layer, init_weights


class TrainModel:
    def __init__(self, args):
        self.device = torch.device('cuda', args.local_rank)
        self.netG = Unet(args.input_nc, args.output_nc, 32, self_attn=False).to(self.device)
        init_weights(self.netG, args.init_type, args.init_gain)
        norm_layer = get_norm_layer(args.normD)
        self.netD = NLayerDiscriminator(args.output_nc, args.ndf, args.n_layers_D, norm_layer).to(self.device)
        init_weights(self.netD, args.init_type, args.init_gain)
        with torch.no_grad():
            feats = self.netG(torch.randn(8, args.input_nc, 256, 512, device=self.device), get_feat=True, encode_only=True)
        self.netF = PatchSampleF(use_mlp=True, nc=args.netF_nc)
        self.netF.create_mlp(feats)
        self.netF = self.netF.to(self.device)
        init_weights(self.netF, args.init_type, args.init_gain)
        summary(self.netG, (1, args.input_nc, 256, 512))
        summary(self.netD, (1, args.output_nc, 256, 512))
        summary(self.netF, input_data=[feats])
        dist.init_process_group(backend="nccl")
        if args.sync_bn:
            self.netG = nn.SyncBatchNorm.convert_sync_batchnorm(self.netG)
            self.netD = nn.SyncBatchNorm.convert_sync_batchnorm(self.netD)
            self.netF = nn.SyncBatchNorm.convert_sync_batchnorm(self.netF)
        self.netG = DDP(self.netG, device_ids=[args.local_rank], output_device=args.local_rank,
                        broadcast_buffers=False)
        self.netD = DDP(self.netD, device_ids=[args.local_rank], output_device=args.local_rank,
                        broadcast_buffers=False)
        self.netF = DDP(self.netF, device_ids=[args.local_rank], output_device=args.local_rank,
                        broadcast_buffers=False)

        self.criterion_gan = GANLoss()
        self.criterionNCE = [PatchNCELoss(args).to(self.device) for _ in range(len(feats))]
        self.loss_names = ['lossG', 'lossD', 'nce_loss_tot']
        dataset = UnAlignedDataset(args.dataroot, (256, 512), args.phase)
        self.dataloader = CreateDataLoader(dataset, args.batch_size, workers=args.workers)
        # if args.local_rank == 0:
        #     val_dataset = UnAlignedDataset(args.dataroot, 1024, phase="test")
        #     val_dataset.img_names = val_dataset.img_names[:20]
        #     self.val_loader = CreateDataLoader(val_dataset, 2, workers=args.workers, shuffle=False, distributed=False)

        self.optG = optim.Adam(self.netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))#, weight_decay=args.wd)
        self.optD = optim.Adam(self.netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))#, weight_decay=args.wd)
        self.optF = optim.Adam(self.netF.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))#, weight_decay=args.wd)
        self.scaler = amp.GradScaler(enabled=not args.no_amp)

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + args.init_epoch - args.n_epochs) / float(args.n_epochs_decay + 1)
            return lr_l
        self.schedulers = [lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule) for opt in [self.optG, self.optD, self.optF]]


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
            pred, feats = self.netG(real, get_feat=True)
            batch_size = real_A.size(0)
            fake_B = pred[:batch_size]
            idt_B = pred[batch_size:]

            fake_out = self.netD(fake_B)#.detach())
            real_out = self.netD(real_B)

            lossD = (self.criterion_gan(fake_out, False)
                     + self.criterion_gan(real_out, True)) * 0.5

            # self.scaler.scale(lossD).backward()
            set_grads(autograd.grad(self.scaler.scale(lossD), self.netD.parameters(), retain_graph=True), self.netD.parameters())
            self.scaler.step(self.optD)
            self.optD.zero_grad(set_to_none=True)

            # fake_out = self.netD(fake_B)
            lossG = self.criterion_gan(fake_out, True) * args.lambda_GAN

            feat_q = self.netG(fake_B, get_feat=True, encode_only=True)
            feat_k = [ft[:batch_size] for ft in feats]
            nce_loss_A = self.calculate_NCE_loss(args, feat_k, feat_q)

            feat_q = self.netG(idt_B, get_feat=True, encode_only=True)
            feat_k = [ft[batch_size:] for ft in feats]
            nce_loss_B = self.calculate_NCE_loss(args, feat_k, feat_q)

            nce_loss_tot = (nce_loss_A + nce_loss_B) * 0.5
            lossG = lossG + nce_loss_tot
        
        GF_params = list(self.netG.parameters()) + list(self.netF.parameters())
        set_grads(autograd.grad(self.scaler.scale(lossG), GF_params), GF_params)

        # self.scaler.scale(lossG).backward()

        self.scaler.step(self.optG)
        self.optG.zero_grad(set_to_none=True)
        self.scaler.step(self.optF)
        self.optF.zero_grad(set_to_none=True)

        self.scaler.update()

        self.loss_avg['lossG'].update(reduce_loss(lossG.detach()), batch_size)
        self.loss_avg['lossD'].update(reduce_loss(lossD.detach()), batch_size)
        self.loss_avg['nce_loss_tot'].update(reduce_loss(nce_loss_tot.detach()), batch_size)

        return fake_B.detach(), idt_B.detach()

    def train_epoch(self, args, epoch):
        self.loss_avg = {nm: AverageMeter() for nm in self.loss_names}
        info = {}
        with tqdm(self.dataloader, desc=f"Epoch {epoch:>2}", disable=args.local_rank != 0) as pbar:
            for step, (real_A, real_B) in enumerate(pbar):
                real_A = real_A.to(self.device, non_blocking=True)
                real_B = real_B.to(self.device, non_blocking=True)
                fake_B, idt_B = self.forward(args, real_A, real_B)

                if args.local_rank == 0:
                    if not step % args.log_interval:
                        info = {nm: float(loss.avg) for nm, loss in self.loss_avg.items()}
                        pbar.set_postfix(info)
                        if args.use_wandb:
                            wandb.log(info)
                            if not step % args.img_log_interval:
                                log_imgs_wandb(real_A=real_A, fake_B=fake_B, real_B=real_B, idt_B=idt_B)
                
        for schd in self.schedulers:
            schd.step()
        return info

    def train_loop(self, args):
        # self.validate(args)
        for epoch in range(args.init_epoch, args.n_epochs):
            self.netG.train()
            self.netD.train()
            self.netF.train()
            self.dataloader.sampler.set_epoch(epoch)
            info = self.train_epoch(args, epoch)
            info['epoch'] = epoch
            if args.local_rank == 0:
                if args.use_wandb:
                    wandb.log({'epoch': epoch})
            self.save_models(args, 'latest', info)
            if not epoch % 1:
                self.save_models(args, epoch, info)
                # self.validate(args)

    def save_models(self, args, epoch='latest', info={}):
        if args.local_rank == 0:
            os.makedirs(os.path.join(args.checkpoints_dir, args.name), exist_ok=True)
            torch.save(self.netG.state_dict(), os.path.join(args.checkpoints_dir, args.name, f"unetG_{epoch}.pth"))
            torch.save(self.optG.state_dict(), os.path.join(args.checkpoints_dir, args.name, f"optG_{epoch}.pth"))
            torch.save(info, os.path.join(args.checkpoints_dir, args.name, f"info_{epoch}.pth"))
            print("[+] Weights saved.")

    def load_models(self, args, epoch='latest'):
        synchronize()
        map_location = {'cuda:0': f'cuda:{args.local_rank}'}
        try:
            self.netG.load_state_dict(torch.load(os.path.join(
                args.checkpoints_dir, args.name, f"unetG_{epoch}.pth"), map_location=map_location))
            if args.phase == 'train':
                self.optG.load_state_dict(torch.load(os.path.join(
                    args.checkpoints_dir, args.name, f"optG_{epoch}.pth"), map_location=map_location))
            if args.local_rank == 0:
                print(f"[+] Weights loaded for {epoch} epoch.")
        except FileNotFoundError as e:
            if args.local_rank == 0:
                print(f"[!] {e}, skipping weights loading.")


def main():
    args = TrainOptions().parse()
    torch.cuda.set_device(args.local_rank)
    seed_everything(args.seed)
    try:
        tm = TrainModel(args)
        if args.resume:
            tm.load_models(args)
        tm.train_loop(args)
        tm.save_models(args)
    except KeyboardInterrupt:
        print("[!] Keyboard Interrupt! Cleaning up and shutting down.")
    finally:
        cleanup()


if __name__ == '__main__':
    main()
