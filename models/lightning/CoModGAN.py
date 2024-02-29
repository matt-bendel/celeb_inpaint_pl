import os

import torch

import pytorch_lightning as pl
import numpy as np
import torch.autograd as autograd
from torchvision.models.inception import inception_v3
from utils.embeddings import WrapInception
from PIL import Image
from torch.nn import functional as F
from models.archs.inpainting.co_mod_gan import Generator, Discriminator
from torchmetrics.functional import peak_signal_noise_ratio

class CoModGAN(pl.LightningModule):
    def __init__(self, args, exp_name, num_gpus):
        super().__init__()
        self.args = args
        self.exp_name = exp_name
        self.num_gpus = num_gpus

        self.in_chans = args.in_chans
        self.out_chans = args.out_chans

        self.generator = Generator(self.args.im_size)
        self.discriminator = Discriminator(self.args.im_size)

        self.feature_extractor = inception_v3(pretrained=True, transform_input=False)
        self.feature_extractor = WrapInception(self.feature_extractor.eval()).eval()

        self.resolution = self.args.im_size

        self.save_hyperparameters()  # Save passed values

    def get_noise(self, num_vectors):
        z = [torch.randn(num_vectors, 512, device=self.device)]
        return z

    def compute_gradient_penalty(self, real_samples, fake_samples, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        Tensor = torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(input=interpolates, label=y)
        # fake = Tensor(real_samples.shape[0], 1, d_interpolates.shape[-1], d_interpolates.shape[-1]).fill_(1.0).to(
        #     self.device)
        fake = Tensor(real_samples.shape[0], 1).fill_(1.0).to(
            self.device)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, y, mask):
        noise = self.get_noise(y.size(0))
        return self.generator(y, mask, noise)

    def adversarial_loss_discriminator(self, fake_pred, real_pred):
        return fake_pred.mean() - real_pred.mean()

    def adversarial_loss_generator(self, fake_pred):
        return - fake_pred.mean()

    def l1_std_p(self, avg_recon, gens, x):
        return F.l1_loss(avg_recon, x) - self.std_mult * np.sqrt(
            2 / (np.pi * self.args.num_z_train * (self.args.num_z_train+ 1))) * torch.std(gens, dim=1).mean()

    def gradient_penalty(self, x_hat, x, y):
        gradient_penalty = self.compute_gradient_penalty(x.data, x_hat.data, y.data)

        return self.args.gp_weight * gradient_penalty

    def get_embed_im(self, inp, mean, std):
        embed_ims = torch.zeros(size=(inp.size(0), 3, 256, 256),
                                device=self.device)
        for i in range(inp.size(0)):
            im = inp[i, :, :, :] * std[i, :, None, None] + mean[i, :, None, None]
            im = 2 * (im - torch.min(im)) / (torch.max(im) - torch.min(im)) - 1
            embed_ims[i, :, :, :] = im

        return self.feature_extractor(embed_ims)

    def drift_penalty(self, real_pred):
        return 0.001 * torch.mean(real_pred ** 2)

    def training_step(self, batch, batch_idx, optimizer_idx):
        y, x, mask, mean, std = batch[0]

        # train generator
        if optimizer_idx == 1:
            x_hat = self.forward(y, mask)
            fake_pred = self.discriminator(input=x_hat, label=y)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss_generator(fake_pred)

            self.log('g_loss', g_loss, prog_bar=True)

            return g_loss

        # train discriminator
        if optimizer_idx == 0:
            x_hat = self.forward(y, mask)

            real_pred = self.discriminator(input=x, label=y)
            fake_pred = self.discriminator(input=x_hat, label=y)

            d_loss = self.adversarial_loss_discriminator(fake_pred, real_pred)
            d_loss += self.gradient_penalty(x_hat, x, y)
            d_loss += self.drift_penalty(real_pred)

            self.log('d_loss', d_loss, prog_bar=True)

            return d_loss

    def validation_step(self, batch, batch_idx, external_test=False):
        y, x, mask, mean, std = batch[0]

        fig_count = 0

        if external_test:
            num_code = self.args.num_z_test
        else:
            num_code = self.args.num_z_valid

        gens = torch.zeros(size=(y.size(0), num_code, 3, self.args.im_size, self.args.im_size),
                           device=self.device)
        for z in range(num_code):
            gens[:, z, :, :, :] = self.forward(y, mask) * std[:, :, None, None] + mean[:, :, None, None]

        avg_gen = torch.mean(gens, dim=1)
        single_gen = gens[:, 0, :, :, :]
        gt = x * std[:, :, None, None] + mean[:, :, None, None]

        psnr_8s = []
        psnr_1s = []

        for j in range(y.size(0)):
            psnr_8s.append(peak_signal_noise_ratio(avg_gen[j], gt[j]))
            psnr_1s.append(peak_signal_noise_ratio(single_gen[j], gt[j]))

        psnr_8s = torch.stack(psnr_8s)
        psnr_1s = torch.stack(psnr_1s)

        self.log('psnr_8_step', psnr_8s.mean(), on_step=True, on_epoch=False, prog_bar=True)
        self.log('psnr_1_step', psnr_1s.mean(), on_step=True, on_epoch=False, prog_bar=True)

        img_e = self.get_embed_im(gens[:, 0, :, :, :], mean, std)
        cond_e = self.get_embed_im(y, mean, std)
        true_e = self.get_embed_im(x, mean, std)

        if batch_idx == 0:
            if self.global_rank == 0 and self.current_epoch % 5 == 0 and fig_count == 0:
                fig_count += 1
                samp_1_np = gens[0, 0, :, :, :].cpu().numpy()
                samp_2_np = gens[0, 1, :, :, :].cpu().numpy()
                samp_3_np = gens[0, 2, :, :, :].cpu().numpy()
                gt_np = gt[0].cpu().numpy()
                y_np = (y * std[:, :, None, None] + mean[:, :, None, None])[0].cpu().numpy()

                plot_gt_np = gt_np

                self.logger.log_image(
                    key=f"epoch_{self.current_epoch}_img",
                    images=[
                        Image.fromarray(np.uint8(np.transpose(plot_gt_np, (1, 2, 0))*255), 'RGB'),
                        Image.fromarray(np.uint8(np.transpose(y_np, (1, 2, 0)) * 255), 'RGB'),
                        Image.fromarray(np.uint8(np.transpose(samp_1_np, (1, 2, 0))*255), 'RGB'),
                        Image.fromarray(np.uint8(np.transpose(samp_2_np, (1, 2, 0)) * 255), 'RGB'),
                        Image.fromarray(np.uint8(np.transpose(samp_3_np, (1, 2, 0)) * 255), 'RGB')
                    ],
                    caption=["x", "y", "Samp 1", "Samp 2", "Samp 3"]
                )

            self.trainer.strategy.barrier()

        return {'psnr_8': psnr_8s.mean(), 'psnr_1': psnr_1s.mean(), 'img_e': img_e, 'cond_e': cond_e, 'true_e': true_e}

    def validation_epoch_end(self, validation_step_outputs):
        avg_psnr = self.all_gather(torch.stack([x['psnr_8'] for x in validation_step_outputs]).mean()).mean()
        avg_single_psnr = self.all_gather(torch.stack([x['psnr_1'] for x in validation_step_outputs]).mean()).mean()

        true_embed = torch.cat([x['true_e'] for x in validation_step_outputs], dim=0)
        image_embed = torch.cat([x['img_e'] for x in validation_step_outputs], dim=0)
        cond_embed = torch.cat([x['cond_e'] for x in validation_step_outputs], dim=0)

        cfid = self.cfid.get_cfid_torch_pinv(y_predict=image_embed, x_true=cond_embed, y_true=true_embed)
        cfid = self.all_gather(cfid).mean()

        self.log('cfid', cfid, prog_bar=True)
        self.trainer.strategy.barrier()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        return [opt_d, opt_g], []
