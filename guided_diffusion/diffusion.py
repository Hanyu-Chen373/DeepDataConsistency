import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data


from functions.ckpt_util import get_ckpt_path, download
from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion

import torchvision.utils as tvu

from guided_diffusion.datasets import get_dataset, data_transform, inverse_data_transform
from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth

import lpips
from skimage.metrics import structural_similarity
from diffusers import UNet2DModel


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)

            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',
                             ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                        self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size,
                        ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale

                cls_fn = cond_fn

        self.sample_loop(model, cls_fn)
            
            
    def sample_loop(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
        )

        # get degradation operator
        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        print("args.deg:",args.deg)        
        def A_DDC(img, scale=4, deg='sr_averagepooling'):
            if deg == 'sr_averagepooling':
                A_scale = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
                return A_scale(img)
            elif deg =='gaussian_blur':
                from util.img_utils import Blurkernel
                conv = Blurkernel(blur_type='gaussian', kernel_size=61, std=3.0,device='cuda').to('cuda')
                kernel = conv.get_kernel()
                conv.update_weights(kernel.type(torch.float32))
                return conv(img)
            elif deg == 'sr_bicubic':
                factor = int(scale)
                from functions.svd_operators import SRConv
                def bicubic_kernel(x, a=-0.5):
                    if abs(x) <= 1:
                        return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                    elif 1 < abs(x) and abs(x) < 2:
                        return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                    else:
                        return 0
                k = np.zeros((factor * 4))
                for i in range(factor * 4):
                    x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                    k[i] = bicubic_kernel(x)
                k = k / np.sum(k)
                kernel = torch.from_numpy(k).float().to(self.device)
                A_funcs = SRConv(kernel / kernel.sum(), \
                                config.data.channels, self.config.data.image_size, self.device, stride=factor)
                y = A_funcs.A(img)
                b, hwc = y.size()
                hw = hwc / 3
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 3, h, w))
                return y
            elif deg == 'inpaint_random':
                def _retrieve_random(img, mask_prob):
                    image_size = img.shape[-1]
                    total = image_size ** 2
                    mask_vec = torch.ones([1, image_size * image_size])
                    samples = np.random.choice(image_size * image_size, int(total * mask_prob), replace=False)
                    mask_vec[:, samples] = 0
                    mask_b = mask_vec.view(1, image_size, image_size)
                    mask_b = mask_b.repeat(3, 1, 1)
                    mask = torch.ones_like(img, device=img.device)
                    mask[:, ...] = mask_b
                    return mask
                mask = _retrieve_random(img, scale)
                y = img * mask
                return y
            elif deg == "jpeg":
                from util.jpeg_torch import jpeg_encode, jpeg_decode
                y = jpeg_decode(jpeg_encode(img, scale), scale)
                return y
                
        DDC_model = UNet2DModel.from_pretrained(os.path.join(self.args.exp, "logs/ddc_network")).to(self.device)
        DDC_model.eval()

        loss_fn = lpips.LPIPS(net='alex')
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)
            y_DDC = A_DDC(x_orig, scale=args.deg_scale, deg=args.deg)
            
            y_DDC = y_DDC + args.sigma_y * torch.randn_like(y_DDC).to(self.device)
            if y_DDC.shape[-1] != 256:
                A_back_to_original = torch.nn.AdaptiveAvgPool2d((256,256))
                y_DDC = A_back_to_original(y_DDC)


            if config.sampling.batch_size!=1:
                raise ValueError("please change the config file to set batch size as 1")

            os.makedirs(os.path.join(self.args.image_folder, "GT"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "results"), exist_ok=True)
            os.makedirs(os.path.join(self.args.image_folder, "measurement"), exist_ok=True)
            for i in range(len(x_orig)):
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"GT/{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, y_DDC[i]),
                    os.path.join(self.args.image_folder, f"measurement/{idx_so_far + i}.png")
                )
                
            # init x_T
            x = torch.randn(
                x_orig.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                xs = [x]

                
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                               config.time_travel.travel_length, 
                                               config.time_travel.travel_repeat,
                                              )
                time_pairs = list(zip(times[:-1], times[1:]))
                
                # reverse diffusion sampling
                for i, j in tqdm.tqdm(time_pairs):
                    i, j = i*skip, j*skip
                    if j<0: j=-1 

                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    new_beta = 1 - at/at_next
                    new_beta_tilde = ((1 - at_next) / (1 - at)) * new_beta

                    xt = xs[-1].to('cuda')

                    et = model(xt, t)

                    if et.size(1) == 6:
                        et, sigma_theta_t  = torch.split(et, 3, dim=1)
                        frac = (sigma_theta_t + 1.0) / 2.0
                        model_log_variance = frac * torch.log(new_beta) + (1-frac) * torch.log(new_beta_tilde)
                    else:
                        model_log_variance = None

                    
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()


                    xt_next_mean = (xt - (new_beta * et) / (1 - at).sqrt()) / (1 - new_beta).sqrt()
                    DDC_input = torch.cat([x0_t, y_DDC], dim=1)
                    DDC_predict = DDC_model(DDC_input, t).sample
                    xt_next_mean = xt_next_mean - 1.0 * ((at_next.sqrt() * new_beta) / (1 - at)) * DDC_predict



                    if model_log_variance is not None:
                        xt_next = xt_next_mean + torch.randn_like(xt) * torch.exp(0.5 * model_log_variance)
                    else:
                        xt_next = xt_next_mean + torch.randn_like(xt) * new_beta_tilde.sqrt()
                    xs.append(xt_next.to('cpu'))


                x = xs[-1]
            x = [inverse_data_transform(config, xi) for xi in x]

            tvu.save_image(
                x[0], os.path.join(self.args.image_folder, f"results/{idx_so_far}.png")
            )
            orig = inverse_data_transform(config, x_orig[0])
            mse = torch.mean((x[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr
            
            ssim = structural_similarity(orig.cpu().numpy().transpose(1,2,0), x[0].cpu().numpy().transpose(1,2,0), multichannel=True)
            avg_ssim += ssim
            with torch.no_grad():
                img0 = lpips.im2tensor(lpips.load_image(os.path.join(self.args.image_folder, f"results/{idx_so_far}.png")))
                img1 = lpips.im2tensor(lpips.load_image(os.path.join(self.args.image_folder, f"GT/{idx_so_far}.png")))
                lpips_score = loss_fn(img0, img1)
                avg_lpips += lpips_score.item()

            idx_so_far += x_orig.shape[0]

            pbar.set_description("PSNR: %(psnr).2f, SSIM: %(ssim).4f, LPIPS: %(lpips).4f" % {'psnr':(avg_psnr / (idx_so_far - idx_init)),
                                                                                              'ssim':avg_ssim / (idx_so_far - idx_init),
                                                                                              'lpips':avg_lpips / (idx_so_far - idx_init)})

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        avg_ssim = avg_ssim / (idx_so_far - idx_init)
        avg_lpips = avg_lpips / (idx_so_far - idx_init)
        print("Average PSNR: %.2f" % avg_psnr)
        print("Average SSIM: %.4f" % avg_ssim)
        print("Average LPIPS: %.4f" % avg_lpips)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        

# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
