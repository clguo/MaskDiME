import os
import itertools
import numpy as np
# Replace brute-force x_t update with DPM-Solver++ in sampling
from PIL import Image
from tqdm import tqdm
from scipy import linalg
from os import path as osp


import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19
from torchvision.transforms.functional import gaussian_blur
from .adam_stabilization import ADAMGradientStabilization


from .gaussian_diffusion import _extract_into_tensor


# =======================================================
# Functions
# =======================================================


def load_from_DDP_model(state_dict):

    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


# =======================================================
# Gradient Extraction Functions
# =======================================================


# @torch.enable_grad()
# def clean_class_cond_fn(x_t, y, classifier,
#                         s, use_logits,dataset,data_dir):
#     '''
#     Computes the classifier gradients for the guidance
#
#     :param x_t: clean instance
#     :param y: target
#     :param classifier: classification model
#     :param s: scaling classifier gradients parameter
#     :param use_logits: compute the loss over the logits
#     '''
#
#     x_in = x_t.detach().requires_grad_(True)
#     logits = classifier(x_in)
#     if dataset == 'ImageNet' and "cat" in data_dir:
#         logits = logits[:, [283, 285]]
#
#
#     y = y.to(logits.device).float()
#     # Select the target logits,
#     # for those of target 1, we take the logits as they are (sigmoid(logits) = p(y=1 | x))
#     # for those of target 0, we take the negative of the logits (sigmoid(-logits) = p(y=0 | x))
#     selected = y * logits - (1 - y) * logits
#     if use_logits:
#         selected = -selected
#     else:
#         selected = -F.logsigmoid(selected)
#
#     selected = selected * s
#     grads = torch.autograd.grad(selected.sum(), x_in)[0]
#
#     return grads


@torch.enable_grad()
def clean_class_cond_TG(x_t, z_t, y, classifier,
                        s, use_logits, dataset, data_dir):
    """
    Returns:
      grads_z: TRUE d(class_loss)/d(z_t)
      grads_x: d(class_loss)/d(x_t) (for mask construction)
    """
    logits = classifier(x_t)  # ❗ do NOT detach x_t

    # ===== your original selected logic unchanged =====
    if dataset == 'ImageNet' and "cat" in data_dir:
        logits = logits[:, [283, 285]]
        logits_binary = logits[:, 1] - logits[:, 0]
        y = y.to(logits.device).float()
        selected = y * logits_binary - (1 - y) * logits_binary
        selected = -selected if use_logits else -F.logsigmoid(selected)

    elif dataset == 'ImageNet' and "cc" in data_dir:
        logits = logits[:, [286, 293]]
        logits_binary = logits[:, 1] - logits[:, 0]
        y = y.to(logits.device).float()
        selected = y * logits_binary - (1 - y) * logits_binary
        selected = -selected if use_logits else -F.logsigmoid(selected)

    elif dataset == 'ImageNet' and ("sorrel_zebra" in data_dir or "mini_sz" in data_dir):
        logits = logits[:, [339, 340]]
        logits_binary = logits[:, 1] - logits[:, 0]
        y = y.to(logits.device).float()
        selected = y * logits_binary - (1 - y) * logits_binary
        selected = -selected if use_logits else -F.logsigmoid(selected)

    else:
        y = y.to(logits.device).float()
        selected = y * logits - (1 - y) * logits
        selected = selected.squeeze(1) if selected.ndim > 1 else selected
        selected = -selected if use_logits else -F.logsigmoid(selected)

    loss = (selected * s).sum()

    grads_z, grads_x = torch.autograd.grad(
        loss, [z_t, x_t],
        retain_graph=True,
        create_graph=False,
        allow_unused=False
    )
    return grads_z, grads_x

@torch.enable_grad()
def dist_cond_TG(x_tau, z_t, x_t,
                 l1_loss, l2_loss,
                 l_perc):
    """
    TRUE distance gradient wrt z_t (one-step graph).
    """
    loss = 0.0

    if l1_loss != 0:
        loss = loss + l1_loss * torch.norm(z_t - x_tau, p=1, dim=1).sum()
    if l2_loss != 0:
        loss = loss + l2_loss * torch.norm(z_t - x_tau, p=2, dim=1).sum()
    if l_perc is not None:
        loss = loss + l_perc(x_t, x_tau)

    if isinstance(loss, float):
        return 0

    grads_z = torch.autograd.grad(
        loss, z_t,
        retain_graph=True,
        create_graph=False,
        allow_unused=False
    )[0]
    return grads_z

@torch.enable_grad()
def clean_class_cond_fn(x_t, y, classifier,
                        s, use_logits, dataset, data_dir):
    '''
    Computes the classifier gradients for the guidance

    :param x_t: clean instance
    :param y: target (0 or 1)
    :param classifier: classification model
    :param s: scaling classifier gradients parameter
    :param use_logits: compute the loss over the logits
    '''
    x_in = x_t.detach().requires_grad_(True)
    logits = classifier(x_in)

    if dataset == 'ImageNet' and "cat" in data_dir:

        # Only keep logits for class 283 and 285
        logits = logits[:, [283, 285]]  # shape: [B, 2]

        # Construct binary logits: logit_285 - logit_283
        logits_binary = logits[:, 1] - logits[:, 0]  # shape: [B]

        y = y.to(logits.device).float()
        selected = y * logits_binary - (1 - y) * logits_binary  # = (2y - 1) * logits_binary

        if use_logits:
            selected = -selected
        else:
            selected = -F.logsigmoid(selected)
    elif dataset == 'ImageNet' and "cc" in data_dir:

        # Only keep logits for class 286 and 293
        logits = logits[:, [286, 293]]  # shape: [B, 2]


        # Construct binary logits: logit_286 - logit_293
        logits_binary = logits[:, 1] - logits[:, 0]  # shape: [B]

        y = y.to(logits.device).float()
        selected = y * logits_binary - (1 - y) * logits_binary  # = (2y - 1) * logits_binary

        if use_logits:
            selected = -selected
        else:
            selected = -F.logsigmoid(selected)
    elif dataset == 'ImageNet' and ("sorrel_zebra" in data_dir or "mini_sz" in data_dir):


        # Only keep logits for class 283 and 285
        logits = logits[:, [339, 340]]  # shape: [B, 2]

        # Construct binary logits: logit_285 - logit_283
        logits_binary = logits[:, 1] - logits[:, 0]  # shape: [B]

        y = y.to(logits.device).float()
        selected = y * logits_binary - (1 - y) * logits_binary  # = (2y - 1) * logits_binary

        if use_logits:
            selected = -selected
        else:
            selected = -F.logsigmoid(selected)

    else:
        # Original binary classifier case
        y = y.to(logits.device).float()
        selected = y * logits - (1 - y) * logits
        selected = selected.squeeze(1) if selected.ndim > 1 else selected

        if use_logits:
            selected = -selected
        else:
            selected = -F.logsigmoid(selected)

    selected = selected * s
    grads = torch.autograd.grad(selected.sum(), x_in)[0]

    return grads



@torch.enable_grad()
def clean_multiclass_cond_fn(x_t, y, classifier,
                             s, use_logits):
    
    x_in = x_t.detach().requires_grad_(True)
    selected = classifier(x_in)

    # Select the target logits
    if not use_logits:
        selected = F.log_softmax(selected, dim=1)
    selected = -selected[range(len(y)), y]
    selected = selected * s
    grads = torch.autograd.grad(selected.sum(), x_in)[0]

    return grads


@torch.enable_grad()
def dist_cond_fn(x_tau, z_t, x_t, alpha_t,
                 l1_loss, l2_loss,
                 l_perc):

    '''
    Computes the distance loss between x_t, z_t and x_tau
    :x_tau: initial image
    :z_t: current noisy instance
    :x_t: current clean instance
    :alpha_t: time dependant constant
    '''

    z_in = z_t.detach().requires_grad_(True)
    x_in = x_t.detach().requires_grad_(True)

    m1 = l1_loss * torch.norm(z_in - x_tau, p=1, dim=1).sum() if l1_loss != 0 else 0
    m2 = l2_loss * torch.norm(z_in - x_tau, p=2, dim=1).sum() if l2_loss != 0 else 0
    mv = l_perc(x_in, x_tau) if l_perc is not None else 0
    
    if isinstance(m1 + m2 + mv, int):
        return 0

    if isinstance(m1 + m2, int):
        grads = 0
    else:
        grads = torch.autograd.grad(m1 + m2, z_in)[0]

    if isinstance(mv, int):
        return grads
    else:
        return grads + torch.autograd.grad(mv, x_in)[0] / alpha_t


import torch
import torch.nn.functional as F


def get_angle_constrained_sampler(use_sampling=False):
    """带角度约束的反事实生成器"""

    @torch.no_grad()
    def p_sample_loop(
            dpm_scheduler,
            diffusion,
            model,
            shape,  # 图像形状 (B,C,H,W)
            num_timesteps,  # 总时间步
            img,  # 原始图像
            t,  # 初始时间步
            z_t=None,  # 可选初始噪声
            clip_denoised=True,  # 是否截断到[-1,1]
            model_kwargs=None,
            device=None,
            # 梯度函数配置
            class_grad_fn=None,  # 分类器梯度函数
            class_grad_kwargs=None,
            dist_grad_fn=None,  # 距离梯度函数
            dist_grad_kargs=None,
            # 采样参数
            x_t_sampling=True,
            is_x_t_sampling=False,
            guided_iterations=9999999,  # 梯度引导的最大迭代次数
            # 超参数
            dpm_step=50,  # DPM求解器步数
            grad_scale=0.01,  # 梯度缩放系数
            power_scale=2,  # 未使用（保留参数）
            topk_ratio=0.1,  # 未使用（保留参数）
            min_grad_weight=0.2,  # 最小梯度权重(0.1-0.4)
            transition_slope=10,  # 权重过渡斜率(5-20)
            angle_threshold=0.5  # 角度约束阈值(cosθ=0.5对应60度)
    ):
        # 初始化变量
        x_t = img.clone()
        batch_size = shape[0]
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []  # 存储中间结果用于可视化
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]  # 倒序时间步

        for jdx, i in enumerate(indices):
            # === 时间步处理 ===
            t = torch.tensor([i] * batch_size, device=device)
            # 配置DPM求解器的时间表
            dpm_scheduler.set_timesteps(dpm_step)
            timesteps_sorted = torch.sort(dpm_scheduler.timesteps)[0]
            t = torch.tensor([timesteps_sorted[i] for i in t.tolist()], device=device)

            # 记录中间结果
            x_t_steps.append(x_t.detach().cpu())
            z_t_steps.append(z_t.detach().cpu())

            # === 去噪过程 ===
            model_pred = model(z_t, t, **(model_kwargs or {}))[:, :3]  # 预测噪声
            t_scalar = t[0].item() if isinstance(t, torch.Tensor) else t
            # DPM求解器单步更新
            out = dpm_scheduler.step(
                model_output=model_pred,
                timestep=t_scalar,
                sample=z_t
            )

            # === 梯度计算 ===
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, z_t.shape)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(z_t.shape) - 1)))

            # 初始化梯度
            class_grad = torch.zeros_like(z_t)
            dist_grad = torch.zeros_like(z_t)

            # 计算分类器梯度
            if (class_grad_fn is not None) and (guided_iterations > jdx):
                class_grad = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t

            # 计算距离梯度
            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                dist_grad = dist_grad_fn(
                    z_t=z_t,
                    x_tau=img,
                    x_t=x_t,
                    alpha_t=alpha_t,
                    **dist_grad_kargs
                )

            # === 动态梯度加权 ===
            if class_grad_fn and dist_grad_fn:
                # 展平梯度以计算余弦相似度
                flat_class = class_grad.flatten(start_dim=1)  # [B, D]
                flat_dist = dist_grad.flatten(start_dim=1)  # [B, D]

                # 计算余弦相似度
                dot_product = torch.sum(flat_class * flat_dist, dim=1)  # [B]
                norm_class = torch.norm(flat_class, dim=1, keepdim=True)  # [B,1]
                norm_dist = torch.norm(flat_dist, dim=1, keepdim=True)  # [B,1]
                cosine_sim = dot_product / (norm_class * norm_dist + 1e-8)  # [B]

                # === 角度约束 ===
                # 生成二值掩码（夹角>60度时激活）
                angle_mask = (cosine_sim < angle_threshold).float()  # [B]
                angle_mask = angle_mask.view(-1, 1, 1, 1)  # [B,1,1,1]

                # 动态权重计算
                transition = torch.sigmoid(transition_slope * (1 - cosine_sim))  # [B]
                # 权重计算公式（保证权重和为1）
                class_weight = min_grad_weight + (1 - 2 * min_grad_weight) * transition  # [B]
                dist_weight = min_grad_weight + (1 - 2 * min_grad_weight) * (1 - transition)  # [B]

                # 应用角度约束
                class_weight = class_weight * angle_mask + (1 - angle_mask) * 0.5  # 不满足时固定为0.5
                dist_weight = dist_weight * angle_mask + (1 - angle_mask) * 0.5  # 但不参与实际梯度计算

                # 维度对齐
                class_weight = class_weight.view(-1, 1, 1, 1)  # [B,1,1,1]
                dist_weight = dist_weight.view(-1, 1, 1, 1)  # [B,1,1,1]

                # 组合梯度并应用角度约束
                grads = (class_weight * class_grad + dist_weight * dist_grad) * angle_mask  # 关键约束
            else:
                grads = class_grad + dist_grad  # 退化情况

            # === 梯度应用 ===
            if clip_denoised:
                out["prev_sample"] = torch.clamp(out["prev_sample"], -1.0, 1.0)
            # 梯度更新步骤
            out["prev_sample"] = out["prev_sample"].float() - grad_scale * grads
            z_t = out["prev_sample"]

            # === 递归生成x_t ===
            if (num_timesteps - (jdx + 1) > 0) and (class_grad_fn is not None) and \
                    (dist_grad_fn is not None) and (guided_iterations > jdx):
                x_t = p_sample_loop(
                    dpm_scheduler=dpm_scheduler,
                    diffusion=diffusion,
                    model=model,
                    model_kwargs=model_kwargs,
                    shape=shape,
                    num_timesteps=num_timesteps - (jdx + 1),
                    img=img,
                    t=None,
                    z_t=z_t,
                    clip_denoised=clip_denoised,
                    device=device,
                    class_grad_fn=class_grad_fn,
                    class_grad_kwargs=class_grad_kwargs,
                    dist_grad_fn=dist_grad_fn,
                    dist_grad_kargs=dist_grad_kargs,
                    x_t_sampling=use_sampling,
                    is_x_t_sampling=True,
                    dpm_step=dpm_step,
                    min_grad_weight=min_grad_weight,
                    transition_slope=transition_slope,
                    angle_threshold=angle_threshold  # 重要！递归传递角度约束参数
                )[0]

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop

def get_TRUEGRAD_sampling_CelebA(use_sampling=False):

    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=use_sampling,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      grads_clip=True,
                      x_t_rate=1,
                      clip_sigma=3,
                      blur_kernel=5,
                      blur_sigma=3,
                      per_channel=True,
                      ):

        x_t = img.detach().clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]
        grad_mask_x_t = None
        grad_mask = None

        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # ---------- default: cheap no-grad forward ----------
            with torch.no_grad():
                out_ng = diffusion.p_mean_variance(
                    model, z_t, t_val,
                    clip_denoised=clip_denoised,
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                )
            out = out_ng
            pred_xstart = out_ng["pred_xstart"].detach()
            mean = out_ng["mean"]
            var = out_ng["variance"]

            grads = None

            # =========================================================
            # Guidance branch: compute TRUE grads wrt z_t (one-step)
            # =========================================================
            if (class_grad_fn is not None) and (jdx < guided_iterations):
                with torch.enable_grad():
                    # z_in must require grad to get TRUE ∇_{z_t}
                    z_in = z_t.detach().clone().requires_grad_(True)

                    out = diffusion.p_mean_variance(
                        model, z_in, t_val,
                        clip_denoised=clip_denoised,
                        denoised_fn=None,
                        model_kwargs=model_kwargs,
                    )

                    # IMPORTANT: do NOT detach pred_xstart
                    pred_xstart = out["pred_xstart"]

                    # masked clean used for loss
                    if grad_mask_x_t is not None:
                        x_for_loss = grad_mask_x_t * pred_xstart + (1 - grad_mask_x_t) * img.detach().clone()
                    else:
                        x_for_loss = pred_xstart

                    # class_grad_fn now returns (grads_z_class, grads_x_class)
                    grads_z_class, grads_x_class = class_grad_fn(
                        x_t=x_for_loss, z_t=z_in, **class_grad_kwargs
                    )

                    # masks built from grads_x_class (same as your method)
                    grad_mask, grad_mask_x_t = compute_grad_masks_nested2(
                        grads_x_class, topk_ratio1=topk_ratio,
                        topk_ratio2=x_t_rate, blur_kernel=blur_kernel,
                        blur_sigma=blur_sigma, per_channel=per_channel
                    )

                    # optional clip on the gradient used to update z
                    if grads_clip is True:
                        m = grads_z_class.mean()
                        s = grads_z_class.std()
                        grads_z_class = grads_z_class.clamp(m - clip_sigma*s, m + clip_sigma*s)

                    # dist_grad_fn should now return grads wrt z_in (TRUE)
                    if dist_grad_fn is not None:
                        grads_z_dist = dist_grad_fn(
                            x_tau=img, z_t=z_in, x_t=x_for_loss, **dist_grad_kargs
                        )
                        grads = grad_scale * (grads_z_class + grads_z_dist)
                    else:
                        grads = grad_scale * grads_z_class

                    # use mean/var from the grad-enabled out, but detach values for update
                    mean = out["mean"].detach()
                    var = out["variance"].detach()

                    # update x_t state for next step (no need grad graph)
                    pred_det = pred_xstart.detach()
                    if grad_mask_x_t is not None:
                        x_t = grad_mask_x_t * pred_det + (1 - grad_mask_x_t) * img.detach().clone()
                    else:
                        x_t = x_t  # keep previous if you prefer

            # ---------- Apply gradient guidance + noisy mask ----------
            if grads is not None and torch.abs(grads).sum() > 0:
                mean = mean - var * grads.detach()
                z_original = diffusion.q_sample(img, t_val)

                if grad_mask is None:
                    grad_mask = torch.ones_like(mean)

                z_t = mean * grad_mask + z_original * (1 - grad_mask)

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop


def get_DiME_iterative_sampling(use_sampling=False):
    '''
    Returns DiME's main algorithm to construct counterfactuals.
    The returned function computes x_t in a recursive way.
    Easy way to set the optional parameters into the sampling
    function such as the use_sampling flag.

    :param use_sampling: use mu + sigma * N(0,1) when computing
     the next iteration when estimating x_t
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      guided_iterations=9999999,
                      x_t_sampling=False,
                      is_x_t_sampling=False,
                      grad_scale=10,
                      topk_ratio=0.1,
                      grads_clip=True,
                      x_t_rate=0.15,
                      sigma=3,
                      blur_kernel=7,
                      blur_sigma=3,):

        '''
        :param :
        :param diffusion: diffusion algorithm
        :param model: DDPM model
        :param num_timesteps: tau, or the depth of the noise chain
        :param img: instance to be explained
        :param t: time variable
        :param z_t: noisy instance. If z_t is instantiated then the model
                    will denoise z_t
        :param clip_denoised: clip the noised data to [-1, 1]
        :param model_kwargs: useful when the model is conditioned
        :param device: torch device
        :param class_grad_fn: class function to compute the gradients of the classifier
                              has at least an input, x_t.
        :param class_grad_kwargs: Additional arguments for class_grad_fn
        :param dist_grad_fn: Similar as class_grad_fn, uses z_t, x_t, x_tau, and alpha_t as inputs
        :param dist_grad_kwargs: Additional args fot dist_grad_fn
        :param x_t_sampling: use sampling when computing x_t
        :param is_x_t_sampling: useful flag to distinguish when x_t is been generated
        :param guided_iterations: Early stop the guided iterations
        '''

        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        for jdx, i in enumerate(indices):

            t = torch.tensor([i] * shape[0], device=device)
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # out is a dictionary with the following (self-explanatory) keys:
            # 'mean', 'variance', 'log_variance'
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # extract sqrtalphacum
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod,
                                           t, shape)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            grads = 0
            x_t_for_classifier = (x_t.clamp(-1, 1) + 1) / 2

            if (class_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + class_grad_fn(x_t=x_t_for_classifier,
                                              **class_grad_kwargs) / alpha_t

            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                grads = grads + dist_grad_fn(z_t=z_t,
                                             x_tau=img,
                                             x_t=x_t,
                                             alpha_t=alpha_t,
                                             **dist_grad_kargs)

            out["mean"] = (
                    out["mean"].float() -
                    out["variance"] * grads
            )

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                z_t = (
                        out["mean"] +
                        nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img)
                )

            # produce x_t in a brute force manner
            if (num_timesteps - (jdx + 1) > 0) and (class_grad_fn is not None) and (dist_grad_fn is not None) and (
                    guided_iterations > jdx):
                x_t = p_sample_loop(
                    diffusion=diffusion,
                    model=model,
                    model_kwargs=model_kwargs,
                    shape=shape,
                    num_timesteps=num_timesteps - (jdx + 1),
                    img=img,
                    t=None,
                    z_t=z_t,
                    clip_denoised=True,
                    device=device,
                    x_t_sampling=use_sampling,
                    is_x_t_sampling=True,
                )[0]

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop


def compute_grad_masks_nested(grads, topk_ratio1=0.3, topk_ratio2=0.1, blur_kernel=7, blur_sigma=3):
    """
    Generate two nested gradient masks (mask2 ⊆ mask1) and apply Gaussian smoothing.

    Args:
        grads: Gradient tensor of shape (B, C, H, W).
        topk_ratio1: Ratio of top gradient regions to keep in the first stage (mask1).
        topk_ratio2: Ratio of top regions within mask1 to keep for a stronger mask (mask2).
        blur_kernel: Kernel size for Gaussian blur (must be an odd number).
        blur_sigma: Standard deviation for Gaussian blur.

    Returns:
        final_mask1, final_mask2: Binary masks of shape (B, 1, H, W), with mask2 ⊆ mask1.
    """
    B, C, H, W = grads.shape

    # Compute gradient magnitude (B, 1, H, W) by averaging the absolute value across channels
    grad_mag = grads.abs().mean(dim=1, keepdim=True)
    flat = grad_mag.view(B, -1)  # Flatten to shape (B, H*W)

    # Top-K selection for mask1
    k1 = max(1, int(topk_ratio1 * H * W))
    _, topk_idx1 = torch.topk(flat, k=k1, dim=1)

    mask1_flat = torch.zeros_like(flat)
    mask1_flat.scatter_(1, topk_idx1, 1.0)
    mask1_binary = mask1_flat.view(B, 1, H, W)

    # Top-K selection within mask1 to get mask2
    k2 = max(1, int(topk_ratio2 * k1))
    topk_vals1 = flat.gather(1, topk_idx1)
    _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
    final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)

    mask2_flat = torch.zeros_like(flat)
    mask2_flat.scatter_(1, final_idx, 1.0)
    mask2_binary = mask2_flat.view(B, 1, H, W)

    # Optional: apply Gaussian blur to smooth masks (ensuring mask2 ⊆ mask1)
    smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
    smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)

    final_mask1 = (smoothed_mask1 > 0).float()
    final_mask2 = (smoothed_mask2 > 0).float()

    # Optional assertion: check that mask2 is a subset of mask1
    assert (final_mask2 <= final_mask1).all(), "mask2 is not a subset of mask1!"
    #
    # final_mask2 = 1.0 - F.max_pool2d(
    #     1.0 - F.max_pool2d(final_mask2, 5, 1, 5 // 2), 5, 1, 5 // 2
    # )
    # # 开运算
    # final_mask2 = F.max_pool2d(
    #     1.0 - F.max_pool2d(1.0 - final_mask2, 3, 1, 3 // 2), 3, 1, 3 // 2
    # )

    return final_mask1, final_mask1




def morph_close(mask: torch.Tensor, kernel_size: int = 3):
    """
    PyTorch 实现的形态学闭运算 (类似 cv2.MORPH_CLOSE)
    mask: [B,H,W] 或 [B,1,H,W] 的二值张量 (0/1)
    """
    if mask.ndim == 3:   # [B,H,W] -> [B,1,H,W]
        mask = mask.unsqueeze(1)

    # --- 膨胀 (dilation) ---
    dilated = F.max_pool2d(mask.float(), kernel_size, stride=1, padding=kernel_size//2)

    # --- 腐蚀 (erosion) ---
    eroded = 1 - F.max_pool2d(1 - dilated, kernel_size, stride=1, padding=kernel_size//2)

    return eroded




def compute_grad_masks_nested2(
    grads,
    topk_ratio1=0.3,
    topk_ratio2=0.1,
    blur_kernel=5,
    blur_sigma=3,
    per_channel=False,
):
    """
    Generate two nested gradient masks (mask2 ⊆ mask1) and apply Gaussian smoothing.

    Args:
        grads: Gradient tensor of shape (B, C, H, W).
        topk_ratio1: Ratio of top gradient regions to keep in the first stage (mask1).
        topk_ratio2: Ratio of top regions within mask1 to keep for a stronger mask (mask2).
        blur_kernel: Kernel size for Gaussian blur (must be an odd number).
        blur_sigma: Standard deviation for Gaussian blur.
        per_channel: If True, compute top-k per channel instead of averaging across channels.

    Returns:
        final_mask1, final_mask2:
            - If per_channel=False: shape (B, 1, H, W)
            - If per_channel=True: shape (B, C, H, W)
    """
    B, C, H, W = grads.shape

    if per_channel:
        # keep per-channel values
        grad_mag = grads.abs()            # (B, C, H, W)
        flat = grad_mag.view(B, C, -1)    # (B, C, H*W)

        k1 = max(1, int(topk_ratio1 * H * W))
        _, topk_idx1 = torch.topk(flat, k=k1, dim=2)

        mask1_flat = torch.zeros_like(flat)
        mask1_flat.scatter_(2, topk_idx1, 1.0)
        mask1_binary = mask1_flat.view(B, C, H, W)

        k2 = max(1, int(topk_ratio2 * k1))
        topk_vals1 = flat.gather(2, topk_idx1)
        _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=2)
        final_idx = topk_idx1.gather(2, topk_idx2_in_topk1)

        mask2_flat = torch.zeros_like(flat)
        mask2_flat.scatter_(2, final_idx, 1.0)
        mask2_binary = mask2_flat.view(B, C, H, W)

    else:
        # average across channels
        grad_mag = grads.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        flat = grad_mag.view(B, -1)  # (B, H*W)

        k1 = max(1, int(topk_ratio1 * H * W))
        _, topk_idx1 = torch.topk(flat, k=k1, dim=1)

        mask1_flat = torch.zeros_like(flat)
        mask1_flat.scatter_(1, topk_idx1, 1.0)
        mask1_binary = mask1_flat.view(B, 1, H, W)

        k2 = max(1, int(topk_ratio2 * k1))
        topk_vals1 = flat.gather(1, topk_idx1)
        _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
        final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)

        mask2_flat = torch.zeros_like(flat)
        mask2_flat.scatter_(1, final_idx, 1.0)
        mask2_binary = mask2_flat.view(B, 1, H, W)

    # Gaussian smoothing
    smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
    smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
    final_mask1 = (smoothed_mask1 > 0).float()
    final_mask2 = (smoothed_mask2 > 0).float()

    assert (final_mask2 <= final_mask1).all(), "mask2 is not a subset of mask1!"

    return final_mask1, final_mask2


def compute_grad_mask(grads, topk_ratio=0.3, blur_kernel=7, blur_sigma=3):
    """
    Generate a gradient mask and its smoothed version.

    Args:
        grads: Gradient tensor of shape (B, C, H, W).
        topk_ratio: Ratio of top gradient regions to keep (binary mask).
        blur_kernel: Kernel size for Gaussian blur (must be an odd number).
        blur_sigma: Standard deviation for Gaussian blur.

    Returns:
        final_mask: Binary mask of shape (B, 1, H, W).
        smoothed_mask: Smoothed (non-binary) mask of shape (B, 1, H, W).  # <- 原第二个mask
    """
    B, C, H, W = grads.shape

    # (B, 1, H, W)
    grad_mag = grads.abs().mean(dim=1, keepdim=True)
    flat = grad_mag.view(B, -1)

    # Top-K selection -> binary mask
    k = max(1, int(topk_ratio * H * W))
    _, topk_idx = torch.topk(flat, k=k, dim=1)

    mask_flat = torch.zeros_like(flat)
    mask_flat.scatter_(1, topk_idx, 1.0)
    mask_binary = mask_flat.view(B, 1, H, W)
    print(mask_binary.mean())
    smoothed_mask = gaussian_blur(mask_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
    final_mask = (smoothed_mask > 0).float()
    # print(final_mask.mean())


    # minv = smoothed_mask.amin(dim=(1, 2, 3), keepdim=True)
    # maxv = smoothed_mask.amax(dim=(1, 2, 3), keepdim=True)
    # norm = (smoothed_mask - minv) / (maxv - minv + 1e-8)
    #
    # # 2. 调整均值到0.5
    # meanv = norm.mean(dim=(1, 2, 3), keepdim=True)
    # scaled = norm * (0.5 / (meanv + 1e-8))
    #
    # smoothed_mask_adj = scaled.clamp(0, 1)
    smoothed_mask = 1.0 - F.max_pool2d(
        1.0 - F.max_pool2d(final_mask, 5, 1, 5 // 2), 5, 1, 5 // 2
    )
    # 开运算
    smoothed_mask = F.max_pool2d(
        1.0 - F.max_pool2d(1.0 - smoothed_mask, 3, 1, 3 // 2), 3, 1, 3 // 2
    )
    print(smoothed_mask.mean())

    return final_mask, smoothed_mask


def compute_topp_grads_mask(grads, topp_ratio=0.9, blur_kernel=7, blur_sigma=3):
    """
    Compute a gradient-based binary mask using Top-P (nucleus) sampling and apply Gaussian smoothing.

    Args:
        grads (Tensor): Gradient tensor of shape (B, C, H, W).
        topp_ratio (float): Cumulative ratio threshold for the mask.
        blur_kernel (int): Kernel size for Gaussian blur (must be odd).
        blur_sigma (float): Standard deviation for Gaussian blur.

    Returns:
        final_mask (Tensor): Binary mask of shape (B, 1, H, W).
        smoothed_mask (Tensor): Smoothed soft mask of shape (B, 1, H, W).
    """
    B, C, H, W = grads.shape
    grad_mag = grads.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
    flat = grad_mag.view(B, -1)  # (B, H*W)

    # Top-P mask
    sorted_vals, sorted_idx = torch.sort(flat, dim=1, descending=True)
    cum_sum = torch.cumsum(sorted_vals, dim=1)
    total_sum = cum_sum[:, -1].unsqueeze(1)
    topp_mask = (cum_sum / total_sum) <= topp_ratio
    topp_mask[:, 0] = 1  # Ensure at least one pixel

    mask_flat = torch.zeros_like(flat)
    for b in range(B):
        idx_b = sorted_idx[b][topp_mask[b]]
        mask_flat[b, idx_b] = 1.0
    mask_binary = mask_flat.view(B, 1, H, W)

    # Apply Gaussian smoothing
    smoothed_mask = gaussian_blur(mask_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
    final_mask = (smoothed_mask > 0).float()
    smoothed_mask = torch.sigmoid(smoothed_mask)




    return final_mask, smoothed_mask


import torch
# 假设你已有 gaussian_blur，可来自 torchvision.transforms.functional 或 kornia 等
# from torchvision.transforms.functional import gaussian_blur

def compute_grad_soft_masks_nested_channelwise(
        grads,
        topk_ratio=0.3,
        binary_ratio=0.15,
        blur_kernel=3,
        blur_sigma=1,
        selection_mode="mean",   # 新增：'per_channel' 或 'mean'
):
    """
    基于梯度选择显著像素并生成软/硬掩码。

    Args:
        grads: (B, C, H, W) 的梯度张量
        topk_ratio: 相对 H*W 的比例，用于确定 top-k 数量
        binary_ratio: 平滑后进行二值化的阈值
        blur_kernel: 高斯平滑核大小
        blur_sigma: 高斯平滑标准差
        selection_mode:
            - 'per_channel': 各通道各取 top-k，然后跨通道并集（与原逻辑一致）
            - 'mean': 先对通道求均值，再在单通道图上取 top-k

    Returns:
        final_mask: (B, 1, H, W) 二值掩码
        smoothed_mask: (B, 1, H, W) 二值化前的平滑软掩码
    """
    B, C, H, W = grads.shape
    grad_mag = grads.abs()  # (B, C, H, W)

    HW = H * W
    k = max(1, min(HW, int(topk_ratio * HW)))

    if selection_mode == "per_channel":
        # —— 原逻辑：各通道 top-k + 并集 ——
        flat = grad_mag.view(B, C, -1)                        # (B, C, HW)
        _, topk_idx = torch.topk(flat, k=k, dim=2)            # (B, C, k)
        mask_flat_ch = torch.zeros_like(flat)                 # (B, C, HW)
        mask_flat_ch.scatter_(2, topk_idx, 1.0)
        mask_ch = mask_flat_ch.view(B, C, H, W)               # (B, C, H, W)
        mask_binary = mask_ch.max(dim=1, keepdim=True).values.float()  # (B, 1, H, W)

    elif selection_mode == "mean":
        # —— 新逻辑：通道均值后，整图 top-k ——
        mean_map = grad_mag.mean(dim=1, keepdim=True)         # (B, 1, H, W)
        flat_mean = mean_map.view(B, 1, -1)                   # (B, 1, HW)
        _, topk_idx = torch.topk(flat_mean, k=k, dim=2)       # (B, 1, k)
        mask_flat = torch.zeros_like(flat_mean)               # (B, 1, HW)
        mask_flat.scatter_(2, topk_idx, 1.0)
        mask_binary = mask_flat.view(B, 1, H, W).float()      # (B, 1, H, W)

    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    # 高斯平滑 + 二值化
    smoothed_mask = gaussian_blur(mask_binary, kernel_size=blur_kernel, sigma=blur_sigma)
    final_mask = (smoothed_mask > binary_ratio).float()
    return final_mask, smoothed_mask


def compute_grad_masks_dialation(
    grads,
    topk_ratio1=0.3,
    topk_ratio2=0.1,
    dilate_kernel=5,
    per_channel=False,
):
    """
    Generate two nested gradient masks (mask2 ⊆ mask1) and apply morphological dilation.

    Args:
        grads: Gradient tensor of shape (B, C, H, W).
        topk_ratio1: Ratio of top gradient regions to keep in the first stage (mask1).
        topk_ratio2: Ratio of top regions within mask1 to keep for a stronger mask (mask2).
        dilate_kernel: Kernel size for morphological dilation (odd number).
        per_channel: If True, compute top-k per channel instead of averaging across channels.

    Returns:
        final_mask1, final_mask2:
            - If per_channel=False: shape (B, 1, H, W)
            - If per_channel=True: shape (B, C, H, W)
    """
    B, C, H, W = grads.shape

    if per_channel:
        grad_mag = grads.abs()            # (B, C, H, W)
        flat = grad_mag.view(B, C, -1)    # (B, C, H*W)

        k1 = max(1, int(topk_ratio1 * H * W))
        _, topk_idx1 = torch.topk(flat, k=k1, dim=2)
        mask1_flat = torch.zeros_like(flat)
        mask1_flat.scatter_(2, topk_idx1, 1.0)
        mask1_binary = mask1_flat.view(B, C, H, W)

        k2 = max(1, int(topk_ratio2 * k1))
        topk_vals1 = flat.gather(2, topk_idx1)
        _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=2)
        final_idx = topk_idx1.gather(2, topk_idx2_in_topk1)
        mask2_flat = torch.zeros_like(flat)
        mask2_flat.scatter_(2, final_idx, 1.0)
        mask2_binary = mask2_flat.view(B, C, H, W)

    else:
        grad_mag = grads.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        flat = grad_mag.view(B, -1)  # (B, H*W)

        k1 = max(1, int(topk_ratio1 * H * W))
        _, topk_idx1 = torch.topk(flat, k=k1, dim=1)
        mask1_flat = torch.zeros_like(flat)
        mask1_flat.scatter_(1, topk_idx1, 1.0)
        mask1_binary = mask1_flat.view(B, 1, H, W)

        k2 = max(1, int(topk_ratio2 * k1))
        topk_vals1 = flat.gather(1, topk_idx1)
        _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
        final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)
        mask2_flat = torch.zeros_like(flat)
        mask2_flat.scatter_(1, final_idx, 1.0)
        mask2_binary = mask2_flat.view(B, 1, H, W)

    # ===== Morphological dilation (instead of Gaussian blur) =====
    pad = dilate_kernel // 2
    kernel = torch.ones((1, 1, dilate_kernel, dilate_kernel), device=grads.device)

    # mask1
    final_mask1 = F.max_pool2d(mask1_binary.float(), kernel_size=dilate_kernel, stride=1, padding=pad)
    # mask2
    final_mask2 = F.max_pool2d(mask2_binary.float(), kernel_size=dilate_kernel, stride=1, padding=pad)

    assert (final_mask2 <= final_mask1).all(), "mask2 is not a subset of mask1!"

    return final_mask1, final_mask2

def get_MaskDiME_sampling_CelebA(use_sampling=False):
    '''
    Returns MaskDiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=use_sampling,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      grads_clip=True,
                      x_t_rate=1,
                      clip_sigma=3,
                      blur_kernel=5,
                      blur_sigma=3,
                      per_channel=True,
                      ):

        x_t = img.detach().clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]
        grad_mask_x_t =None

        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # Save current state
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # Predict mean and variance at the current timestep
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            pred_xstart = out["pred_xstart"].detach()
            if grad_mask_x_t is not None:
                x_t = grad_mask_x_t * pred_xstart + (1 - grad_mask_x_t) * img.detach().clone()
            else:
                x_t = x_t

            # Compute gradient (if required)
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            # nonzero_mask = (t_val != 0).float().view(-1, *([1] * (len(shape) - 1)))
            grads = None
            if (class_grad_fn is not None) and (jdx < guided_iterations):
                # === Key Modification 2: Compute gradients using the blended x_t ===
                grads_class = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t
                grad_mask, grad_mask_x_t = compute_grad_masks_nested2(grads_class, topk_ratio1=topk_ratio,
                                                                      topk_ratio2=x_t_rate, blur_kernel=blur_kernel,
                                                                      blur_sigma=blur_sigma, per_channel=per_channel)

                if grads_clip is True:
                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - clip_sigma * std
                    upper = mean + clip_sigma * std
                    grads_class = grads_class.clamp(min=lower, max=upper)

                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class


            # Apply gradient guidance
            if grads is not None and torch.abs(grads).sum() > 0:
                out["mean"] = out["mean"] - out["variance"] * grads
                z_original = diffusion.q_sample(img, t_val)
                z_t = out["mean"] * grad_mask + z_original * (1 - grad_mask)


        return z_t, x_t_steps, z_t_steps

    return p_sample_loop



import torch



def get_nocleanmask_sampling_CelebA(use_sampling=False):
    '''
    Returns MaskDiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=use_sampling,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      grads_clip=True,
                      x_t_rate=1,
                      clip_sigma=3,
                      blur_kernel=5,
                      blur_sigma=3,
                      per_channel=True,
                      ):

        x_t = img.detach().clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]
        grad_mask_x_t =None

        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # Save current state
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # Predict mean and variance at the current timestep
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            pred_xstart = out["pred_xstart"].detach()
            if jdx > 0:
                x_t = pred_xstart

            # Compute gradient (if required)
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            # nonzero_mask = (t_val != 0).float().view(-1, *([1] * (len(shape) - 1)))
            grads = None
            if (class_grad_fn is not None) and (jdx < guided_iterations):
                # === Key Modification 2: Compute gradients using the blended x_t ===
                grads_class = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t
                grad_mask, grad_mask_x_t = compute_grad_masks_nested2(grads_class, topk_ratio1=topk_ratio,
                                                                      topk_ratio2=x_t_rate, blur_kernel=blur_kernel,
                                                                      blur_sigma=blur_sigma, per_channel=per_channel)

                if grads_clip is True:
                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - clip_sigma * std
                    upper = mean + clip_sigma * std
                    grads_class = grads_class.clamp(min=lower, max=upper)

                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class


            # Apply gradient guidance
            if grads is not None and torch.abs(grads).sum() > 0:
                out["mean"] = out["mean"] - out["variance"] * grads
                z_original = diffusion.q_sample(img, t_val)
                z_t = out["mean"] * grad_mask + z_original * (1 - grad_mask)


        return z_t, x_t_steps, z_t_steps

    return p_sample_loop


def get_fixed_sampling_CelebA(use_sampling=False):
    '''
    Returns MaskDiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=use_sampling,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      grads_clip=True,
                      x_t_rate=1,
                      clip_sigma=3,
                      blur_kernel=5,
                      blur_sigma=3,
                      per_channel=True,
                      ):

        x_t = img.detach().clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        grad_mask_x_t = None
        grad_mask = None

        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # Save current state
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # Predict mean and variance at the current timestep
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            pred_xstart = out["pred_xstart"].detach()
            if grad_mask_x_t is not None:
                x_t = grad_mask_x_t * pred_xstart + (1 - grad_mask_x_t) * img.detach().clone()
            else:
                x_t = x_t

            # Compute gradient (if required)
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            grads = None

            if (class_grad_fn is not None) and (jdx < guided_iterations):
                grads_class = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t

                # ===== Fixed mask: compute once =====
                if grad_mask is None:
                    grad_mask, grad_mask_x_t = compute_grad_masks_nested2(
                        grads_class,
                        topk_ratio1=topk_ratio,
                        topk_ratio2=x_t_rate,
                        blur_kernel=blur_kernel,
                        blur_sigma=blur_sigma,
                        per_channel=per_channel
                    )

                if grads_clip is True:
                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - clip_sigma * std
                    upper = mean + clip_sigma * std
                    grads_class = grads_class.clamp(min=lower, max=upper)

                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class

            # Apply gradient guidance
            if grads is not None and torch.abs(grads).sum() > 0:
                out["mean"] = out["mean"] - out["variance"] * grads
                z_original = diffusion.q_sample(img, t_val)
                z_t = out["mean"] * grad_mask + z_original * (1 - grad_mask)

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop


def get_DiMElastmask_sampling_CelebA(use_sampling=False):
    '''
    noMaskDiME sampling:
    - No mask during diffusion trajectory
    - Compute TOP-K mask BASED ON ORIGINAL IMAGE (img)
    - Apply mask ONLY at the final step
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=use_sampling,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      grads_clip=True,
                      x_t_rate=1,
                      clip_sigma=3,
                      blur_kernel=5,
                      blur_sigma=3,
                      per_channel=True,
                      ):

        # --------------------------------------------------
        # Initialization
        # --------------------------------------------------
        x_t = None
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []

        indices = list(range(num_timesteps))[::-1]

        # --------------------------------------------------
        # Reverse diffusion
        # --------------------------------------------------
        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # Save states
            x_t_steps.append(
                x_t.detach() if x_t is not None else img.detach()
            )
            z_t_steps.append(z_t.detach())

            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            pred_xstart = out["pred_xstart"].detach()
            x_t = pred_xstart if x_t is not None else img.detach()

            alpha_t = _extract_into_tensor(
                diffusion.sqrt_alphas_cumprod, t_val, shape
            )

            nonzero_mask = (t_val != 0).float().view(
                -1, *([1] * (len(shape) - 1))
            )

            grads = None

            # -----------------------------
            # Standard guidance (no mask)
            # -----------------------------
            if (class_grad_fn is not None) and (jdx < guided_iterations):
                grads_class = class_grad_fn(
                    x_t=x_t, **class_grad_kwargs
                ) / alpha_t

                if grads_clip:
                    mean = grads_class.mean()
                    std = grads_class.std()
                    grads_class = grads_class.clamp(
                        mean - clip_sigma * std,
                        mean + clip_sigma * std
                    )

                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class

            if grads is not None and torch.abs(grads).sum() > 0:
                out["mean"] = out["mean"] - out["variance"] * grads

            # -----------------------------
            # z_t update
            # -----------------------------
            if not x_t_sampling:
                z_t = out["mean"]
            else:
                noise = torch.randn_like(img) if jdx < len(indices) - 1 else torch.zeros_like(img)
                z_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

            # ==================================================
            # ⭐ FINAL STEP ONLY: mask computed FROM ORIGINAL IMAGE
            # ==================================================
            if i == 0 and class_grad_fn is not None:
                grads_img = class_grad_fn(
                    x_t=img, **class_grad_kwargs
                )

                grad_mask, _ = compute_grad_masks_nested2(
                    grads_img,
                    topk_ratio1=topk_ratio,
                    topk_ratio2=x_t_rate,
                    blur_kernel=blur_kernel,
                    blur_sigma=blur_sigma,
                    per_channel=per_channel
                )

                z_original = diffusion.q_sample(img, t_val)
                z_t = z_t * grad_mask + z_original * (1 - grad_mask)

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop

def get_noMaskDiME_sampling_CelebA(use_sampling=False):
    '''
    Returns noMaskDiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=use_sampling,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      grads_clip=True,
                      x_t_rate=1,
                      clip_sigma=3,
                      blur_kernel=5,
                      blur_sigma=3,
                      per_channel=True,
                      ):

        x_t = None
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # Save current state
            if x_t is not None:
                x_t_steps.append(x_t.detach())
            else:
                x_t_steps.append(img.clone().detach())

            z_t_steps.append(z_t.detach())

            # Predict mean and variance at the current timestep
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # === Key Modification 1: Use mask to blend predicted result with original image ===
            # Use predicted x0 only in masked regions, preserve original image outside the mask
            pred_xstart = out["pred_xstart"].detach()
            if x_t is not None:
                x_t = pred_xstart
            else:
                x_t =  img.clone()

            # Compute gradient (if required)
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            nonzero_mask = (t_val != 0).float().view(-1, *([1] * (len(shape) - 1)))
            grads = None
            if (class_grad_fn is not None) and (jdx < guided_iterations):
                # === Key Modification 2: Compute gradients using the blended x_t ===
                grads_class = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t

                if grads_clip is True:
                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - clip_sigma * std
                    upper = mean + clip_sigma * std
                    grads_class = grads_class.clamp(min=lower, max=upper)

                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,  # Use blended x_t
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class


            # Apply gradient guidance
            if grads is not None and torch.abs(grads).sum() > 0:
                out["mean"] = out["mean"] - out["variance"] * grads


            if not x_t_sampling:
                z_t = out["mean"]
            else:
                noise = torch.randn_like(img) if jdx < len(indices) - 1 else torch.zeros_like(img)
                z_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop



def get_MaskDiME_iterative_sampling_CelebA(use_sampling=False):
    '''
    Returns MaskDiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      is_x_t_sampling=True,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      grads_clip=True,
                      x_t_rate=1,
                      sigma=3,
                      MAX_MASK_ACCUM_STEPS=3,
                      mask_class="binary",
                      accum_method="max",
                      ):
        fixed_epsilon = torch.randn_like(img)

        pred_xstart_steps=[]
        outmean_steps=[]
        guidedmean_steps=[]
        z_o_steps=[]
        z_t_mask_steps = []
        c_z_t_mask_steps = []
        x_t_mask_steps = []
        c_x_t_mask_steps = []
        grads_steps=[]
        x_t = img.clone()
        z_t = diffusion.q_sample(img, t, noise=fixed_epsilon) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]


        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # 保存当前状态

            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # 预测当前步的均值和方差
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # === 关键修改1：使用掩码混合预测结果和原始图像 ===
            # 只在掩码区域内使用预测的x0，掩码区域外保持原始图像
            pred_xstart = out["pred_xstart"].detach()
            pred_xstart_steps.append(pred_xstart.detach())



            # 计算梯度（如果需要）
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            nonzero_mask = (t_val != 0).float().view(-1, *([1] * (len(shape) - 1)))
            grads = None
            if (class_grad_fn is not None) and (jdx < guided_iterations):
                # === 关键修改2：使用混合后的x_t计算梯度 ===
                grads_class = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t

                if grads_clip == True:
                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - sigma * std
                    upper = mean + sigma * std
                    grads_class = grads_class.clamp(min=lower, max=upper)

                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,  # 使用混合后的x_t
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class
            grad_mask, grad_mask_x_t = compute_grad_masks_nested(grads_class, topk_ratio, x_t_rate)
            # 应用梯度引导
            if grads is not None and torch.abs(grads).sum() > 0:


                outmean_steps.append(out["mean"].detach())
                out["mean"]  = out["mean"] - out["variance"] * grads
                grads_steps.append(grads)
                guidedmean_steps.append(out["mean"].detach())

                z_original = diffusion.q_sample(img, t_val,noise=fixed_epsilon)
                out["mean"] = out["mean"]  * grad_mask + z_original * (1 - grad_mask)
            z_o_steps.append(z_original.detach())
            z_t_mask_steps.append(grad_mask.detach())
            c_z_t_mask_steps.append((1-grad_mask).detach())
            x_t_mask_steps.append(grad_mask_x_t.detach())
            c_x_t_mask_steps.append((1-grad_mask_x_t).detach())

            if not is_x_t_sampling:
                z_t = out["mean"]
            else:
                noise = torch.randn_like(img) if jdx < len(indices) - 1 else torch.zeros_like(img)
                z_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

            x_t = grad_mask_x_t * pred_xstart + (1 - grad_mask_x_t) * img


        return z_t, x_t_steps, z_t_steps, outmean_steps, z_t_mask_steps,c_z_t_mask_steps, x_t_mask_steps,c_x_t_mask_steps,pred_xstart_steps,z_o_steps,guidedmean_steps,grads_steps


    return p_sample_loop


def get_MaskDiME_iterative_sampling_ImageNet(use_sampling=False):


    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      is_x_t_sampling=False,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      grads_clip=True,
                      x_t_rate=1,
                      sigma=3,
                      ):

        pred_xstart_steps=[]
        outmean_steps=[]
        guidedmean_steps=[]
        z_o_steps=[]
        z_t_mask_steps = []
        c_z_t_mask_steps = []
        x_t_mask_steps = []
        c_x_t_mask_steps = []
        grads_steps=[]
        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]
        grad_mask_x_t=None


        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # 预测当前步的均值和方差
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )


            pred_xstart = out["pred_xstart"].detach()
            if grad_mask_x_t is not None:
                x_t = grad_mask_x_t * pred_xstart + (1 - grad_mask_x_t) * img.detach().clone()
            else:
                x_t = x_t
            pred_xstart_steps.append(pred_xstart.detach())

            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            nonzero_mask = (t_val != 0).float().view(-1, *([1] * (len(shape) - 1)))
            grads = None

            x_t_for_classifier = (x_t.clamp(-1, 1) + 1) / 2

            if (class_grad_fn is not None) and (jdx < guided_iterations):
                # === 关键修改2：使用混合后的x_t计算梯度 ===
                grads_class = class_grad_fn(x_t=x_t_for_classifier, **class_grad_kwargs) / alpha_t
                grad_mask, grad_mask_x_t = compute_grad_masks_nested2(grads_class, topk_ratio, x_t_rate,blur_kernel=5,sigma=3)

                if grads_clip==True:
                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - sigma * std
                    upper = mean + sigma * std
                    grads_class = grads_class.clamp(min=lower, max=upper)
                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,  # 使用混合后的x_t
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class

            # 应用梯度引导
            if grads is not None and torch.abs(grads).sum() > 0:
                outmean_steps.append(out["mean"].detach())
                out["mean"]  = out["mean"] - out["variance"] * grads
                grads_steps.append(grads)
                guidedmean_steps.append(out["mean"].detach())
                z_original = diffusion.q_sample(img, t_val)
                out["mean"] = out["mean"]  * grad_mask + z_original * (1 - grad_mask)
            z_o_steps.append(z_original.detach())
            z_t_mask_steps.append(grad_mask.detach())
            c_z_t_mask_steps.append((1-grad_mask).detach())
            x_t_mask_steps.append(grad_mask_x_t.detach())
            c_x_t_mask_steps.append((1-grad_mask_x_t).detach())
            # 采样下一步
            if not is_x_t_sampling:
                z_t = out["mean"]
            else:
                noise = torch.randn_like(img) if jdx < len(indices) - 1 else torch.zeros_like(img)
                z_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise


        return z_t, x_t_steps, z_t_steps, outmean_steps, z_t_mask_steps,c_z_t_mask_steps, x_t_mask_steps,c_x_t_mask_steps,pred_xstart_steps,z_o_steps,guidedmean_steps,grads_steps


    return p_sample_loop

def get_MaskDiME_iterative_sampling():
    '''
    Returns MaskDiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      guided_iterations=9999999,
                      grad_scale=10,
                      topk_ratio=0.1,
                      grads_clip=True,
                      x_t_rate=0.15,
                      sigma=3,
                      blur_kernel=7,
                      blur_sigma=3
                      ):
        x_t = img.detach().clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]
        grad_mask_x_t=None

        def compute_grad_masks(
                grads,
                topk_ratio1=0.3,
                topk_ratio2=0.1,
                blur_kernel=5,
                blur_sigma=3,
                per_channel=False,
        ):
            """
            Generate two nested gradient masks (mask2 ⊆ mask1) and apply Gaussian smoothing.

            Args:
                grads: Gradient tensor of shape (B, C, H, W).
                topk_ratio1: Ratio of top gradient regions to keep in the first stage (mask1).
                topk_ratio2: Ratio of top regions within mask1 to keep for a stronger mask (mask2).
                blur_kernel: Kernel size for Gaussian blur (must be an odd number).
                blur_sigma: Standard deviation for Gaussian blur.
                per_channel: If True, compute top-k per channel instead of averaging across channels.

            Returns:
                final_mask1, final_mask2:
                    - If per_channel=False: shape (B, 1, H, W)
                    - If per_channel=True: shape (B, C, H, W)
            """
            B, C, H, W = grads.shape

            if per_channel:
                # keep per-channel values
                grad_mag = grads.abs()  # (B, C, H, W)
                flat = grad_mag.view(B, C, -1)  # (B, C, H*W)

                k1 = max(1, int(topk_ratio1 * H * W))
                _, topk_idx1 = torch.topk(flat, k=k1, dim=2)

                mask1_flat = torch.zeros_like(flat)
                mask1_flat.scatter_(2, topk_idx1, 1.0)
                mask1_binary = mask1_flat.view(B, C, H, W)

                k2 = max(1, int(topk_ratio2 * k1))
                topk_vals1 = flat.gather(2, topk_idx1)
                _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=2)
                final_idx = topk_idx1.gather(2, topk_idx2_in_topk1)

                mask2_flat = torch.zeros_like(flat)
                mask2_flat.scatter_(2, final_idx, 1.0)
                mask2_binary = mask2_flat.view(B, C, H, W)

            else:
                # average across channels
                grad_mag = grads.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
                flat = grad_mag.view(B, -1)  # (B, H*W)

                k1 = max(1, int(topk_ratio1 * H * W))
                _, topk_idx1 = torch.topk(flat, k=k1, dim=1)

                mask1_flat = torch.zeros_like(flat)
                mask1_flat.scatter_(1, topk_idx1, 1.0)
                mask1_binary = mask1_flat.view(B, 1, H, W)

                k2 = max(1, int(topk_ratio2 * k1))
                topk_vals1 = flat.gather(1, topk_idx1)
                _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
                final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)

                mask2_flat = torch.zeros_like(flat)
                mask2_flat.scatter_(1, final_idx, 1.0)
                mask2_binary = mask2_flat.view(B, 1, H, W)

            # Gaussian smoothing
            smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            final_mask1 = (smoothed_mask1 > 0).float()
            final_mask2 = (smoothed_mask2 > 0).float()

            assert (final_mask2 <= final_mask1).all(), "mask2 is not a subset of mask1!"

            return final_mask1, final_mask2

        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)


            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())


            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            pred_xstart = out["pred_xstart"].detach()
            if grad_mask_x_t is not None:
                x_t = grad_mask_x_t * pred_xstart + (1 - grad_mask_x_t) * img.detach().clone()
            else:
                x_t = x_t
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)

            grads = None

            x_t_for_classifier = (x_t.clamp(-1, 1) + 1) / 2

            if (class_grad_fn is not None) and (jdx < guided_iterations):
                # === 关键修改2：使用混合后的x_t计算梯度 ===
                grads_class = class_grad_fn(x_t=x_t_for_classifier, **class_grad_kwargs) / alpha_t
                grad_mask, grad_mask_x_t = compute_grad_masks(grads_class, topk_ratio1=topk_ratio, topk_ratio2=x_t_rate)

                if grads_clip==True:
                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - sigma * std
                    upper = mean + sigma * std
                    grads_class = grads_class.clamp(min=lower, max=upper)
                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,  # 使用混合后的x_t
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class

            # 应用梯度引导
            if grads is not None and torch.abs(grads).sum() > 0:
                out["mean"]  = out["mean"] - out["variance"] * grads
                z_original = diffusion.q_sample(img, t_val)
                z_t = out["mean"]  * grad_mask + z_original * (1 - grad_mask)

        return z_t, x_t_steps, z_t_steps


    return p_sample_loop

def get_FastDiME_Mask_iterative_sampling_BDD2(use_sampling=False):
    '''
    Returns DiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      is_x_t_sampling=True,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      MAX_MASK_ACCUM_STEPS=1,
                      mask_class="binary",
                      grads_clip=True,
                      accum_method="max",
                      x_t_rate=0.05
                      ):

        def compute_grad_mask_topk_binary(grads, topk_ratio=0.3):
            B, C, H, W = grads.shape
            grad_magnitude = torch.abs(grads)
            grad_magnitude = grad_magnitude.mean(dim=1, keepdim=True)
            flat = grad_magnitude.view(B, -1)
            k = max(1, int(topk_ratio * H * W))
            topk_vals, topk_idx = torch.topk(flat, k=k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, topk_idx, 1.0)
            return mask.view(B, 1, H, W)

        def compute_grad_soft_masks_nested(grads, topk_ratio1=0.3, topk_ratio2=0.1, blur_kernel=7, blur_sigma=3):
            """
            Generate two nested gradient masks (mask2 ⊆ mask1) and apply Gaussian smoothing.

            Args:
                grads: Gradient tensor of shape (B, C, H, W).
                topk_ratio1: Ratio of top gradient regions to keep in the first stage (mask1).
                topk_ratio2: Ratio of top regions within mask1 to keep for a stronger mask (mask2).
                blur_kernel: Kernel size for Gaussian blur (must be an odd number).
                blur_sigma: Standard deviation for Gaussian blur.

            Returns:
                final_mask1, final_mask2: Binary masks of shape (B, 1, H, W), with mask2 ⊆ mask1.
            """
            B, C, H, W = grads.shape

            # Compute gradient magnitude (B, 1, H, W) by averaging the absolute value across channels
            grad_mag = grads.abs().mean(dim=1, keepdim=True)
            flat = grad_mag.view(B, -1)  # Flatten to shape (B, H*W)

            # Top-K selection for mask1
            k1 = max(1, int(topk_ratio1 * H * W))
            _, topk_idx1 = torch.topk(flat, k=k1, dim=1)

            mask1_flat = torch.zeros_like(flat)
            mask1_flat.scatter_(1, topk_idx1, 1.0)
            mask1_binary = mask1_flat.view(B, 1, H, W)

            # Top-K selection within mask1 to get mask2
            k2 = max(1, int(topk_ratio2 * k1))
            topk_vals1 = flat.gather(1, topk_idx1)
            _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
            final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)

            mask2_flat = torch.zeros_like(flat)
            mask2_flat.scatter_(1, final_idx, 1.0)
            mask2_binary = mask2_flat.view(B, 1, H, W)
            #
            # # Optional: apply Gaussian blur to smooth masks (ensuring mask2 ⊆ mask1)
            # smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            # smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            #
            # # Binarize after smoothing
            # final_mask1 = (smoothed_mask1 > 0).float()
            # final_mask2 = (smoothed_mask2 > 0).float()

            # Optional assertion: check that mask2 is a subset of mask1
            # assert (final_mask2 <= final_mask1).all(), "mask2 is not a subset of mask1!"

            return mask1_binary, mask2_binary

        mask_buffer = []
        mask_x_t_buffer=[]
        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # 保存当前状态
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # 预测当前步的均值和方差
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            pred_xstart = out["pred_xstart"].detach()
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            nonzero_mask = (t_val != 0).float().view(-1, *([1] * (len(shape) - 1)))
            grads = None

            x_t_for_classifier = (x_t.clamp(-1, 1) + 1) / 2

            if (class_grad_fn is not None) and (jdx < guided_iterations):
                grads_class = class_grad_fn(x_t=x_t_for_classifier, **class_grad_kwargs) / alpha_t
                if grads_clip==True:

                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - 4 * std
                    upper = mean + 4 * std
                    grads_class = grads_class.clamp(min=lower, max=upper)
                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class

            # 应用梯度引导
            if grads is not None and torch.abs(grads).sum() > 0:
                # 计算动态TopK掩码
                if mask_class == "binary":

                    # grad_mask = compute_grad_mask_topk_binary(grads_class,  topk_ratio)
                    # grad_mask_x_t = compute_grad_mask_topk_binary(grads_class,  x_t_rate*topk_ratio)
                    grad_mask,grad_mask_x_t = compute_grad_soft_masks_nested(grads_class, topk_ratio,x_t_rate)


                # 更新掩码缓冲区
                mask_buffer.append(grad_mask.detach())
                mask_x_t_buffer.append(grad_mask_x_t.detach())
                if len(mask_buffer) > MAX_MASK_ACCUM_STEPS:
                    mask_buffer.pop(0)
                if len(mask_x_t_buffer) > MAX_MASK_ACCUM_STEPS:
                    mask_x_t_buffer.pop(0)

                if accum_method=="mean":
                    accum_mask = torch.stack(mask_buffer).mean(dim=0)
                    accum_mask_x_t = torch.stack(mask_x_t_buffer).mean(dim=0)
                elif accum_method=="max":
                    accum_mask = torch.stack(mask_buffer).max(dim=0).values
                    accum_mask_x_t = torch.stack(mask_x_t_buffer).max(dim=0).values

                out["mean"]  = out["mean"] - out["variance"] * grads
                z_original = diffusion.q_sample(img, t_val)
                out["mean"] = out["mean"]  * accum_mask + z_original * (1 - accum_mask)

            # 采样下一步
            if not is_x_t_sampling:
                z_t = out["mean"]
            else:
                noise = torch.randn_like(img) if jdx < len(indices) - 1 else torch.zeros_like(img)
                z_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

            x_t = accum_mask_x_t * pred_xstart + (1 - accum_mask_x_t) * img

        return z_t, x_t_steps, z_t_steps


    return p_sample_loop







def get_ThDiME_iterative_samplingHQ(use_sampling=False):
    '''
    Returns DiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=use_sampling,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      MAX_MASK_ACCUM_STEPS=3,
                      mask_class="binary",
                      grads_clip=True,
                      accum_method="max",
                      x_t_rate=0.05
                      ):

        stabilizer = ADAMGradientStabilization(
            beta_1=0.9,
            beta_2=0.999,
            eps=1e-8,
            reset_step=20,
        )

        # ========== 新增：momentum 配置（同时用于 grad_mask 与 grad_mask_x_t） ==========
        momentum_beta = 0.9
        prev_grad_mask_x_t_ema = None   # 软掩码的 EMA 缓存
        prev_grad_mask_ema = None       # 二值掩码的 EMA 缓存（软态缓存，用于阈值回二值）
        # ======================================================================

        # ========== 新增：分类器梯度的 EMA momentum（与 mask 一致的思路） ==========
        grad_momentum_beta = 0.9
        prev_grads_class_ema = None
        # ======================================================================

        # ========== 新增：momentum 配置（同时用于 grad_mask 与 grad_mask_x_t） =========
        prev_grad_mask_x_t_ema = None  # 软掩码的 EMA 缓存
        # --- 新增：x_t 的 EMA momentum 系数 ---
        x_t_momentum_beta = 0.9

        # ==========================================================================

        def compute_grad_mask(grads, topk_ratio=0.3, blur_kernel=7, blur_sigma=3):
            """
            Generate a gradient mask and its smoothed version.
            """
            B, C, H, W = grads.shape
            grad_mag = grads.abs().mean(dim=1, keepdim=True)
            flat = grad_mag.view(B, -1)
            k = max(1, int(topk_ratio * H * W))
            _, topk_idx = torch.topk(flat, k=k, dim=1)

            mask_flat = torch.zeros_like(flat)
            mask_flat.scatter_(1, topk_idx, 1.0)
            mask_binary = mask_flat.view(B, 1, H, W)

            smoothed_mask = gaussian_blur(mask_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            final_mask = (smoothed_mask > 0).float()
            return final_mask, smoothed_mask

        def compute_grad_soft_masks_nested2(
                grads,
                topk_ratio1=0.3,
                topk_ratio2=0.1,
                blur_kernel=7,
                blur_sigma=3,
                per_channel=True,
        ):
            """
            生成两个嵌套的梯度mask（mask2 ⊆ mask1），并进行高斯平滑。
            如果 per_channel=True，则对每个通道单独取 topk。
            """
            B, C, H, W = grads.shape

            if per_channel:
                # 保持通道维度，不做平均
                grad_mag = grads.abs()  # [B, C, H, W]
                flat = grad_mag.view(B, C, -1)  # [B, C, HW]

                k1 = max(1, int(topk_ratio1 * H * W))
                _, topk_idx1 = torch.topk(flat, k=k1, dim=2)

                mask1_flat = torch.zeros_like(flat)
                mask1_flat.scatter_(2, topk_idx1, 1.0)
                mask1_binary = mask1_flat.view(B, C, H, W)

                k2 = max(1, int(topk_ratio2 * k1))
                topk_vals1 = flat.gather(2, topk_idx1)
                _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=2)
                final_idx = topk_idx1.gather(2, topk_idx2_in_topk1)

                mask2_flat = torch.zeros_like(flat)
                mask2_flat.scatter_(2, final_idx, 1.0)
                mask2_binary = mask2_flat.view(B, C, H, W)

            else:
                # 原始逻辑：通道平均
                grad_mag = grads.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
                flat = grad_mag.view(B, -1)  # [B, HW]

                k1 = max(1, int(topk_ratio1 * H * W))
                _, topk_idx1 = torch.topk(flat, k=k1, dim=1)

                mask1_flat = torch.zeros_like(flat)
                mask1_flat.scatter_(1, topk_idx1, 1.0)
                mask1_binary = mask1_flat.view(B, 1, H, W)

                k2 = max(1, int(topk_ratio2 * k1))
                topk_vals1 = flat.gather(1, topk_idx1)
                _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
                final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)

                mask2_flat = torch.zeros_like(flat)
                mask2_flat.scatter_(1, final_idx, 1.0)
                mask2_binary = mask2_flat.view(B, 1, H, W)

            # 平滑
            smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)

            final_mask1 = (smoothed_mask1 > 0).float()
            final_mask2 = (smoothed_mask2 > 0).float()

            assert (final_mask2 <= final_mask1).all(), "mask2 不是 mask1 的子集！"

            return final_mask1, final_mask2

        def compute_grad_soft_masks_nested(grads, topk_ratio1=0.3, topk_ratio2=0.1, blur_kernel=7, blur_sigma=3):
            """
            生成两个嵌套的梯度mask（mask2 ⊆ mask1），并进行高斯平滑。
            """
            B, C, H, W = grads.shape
            grad_mag = grads.abs().mean(dim=1, keepdim=True)
            flat = grad_mag.view(B, -1)

            k1 = max(1, int(topk_ratio1 * H * W))
            _, topk_idx1 = torch.topk(flat, k=k1, dim=1)

            mask1_flat = torch.zeros_like(flat)
            mask1_flat.scatter_(1, topk_idx1, 1.0)
            mask1_binary = mask1_flat.view(B, 1, H, W)

            k2 = max(1, int(topk_ratio2 * k1))
            topk_vals1 = flat.gather(1, topk_idx1)
            _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
            final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)

            mask2_flat = torch.zeros_like(flat)
            mask2_flat.scatter_(1, final_idx, 1.0)
            mask2_binary = mask2_flat.view(B, 1, H, W)

            smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)

            final_mask1 = (smoothed_mask1 > 0).float()
            final_mask2 = (smoothed_mask2 > 0).float()

            assert (final_mask2 <= final_mask1).all(), "mask2 不是 mask1 的子集！"

            return final_mask1, final_mask2

        import torch
        # 假设你已有 gaussian_blur(tensor, kernel_size, sigma)

        def compute_grad_soft_masks_nested_channelwise(
                grads, topk_ratio=0.3, binary_ratio=0.15, blur_kernel=3, blur_sigma=1
        ):
            """
            Select top-k pixels per channel (by |grad| magnitude), then take the union
            across channels (max over channel) to get a spatial mask.
            Output two nested masks (mask2 ⊆ mask1), with Gaussian smoothing and binarization.

            Args:
                grads: Gradient tensor of shape (B, C, H, W)
                topk_ratio: Proportion of pixels to keep per channel (relative to H*W)
                binary_ratio: Threshold for binarization after smoothing
                blur_kernel: Kernel size for Gaussian blur
                blur_sigma: Standard deviation for Gaussian blur

            Returns:
                final_mask: Binary mask of shape (B, 1, H, W)
                smoothed_mask: Soft mask of shape (B, 1, H, W), before hard thresholding
            """
            B, C, H, W = grads.shape
            # Compute per-pixel magnitude
            grad_mag = grads.abs()  # (B, C, H, W)
            flat = grad_mag.view(B, C, -1)  # (B, C, HW)

            HW = H * W
            k = max(1, min(HW, int(topk_ratio * HW)))
            # Select top-k per channel
            _, topk_idx = torch.topk(flat, k=k, dim=2)  # (B, C, k)

            # Build per-channel masks
            mask_flat_ch = torch.zeros_like(flat)  # (B, C, HW)
            mask_flat_ch.scatter_(2, topk_idx, 1.0)
            mask_ch = mask_flat_ch.view(B, C, H, W)  # (B, C, H, W)

            # Union across channels: pixel is selected if any channel selects it
            mask_binary = mask_ch.max(dim=1, keepdim=True).values.float()  # (B, 1, H, W)

            # Gaussian smoothing + binarization
            smoothed_mask = gaussian_blur(mask_binary, kernel_size=blur_kernel, sigma=blur_sigma)
            final_mask = (smoothed_mask > binary_ratio).float()

            return final_mask, smoothed_mask

        def compute_thgrad_mask(grads, p95_abs, blur_kernel=7, blur_sigma=3):
            """
            用固定阈值 p95_abs 生成梯度掩码并高斯平滑。
            逻辑：先对通道取平均，再判断是否 > p95_abs。
            返回：smoothed_mask（软掩码）
            """
            B, C, H, W = grads.shape
            if blur_kernel % 2 == 0:
                blur_kernel += 1

            thr = torch.as_tensor(p95_abs, device=grads.device, dtype=grads.dtype).view(1, 1, 1, 1)

            # 通道均值
            grad_mean = grads.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)

            # 二值掩码：均值是否超过阈值
            mask_binary = (grad_mean >= thr).float()

            # 高斯平滑 → soft mask
            smoothed_mask = gaussian_blur(mask_binary, kernel_size=blur_kernel, sigma=blur_sigma)

            return smoothed_mask

        def compute_topp_masks_nested(grads, topp_ratio1=0.9, topp_ratio2=0.5, blur_kernel=7, blur_sigma=3):
            """
            Compute two nested gradient-based masks using Top-P (nucleus) sampling and apply Gaussian smoothing.
            """
            B, C, H, W = grads.shape
            grad_mag = grads.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
            flat = grad_mag.view(B, -1)  # (B, H*W)

            # Top-P mask1
            sorted_vals1, sorted_idx1 = torch.sort(flat, dim=1, descending=True)
            cum_sum1 = torch.cumsum(sorted_vals1, dim=1)
            total_sum1 = cum_sum1[:, -1].unsqueeze(1)
            topp_mask1 = (cum_sum1 / total_sum1) <= topp_ratio1
            topp_mask1[:, 0] = 1  # Ensure at least one pixel

            mask1_flat = torch.zeros_like(flat)
            for b in range(B):
                idx_b = sorted_idx1[b][topp_mask1[b]]
                mask1_flat[b, idx_b] = 1.0
            mask1_binary = mask1_flat.view(B, 1, H, W)

            # Top-P mask2 within mask1 region
            flat_masked2 = flat * mask1_flat
            sorted_vals2, sorted_idx2 = torch.sort(flat_masked2, dim=1, descending=True)
            cum_sum2 = torch.cumsum(sorted_vals2, dim=1)
            total_sum2 = cum_sum2[:, -1].unsqueeze(1)
            topp_mask2 = (cum_sum2 / total_sum2) <= topp_ratio2
            topp_mask2[:, 0] = 1

            mask2_flat = torch.zeros_like(flat)
            for b in range(B):
                idx_b = sorted_idx2[b][topp_mask2[b]]
                mask2_flat[b, idx_b] = 1.0
            mask2_binary = mask2_flat.view(B, 1, H, W)

            # Apply Gaussian smoothing
            smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)

            final_mask1 = (smoothed_mask1 > 0).float()

            return final_mask1, smoothed_mask1

        def compute_energy_mask_sum_based(grads, eta=0.3, blur=False, blur_kernel=7, blur_sigma=3):
            """
            Compute a gradient-based binary mask using cumulative energy ratio,
            based on total gradient sum (not per-channel mean).

            Args:
                grads: Tensor of shape (B, C, H, W)
                eta: energy ratio threshold (e.g., 0.3 means keep pixels contributing to 30% of total gradient energy)
                blur: whether to apply Gaussian blur
            Returns:
                Binary mask: (B, 1, H, W)
            """
            B, C, H, W = grads.shape
            grad_sum = grads.abs().sum(dim=1, keepdim=False)  # (B, H, W)

            flat = grad_sum.view(B, -1)  # (B, H*W)
            sorted_vals, sorted_idx = torch.sort(flat, dim=1, descending=True)
            cum_sum = torch.cumsum(sorted_vals, dim=1)
            total_sum = cum_sum[:, -1].unsqueeze(1)

            # Determine threshold mask
            energy_mask = (cum_sum / total_sum) <= eta
            energy_mask[:, 0] = 1  # ensure at least one pixel

            # Create binary mask
            mask_flat = torch.zeros_like(flat)
            for b in range(B):
                idx_b = sorted_idx[b][energy_mask[b]]
                mask_flat[b, idx_b] = 1.0
            mask_binary = mask_flat.view(B, 1, H, W)

            # Optional: smooth to get soft mask
            if blur:
                mask_binary = gaussian_blur(mask_binary, kernel_size=blur_kernel, sigma=blur_sigma)

            return mask_binary

        def compute_energy_mask_sum_based(grads, eta=0.3, blur=True, blur_kernel=7, blur_sigma=3):
            B, C, H, W = grads.shape
            grad_sum = grads.abs().sum(dim=1, keepdim=False)  # (B, H, W)

            flat = grad_sum.view(B, -1)  # (B, H*W)
            sorted_vals, sorted_idx = torch.sort(flat, dim=1, descending=True)
            cum_sum = torch.cumsum(sorted_vals, dim=1)
            total_sum = cum_sum[:, -1].unsqueeze(1)

            # Determine threshold mask
            energy_mask = (cum_sum / total_sum) <= eta
            energy_mask[:, 0] = 1  # ensure at least one pixel

            # Create binary mask
            mask_flat = torch.zeros_like(flat)
            for b in range(B):
                idx_b = sorted_idx[b][energy_mask[b]]
                mask_flat[b, idx_b] = 1.0
            mask_binary = mask_flat.view(B, 1, H, W)

            # 打印每个 batch 中 1 的个数
            ones_per_batch = mask_binary.sum(dim=[1, 2, 3])
            print("每个 batch 中 1 的数量:", ones_per_batch.tolist())

            if blur:
                mask_binary = gaussian_blur(mask_binary, kernel_size=blur_kernel, sigma=blur_sigma)

            return mask_binary

        def compute_thgrad_mask(grads, p95_abs, blur_kernel=3, blur_sigma=1):
            """
            用固定阈值 p95_abs 生成梯度掩码并高斯平滑。
            返回：final_mask（二值）、smoothed_mask（软）。
            """
            B, C, H, W = grads.shape
            if blur_kernel % 2 == 0:
                blur_kernel += 1

            thr = torch.as_tensor(p95_abs, device=grads.device, dtype=grads.dtype).view(1, 1, 1, 1)
            mask_ch = (grads.abs() >= thr).float()
            mask_binary = mask_ch.amax(dim=1, keepdim=True)
            smoothed_mask = gaussian_blur(mask_binary, kernel_size=blur_kernel, sigma=blur_sigma)
            # final_mask = (smoothed_mask > 0).float()
            return smoothed_mask



        def compute_mask_topk(grads, topk_rate=0.05, blur_kernel=7, blur_sigma=3):
            """
            仅使用 top-k 掩码：对每个样本，按 topk_rate 求分位阈值并二值化，再高斯平滑。
            返回：smoothed_mask（B,1,H,W）
            """
            assert grads.dim() == 4, "grads should be [B,C,H,W]"
            B, C, H, W = grads.shape
            if blur_kernel % 2 == 0:
                blur_kernel += 1

            # 通道取最大，作为像素分数
            score = grads.abs().mean(dim=1)
            score_flat = score.view(B, -1)  # [B,HW]

            # 每个样本的 top-k 阈值（例如 5% -> q=0.95）
            q = 1.0 - float(topk_rate)
            thr_topk = torch.quantile(score_flat, q, dim=1, keepdim=True).view(B, 1, 1)  # [B,1,1]

            # 二值掩码 -> 平滑
            mask = (score >= thr_topk).float().unsqueeze(1)  # [B,1,H,W]
            smoothed_mask = gaussian_blur(mask, kernel_size=blur_kernel, sigma=blur_sigma)
            return smoothed_mask


        pred_xstart_steps=[]
        outmean_steps=[]
        guidedmean_steps=[]
        z_o_steps=[]
        z_t_mask_steps = []
        c_z_t_mask_steps = []
        x_t_mask_steps = []
        c_x_t_mask_steps = []
        grads_steps=[]
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]
        x_t = img.clone()
        grad_mask_x_t = None

        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # 保存当前状态
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # === 关键修改1：使用掩码混合预测结果和原始图像 ===
            pred_xstart = out["pred_xstart"].detach()
            pred_xstart_steps.append(pred_xstart.detach())
            if grad_mask_x_t is not None:
                # x_t = pred_xstart
                x_t = grad_mask_x_t * pred_xstart + (1 - grad_mask_x_t) * img.clone()

            # if grad_mask_x_t is not None:
            #     # x_t = pred_xstart
            #     x_t_candidate = grad_mask_x_t * pred_xstart + (1 - grad_mask_x_t) * img.clone()
            #     # 新增：对 x_t 应用 EMA momentum（x_t 的初始化已是 img）
            #     x_t = x_t_momentum_beta * x_t + (1.0 - x_t_momentum_beta) * x_t_candidate

            # 计算梯度（如果需要）
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            nonzero_mask = (t_val != 0).float().view(-1, *([1] * (len(shape) - 1)))
            grads = None

            if (class_grad_fn is not None) and (jdx < guided_iterations):
                grads_class = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t

                # # === 新增：对分类器梯度使用 EMA momentum（与 mask 一致的思路） ===
                # if prev_grads_class_ema is None:
                #     ema_grads_class = grads_class
                # else:
                #     ema_grads_class = grad_momentum_beta * prev_grads_class_ema + (1.0 - grad_momentum_beta) * grads_class
                # grads_class = ema_grads_class
                # prev_grads_class_ema = ema_grads_class.detach()
                # # ============================================================

                # print("steps:", i)
                g = grads_class.detach()

                # 统一取绝对值
                abs_vals = g.abs()

                # if abs_vals.numel() > 0:
                #     p99 = torch.quantile(abs_vals, 0.99)
                #     p95 = torch.quantile(abs_vals, 0.95)
                #     print(f"Abs -> p99={p99.item():.6f}, p95={p95.item():.6f}")
                # else:
                #     print("Abs -> no values")

                # abs_grad = grads_class.abs()
                # if abs_grad.dim() == 4:
                #     # [B,C,H,W] -> 通道平均 -> [B,H,W] -> 展平到一维
                #     abs_grad = abs_grad.mean(dim=1).flatten()
                # elif abs_grad.dim() == 3:
                #     # [C,H,W] -> 通道平均 -> [H,W] -> 展平
                #     abs_grad = abs_grad.mean(dim=0).flatten()
                # else:
                #     # 其它情况就直接展平
                #     abs_grad = abs_grad.flatten()


                if i == 59 or p_abs is None:
                    p_abs = torch.quantile(abs_vals, topk_ratio).item()
                # elif i < 59:
                #     q = torch.quantile(abs_vals, topk_ratio).item()
                #     if q > p_abs:
                #         p_abs = q
                #         print(f"Abs -> p95={p_abs:.6f}")

                # grad_mask_x_t = compute_thgrad_mask(grads_class, p_abs)
                # grad_mask_x_t = compute_energy_mask_sum_based(grads_class,topk_ratio)
                # grad_mask, grad_mask_x_t = compute_grad_soft_masks_nested_channelwise(grads_class,topk_ratio=topk_ratio,binary_ratio=x_t_rate)
                grad_mask, grad_mask_x_t =compute_grad_soft_masks_nested2(grads_class,topk_ratio1=topk_ratio,topk_ratio2=x_t_rate)


                # grad_mask_x_t = compute_thgrad_mask(grads_class,p95_abs,topk_ratio)  # (B,1,H,W)

                # if prev_grad_mask_x_t_ema is None:
                #     ema_mask_x_t = grad_mask_x_t.float()
                # else:
                #     ema_mask_x_t = momentum_beta * prev_grad_mask_x_t_ema + (1.0 - momentum_beta) * grad_mask_x_t.float()
                # grad_mask_x_t = ema_mask_x_t
                # prev_grad_mask_x_t_ema = ema_mask_x_t.detach()
                # grad_mask = (grad_mask_x_t > x_t_rate).float()

                # === 原有梯度裁剪/稳定化 ===
                if grads_clip==True:
                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - 3 * std
                    upper = mean + 3 * std
                    grads_class = grads_class.clamp(min=lower, max=upper)
                else:
                    grads_class = stabilizer(grads_class)


                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,  # 使用混合后的x_t
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class

                outmean_steps.append(out["mean"].detach())
                out["mean"]  = out["mean"] - out["variance"] * grads
                grads_steps.append(grads)
                guidedmean_steps.append(out["mean"].detach())
                #
                z_original = diffusion.q_sample(img, t_val)
                out["mean"] = out["mean"]  * grad_mask + z_original * (1 - grad_mask)

            z_o_steps.append(z_original.detach())
            z_t_mask_steps.append(grad_mask.detach())
            c_z_t_mask_steps.append((1-grad_mask).detach())
            x_t_mask_steps.append(grad_mask_x_t.detach())
            c_x_t_mask_steps.append((1-grad_mask_x_t).detach())

            # 采样下一步
            if not x_t_sampling:
                z_t = out["mean"]
            else:
                noise = torch.randn_like(img) if jdx < len(indices) - 1 else torch.zeros_like(img)
                z_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

        return z_t, x_t_steps, z_t_steps, outmean_steps, z_t_mask_steps,c_z_t_mask_steps, x_t_mask_steps,c_x_t_mask_steps,pred_xstart_steps,z_o_steps,guidedmean_steps,grads_steps

    return p_sample_loop


import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def save_attribution_maps(attribution_maps, save_dir='attribution_maps', prefix='attribution'):
    """
    保存归因图为灰度PNG图像

    Args:
        attribution_maps: 归因图张量 [B, 1, H, W]
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 确保是CPU tensor
    attribution_maps = attribution_maps.cpu()

    batch_size = attribution_maps.shape[0]

    print(f"保存 {batch_size} 张灰度归因图到目录: {save_dir}")

    for i in range(batch_size):
        # 获取第i张归因图 [1, 256, 256]
        single_map = attribution_maps[i]

        # 移除通道维度并转换为numpy [256, 256]
        map_np = single_map.squeeze().numpy()

        # 归一化到 [0, 1] - 保持连续值
        map_np = (map_np - map_np.min()) / (map_np.max() - map_np.min() + 1e-8)

        # 转换为0-255的uint8 - 这会生成256级灰度
        map_uint8 = (map_np * 255).astype(np.uint8)

        # 创建PIL图像 - 模式设置为'L'表示灰度
        img = Image.fromarray(map_uint8, mode='L')

        # 保存为PNG
        filename = f"{prefix}_{i:03d}.png"
        save_path = os.path.join(save_dir, filename)
        img.save(save_path)

        print(f"已保存灰度图: {filename}")

    print("所有灰度归因图保存完成！")


def get_ThDiME_iterative_samplingImageNet(use_sampling=False):
    '''
    Returns DiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      get_attribution_maps=None,
                      get_attribution_kwargs=None,
                      x_t_sampling=use_sampling,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      MAX_MASK_ACCUM_STEPS=3,
                      mask_class="binary",
                      grads_clip=True,
                      accum_method="max",
                      x_t_rate=0.05
                      ):

        stabilizer = ADAMGradientStabilization(
            beta_1=0.9,
            beta_2=0.999,
            eps=1e-8,
            reset_step=20,
        )

        def compute_grad_soft_masks_nested2(
                grads,
                topk_ratio1=0.3,
                topk_ratio2=0.1,
                blur_kernel=3,
                blur_sigma=1,
                per_channel=True,
        ):
            """
            生成两个嵌套的梯度mask（mask2 ⊆ mask1），并进行高斯平滑。
            如果 per_channel=True，则对每个通道单独取 topk。
            """
            B, C, H, W = grads.shape

            if per_channel:
                # 保持通道维度，不做平均
                grad_mag = grads.abs()  # [B, C, H, W]
                flat = grad_mag.view(B, C, -1)  # [B, C, HW]

                k1 = max(1, int(topk_ratio1 * H * W))
                _, topk_idx1 = torch.topk(flat, k=k1, dim=2)

                mask1_flat = torch.zeros_like(flat)
                mask1_flat.scatter_(2, topk_idx1, 1.0)
                mask1_binary = mask1_flat.view(B, C, H, W)

                k2 = max(1, int(topk_ratio2 * k1))
                topk_vals1 = flat.gather(2, topk_idx1)
                _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=2)
                final_idx = topk_idx1.gather(2, topk_idx2_in_topk1)

                mask2_flat = torch.zeros_like(flat)
                mask2_flat.scatter_(2, final_idx, 1.0)
                mask2_binary = mask2_flat.view(B, C, H, W)

            else:
                # 原始逻辑：通道平均
                grad_mag = grads.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
                flat = grad_mag.view(B, -1)  # [B, HW]

                k1 = max(1, int(topk_ratio1 * H * W))
                _, topk_idx1 = torch.topk(flat, k=k1, dim=1)

                mask1_flat = torch.zeros_like(flat)
                mask1_flat.scatter_(1, topk_idx1, 1.0)
                mask1_binary = mask1_flat.view(B, 1, H, W)

                k2 = max(1, int(topk_ratio2 * k1))
                topk_vals1 = flat.gather(1, topk_idx1)
                _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
                final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)

                mask2_flat = torch.zeros_like(flat)
                mask2_flat.scatter_(1, final_idx, 1.0)
                mask2_binary = mask2_flat.view(B, 1, H, W)

            # 平滑
            smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)

            final_mask1 = (smoothed_mask1 > 0).float()
            final_mask2 = (smoothed_mask2 > 0).float()

            assert (final_mask2 <= final_mask1).all(), "mask2 不是 mask1 的子集！"

            return final_mask1,final_mask2




        def compute_grad_mask(grads, topk_ratio=0.3, blur_kernel=7, blur_sigma=3):
            """
            Generate a gradient mask and its smoothed version.
            """
            B, C, H, W = grads.shape
            grad_mag = grads.abs().mean(dim=1, keepdim=True)
            flat = grad_mag.view(B, -1)
            k = max(1, int(topk_ratio * H * W))
            _, topk_idx = torch.topk(flat, k=k, dim=1)

            mask_flat = torch.zeros_like(flat)
            mask_flat.scatter_(1, topk_idx, 1.0)
            mask_binary = mask_flat.view(B, 1, H, W)

            smoothed_mask = gaussian_blur(mask_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            final_mask = (smoothed_mask > 0).float()
            return final_mask, smoothed_mask

        def compute_grad_soft_masks_nested(grads, topk_ratio1=0.3, topk_ratio2=0.1, blur_kernel=7, blur_sigma=3):
            """
            生成两个嵌套的梯度mask（mask2 ⊆ mask1），并进行高斯平滑。
            """
            B, C, H, W = grads.shape
            grad_mag = grads.abs().mean(dim=1, keepdim=True)
            flat = grad_mag.view(B, -1)

            k1 = max(1, int(topk_ratio1 * H * W))
            _, topk_idx1 = torch.topk(flat, k=k1, dim=1)

            mask1_flat = torch.zeros_like(flat)
            mask1_flat.scatter_(1, topk_idx1, 1.0)
            mask1_binary = mask1_flat.view(B, 1, H, W)

            k2 = max(1, int(topk_ratio2 * k1))
            topk_vals1 = flat.gather(1, topk_idx1)
            _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
            final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)

            mask2_flat = torch.zeros_like(flat)
            mask2_flat.scatter_(1, final_idx, 1.0)
            mask2_binary = mask2_flat.view(B, 1, H, W)

            smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)

            final_mask1 = (smoothed_mask1 > 0).float()
            final_mask2 = (smoothed_mask2 > 0).float()

            assert (final_mask2 <= final_mask1).all(), "mask2 不是 mask1 的子集！"

            return final_mask1, final_mask2




        def compute_grad_soft_masks_nested_channelwise(
                grads,
                topk_ratio=0.3,
                binary_ratio=0.15,
                blur_kernel=3,
                blur_sigma=1,
                reduce="channel",  # "channel" | "mean"
        ):
            """
            根据 reduce 选择两种方式生成空间 mask（最后都变成单通道）：
              - "channel": 各通道各自 top-k，之后对通道做并集（max）
              - "mean":    在通道上求平均后，再对单通道做 top-k

            返回:
                final_mask: (B, 1, H, W) 经过平滑后阈值化的二值 mask
                smoothed_mask: (B, 1, H, W) 平滑后的软 mask（阈值化前）
            """
            assert gaussian_blur is not None, "请传入 gaussian_blur 函数"
            B, C, H, W = grads.shape
            HW = H * W
            k = max(1, min(HW, int(topk_ratio * HW)))

            if reduce == "channel":
                # (B, C, HW) 上按每个通道取 top-k → 并集
                flat = grads.abs().view(B, C, HW)  # (B, C, HW)
                _, topk_idx = torch.topk(flat, k=k, dim=2)  # (B, C, k)

                mask_flat_ch = torch.zeros_like(flat)  # (B, C, HW)
                mask_flat_ch.scatter_(2, topk_idx, 1.0)
                mask_ch = mask_flat_ch.view(B, C, H, W)  # (B, C, H, W)

                # 通道并集 → 单通道空间 mask
                mask_binary = mask_ch.max(dim=1, keepdim=True).values.float()  # (B, 1, H, W)

            elif reduce == "mean":
                # 先在通道求平均 → (B,1,H,W)，再做 top-k
                grad_mean = grads.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
                flat = grad_mean.view(B, 1, HW)  # (B, 1, HW)
                _, topk_idx = torch.topk(flat, k=k, dim=2)  # (B, 1, k)

                mask_flat = torch.zeros_like(flat)  # (B, 1, HW)
                mask_flat.scatter_(2, topk_idx, 1.0)
                mask_binary = mask_flat.view(B, 1, H, W).float()  # (B, 1, H, W)

            else:
                raise ValueError('reduce 应为 "channel" 或 "mean"')

            # 高斯平滑 + 二值化
            smoothed_mask = gaussian_blur(mask_binary, kernel_size=blur_kernel, sigma=blur_sigma)
            # final_mask = (smoothed_mask > binary_ratio).float()

            return smoothed_mask

        def compute_thgrad_mask(grads, p95_abs, blur_kernel=7, blur_sigma=3):
            """
            用固定阈值 p95_abs 生成梯度掩码并高斯平滑。
            逻辑：先对通道取平均，再判断是否 > p95_abs。
            返回：smoothed_mask（软掩码）
            """
            B, C, H, W = grads.shape
            if blur_kernel % 2 == 0:
                blur_kernel += 1

            thr = torch.as_tensor(p95_abs, device=grads.device, dtype=grads.dtype).view(1, 1, 1, 1)

            # 通道均值
            grad_mean = grads.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)

            # 二值掩码：均值是否超过阈值
            mask_binary = (grad_mean >= thr).float()

            # 高斯平滑 → soft mask
            smoothed_mask = gaussian_blur(mask_binary, kernel_size=blur_kernel, sigma=blur_sigma)

            return smoothed_mask

        def compute_topp_masks_nested(grads, topp_ratio1=0.9, topp_ratio2=0.5, blur_kernel=7, blur_sigma=3):
            """
            Compute two nested gradient-based masks using Top-P (nucleus) sampling and apply Gaussian smoothing.
            """
            B, C, H, W = grads.shape
            grad_mag = grads.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
            flat = grad_mag.view(B, -1)  # (B, H*W)

            # Top-P mask1
            sorted_vals1, sorted_idx1 = torch.sort(flat, dim=1, descending=True)
            cum_sum1 = torch.cumsum(sorted_vals1, dim=1)
            total_sum1 = cum_sum1[:, -1].unsqueeze(1)
            topp_mask1 = (cum_sum1 / total_sum1) <= topp_ratio1
            topp_mask1[:, 0] = 1  # Ensure at least one pixel

            mask1_flat = torch.zeros_like(flat)
            for b in range(B):
                idx_b = sorted_idx1[b][topp_mask1[b]]
                mask1_flat[b, idx_b] = 1.0
            mask1_binary = mask1_flat.view(B, 1, H, W)

            # Top-P mask2 within mask1 region
            flat_masked2 = flat * mask1_flat
            sorted_vals2, sorted_idx2 = torch.sort(flat_masked2, dim=1, descending=True)
            cum_sum2 = torch.cumsum(sorted_vals2, dim=1)
            total_sum2 = cum_sum2[:, -1].unsqueeze(1)
            topp_mask2 = (cum_sum2 / total_sum2) <= topp_ratio2
            topp_mask2[:, 0] = 1

            mask2_flat = torch.zeros_like(flat)
            for b in range(B):
                idx_b = sorted_idx2[b][topp_mask2[b]]
                mask2_flat[b, idx_b] = 1.0
            mask2_binary = mask2_flat.view(B, 1, H, W)

            # Apply Gaussian smoothing
            smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)

            final_mask1 = (smoothed_mask1 > 0).float()

            return final_mask1, smoothed_mask1

        def compute_energy_mask_sum_based(grads, eta=0.3, blur=False, blur_kernel=7, blur_sigma=3):
            """
            Compute a gradient-based binary mask using cumulative energy ratio,
            based on total gradient sum (not per-channel mean).

            Args:
                grads: Tensor of shape (B, C, H, W)
                eta: energy ratio threshold (e.g., 0.3 means keep pixels contributing to 30% of total gradient energy)
                blur: whether to apply Gaussian blur
            Returns:
                Binary mask: (B, 1, H, W)
            """
            B, C, H, W = grads.shape
            grad_sum = grads.abs().sum(dim=1, keepdim=False)  # (B, H, W)

            flat = grad_sum.view(B, -1)  # (B, H*W)
            sorted_vals, sorted_idx = torch.sort(flat, dim=1, descending=True)
            cum_sum = torch.cumsum(sorted_vals, dim=1)
            total_sum = cum_sum[:, -1].unsqueeze(1)

            # Determine threshold mask
            energy_mask = (cum_sum / total_sum) <= eta
            energy_mask[:, 0] = 1  # ensure at least one pixel

            # Create binary mask
            mask_flat = torch.zeros_like(flat)
            for b in range(B):
                idx_b = sorted_idx[b][energy_mask[b]]
                mask_flat[b, idx_b] = 1.0
            mask_binary = mask_flat.view(B, 1, H, W)

            # Optional: smooth to get soft mask
            if blur:
                mask_binary = gaussian_blur(mask_binary, kernel_size=blur_kernel, sigma=blur_sigma)

            return mask_binary

        def compute_energy_mask_sum_based(grads, eta=0.3, blur=True, blur_kernel=7, blur_sigma=3):
            B, C, H, W = grads.shape
            grad_sum = grads.abs().sum(dim=1, keepdim=False)  # (B, H, W)

            flat = grad_sum.view(B, -1)  # (B, H*W)
            sorted_vals, sorted_idx = torch.sort(flat, dim=1, descending=True)
            cum_sum = torch.cumsum(sorted_vals, dim=1)
            total_sum = cum_sum[:, -1].unsqueeze(1)

            # Determine threshold mask
            energy_mask = (cum_sum / total_sum) <= eta
            energy_mask[:, 0] = 1  # ensure at least one pixel

            # Create binary mask
            mask_flat = torch.zeros_like(flat)
            for b in range(B):
                idx_b = sorted_idx[b][energy_mask[b]]
                mask_flat[b, idx_b] = 1.0
            mask_binary = mask_flat.view(B, 1, H, W)

            # 打印每个 batch 中 1 的个数
            ones_per_batch = mask_binary.sum(dim=[1, 2, 3])
            print("每个 batch 中 1 的数量:", ones_per_batch.tolist())

            if blur:
                mask_binary = gaussian_blur(mask_binary, kernel_size=blur_kernel, sigma=blur_sigma)

            return mask_binary

        def compute_thgrad_mask(grads, p95_abs, blur_kernel=3, blur_sigma=1):
            """
            用固定阈值 p95_abs 生成梯度掩码并高斯平滑。
            返回：final_mask（二值）、smoothed_mask（软）。
            """
            B, C, H, W = grads.shape
            if blur_kernel % 2 == 0:
                blur_kernel += 1

            thr = torch.as_tensor(p95_abs, device=grads.device, dtype=grads.dtype).view(1, 1, 1, 1)
            mask_ch = (grads.abs() >= thr).float()
            mask_binary = mask_ch.amax(dim=1, keepdim=True)
            smoothed_mask = gaussian_blur(mask_binary, kernel_size=blur_kernel, sigma=blur_sigma)
            # final_mask = (smoothed_mask > 0).float()
            return smoothed_mask



        def compute_mask_topk(grads, topk_rate=0.05, blur_kernel=7, blur_sigma=3):
            """
            仅使用 top-k 掩码：对每个样本，按 topk_rate 求分位阈值并二值化，再高斯平滑。
            返回：smoothed_mask（B,1,H,W）
            """
            assert grads.dim() == 4, "grads should be [B,C,H,W]"
            B, C, H, W = grads.shape
            if blur_kernel % 2 == 0:
                blur_kernel += 1

            # 通道取最大，作为像素分数
            score = grads.abs().mean(dim=1)
            score_flat = score.view(B, -1)  # [B,HW]

            # 每个样本的 top-k 阈值（例如 5% -> q=0.95）
            q = 1.0 - float(topk_rate)
            thr_topk = torch.quantile(score_flat, q, dim=1, keepdim=True).view(B, 1, 1)  # [B,1,1]

            # 二值掩码 -> 平滑
            mask = (score >= thr_topk).float().unsqueeze(1)  # [B,1,H,W]
            smoothed_mask = gaussian_blur(mask, kernel_size=blur_kernel, sigma=blur_sigma)
            return smoothed_mask


        pred_xstart_steps=[]
        outmean_steps=[]
        guidedmean_steps=[]
        z_o_steps=[]
        z_t_mask_steps = []
        c_z_t_mask_steps = []
        x_t_mask_steps = []
        c_x_t_mask_steps = []
        grads_steps=[]
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]
        x_t = img.clone()


        grad_mask_x_t = None
        # ========== 新增：momentum 配置（同时用于 grad_mask 与 grad_mask_x_t） ==========
        momentum_beta = 0.9
        prev_grad_mask_x_t_ema = None   # 软掩码的 EMA 缓存

        prev_grad_mask_x_t_ema = None  # 软掩码的 EMA 缓存
        # --- 新增：x_t 的 EMA momentum 系数 ---
        x_t_momentum_beta = 0.9

        # x_t_for_attribution_map = (x_t.clamp(-1, 1) + 1) / 2

        # attribution_maps = get_attribution_maps(x_t_for_attribution_map,**get_attribution_kwargs)
        # print(attribution_maps.shape)
        # save_attribution_maps(attribution_maps,"/work3/chagu/extraspace/cvpr/ImageNet/mini_sz/4_0.01_0.15_channel_mom")


        # ======================================================================

        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # 保存当前状态
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # === 关键修改1：使用掩码混合预测结果和原始图像 ===
            pred_xstart = out["pred_xstart"].detach()
            pred_xstart_steps.append(pred_xstart.detach())
            if grad_mask_x_t is not None:

                x_t = grad_mask_x_t * pred_xstart + (1 - grad_mask_x_t) * img.clone()


            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            nonzero_mask = (t_val != 0).float().view(-1, *([1] * (len(shape) - 1)))
            grads = None
            x_t_for_classifier = (x_t.clamp(-1, 1) + 1) / 2
            if (class_grad_fn is not None) and (jdx < guided_iterations):
                grads_class = class_grad_fn(x_t=x_t_for_classifier, **class_grad_kwargs) / alpha_t
                grad_mask, grad_mask_x_t=compute_grad_soft_masks_nested2(grads_class,topk_ratio2=topk_ratio,topk_ratio1=x_t_rate)
                # grad_mask_x_t = compute_grad_soft_masks_nested_channelwise(grads_class,topk_ratio=topk_ratio)
                # grad_mask = (grad_mask_x_t>0).float()
                # print("whatb fuck!!! ")
                # if i >= 39:  # <= 只在 59→39 这段区间更新
                #     grad_mask_x_t = compute_grad_soft_masks_nested_channelwise(
                #         grads_class,
                #         topk_ratio=topk_ratio,
                #         binary_ratio=x_t_rate
                #     )
                    # if prev_grad_mask_x_t_ema is None:
                    #     ema_mask_x_t = grad_mask_x_t.float()
                    # else:
                    #     ema_mask_x_t = momentum_beta * prev_grad_mask_x_t_ema + (
                    #                 1.0 - momentum_beta) * grad_mask_x_t.float()
                    # grad_mask_x_t = ema_mask_x_t
                    # prev_grad_mask_x_t_ema = ema_mask_x_t.detach()
                    # grad_mask = (grad_mask_x_t > x_t_rate).float()

                if grads_clip==True:
                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - 3 * std
                    upper = mean + 3 * std
                    grads_class = grads_class.clamp(min=lower, max=upper)
                else:
                    grads_class = stabilizer(grads_class)


                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,  # 使用混合后的x_t
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class


                outmean_steps.append(out["mean"].detach())
                out["mean"]  = out["mean"] - out["variance"] * grads
                grads_steps.append(grads)
                guidedmean_steps.append(out["mean"].detach())
                #
                z_original = diffusion.q_sample(img, t_val)
                out["mean"] = out["mean"]  * grad_mask + z_original * (1 - grad_mask)

            z_o_steps.append(z_original.detach())
            z_t_mask_steps.append(grad_mask.detach())
            c_z_t_mask_steps.append((1-grad_mask).detach())
            x_t_mask_steps.append(grad_mask_x_t.detach())
            c_x_t_mask_steps.append((1-grad_mask_x_t).detach())


            z_t = out["mean"]

        return z_t, x_t_steps, z_t_steps, outmean_steps, z_t_mask_steps,c_z_t_mask_steps, x_t_mask_steps,c_x_t_mask_steps,pred_xstart_steps,z_o_steps,guidedmean_steps,grads_steps

    return p_sample_loop


@torch.no_grad()
def generate_mask(x1, x2, dilation):
    '''
    Extracts a mask by binarizing the difference between
    denoised image at time-step t and original input.
    We generate the mask similar to ACE.

    :x1: denoised image at time-step t
    :x2: original input image
    :dilation: dilation parameters
    '''
    assert (dilation % 2) == 1, 'dilation must be an odd number'
    x1 = (x1 + 1) / 2
    x2 = (x2 + 1) / 2
    mask =  (x1 - x2).abs().sum(dim=1, keepdim=True)
    mask = mask / mask.view(mask.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    dil_mask = F.max_pool2d(mask, dilation, stride=1, padding=(dilation - 1) // 2)
    return mask, dil_mask
def get_FastDiME_Mask_iterative_samplingHQ(use_sampling=False):
    '''
    Returns DiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=use_sampling,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      MAX_MASK_ACCUM_STEPS=3,
                      mask_class="binary",
                      grads_clip=True,
                      accum_method="max",
                      x_t_rate=0.05
                      ):
        fixed_epsilon = torch.randn_like(img)


        def compute_grad_soft_masks_nested2(
                grads,
                topk_ratio1=0.3,
                topk_ratio2=0.1,
                blur_kernel=5,
                blur_sigma=3,
                per_channel=False,
        ):
            """
            生成两个嵌套的梯度mask（mask2 ⊆ mask1），并进行高斯平滑。
            如果 per_channel=True，则对每个通道单独取 topk。
            """
            B, C, H, W = grads.shape

            if per_channel:
                # 保持通道维度，不做平均
                grad_mag = grads.abs()  # [B, C, H, W]
                flat = grad_mag.view(B, C, -1)  # [B, C, HW]

                k1 = max(1, int(topk_ratio1 * H * W))
                _, topk_idx1 = torch.topk(flat, k=k1, dim=2)

                mask1_flat = torch.zeros_like(flat)
                mask1_flat.scatter_(2, topk_idx1, 1.0)
                mask1_binary = mask1_flat.view(B, C, H, W)

                k2 = max(1, int(topk_ratio2 * k1))
                topk_vals1 = flat.gather(2, topk_idx1)
                _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=2)
                final_idx = topk_idx1.gather(2, topk_idx2_in_topk1)

                mask2_flat = torch.zeros_like(flat)
                mask2_flat.scatter_(2, final_idx, 1.0)
                mask2_binary = mask2_flat.view(B, C, H, W)

            else:
                # 原始逻辑：通道平均
                grad_mag = grads.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
                flat = grad_mag.view(B, -1)  # [B, HW]

                k1 = max(1, int(topk_ratio1 * H * W))
                _, topk_idx1 = torch.topk(flat, k=k1, dim=1)

                mask1_flat = torch.zeros_like(flat)
                mask1_flat.scatter_(1, topk_idx1, 1.0)
                mask1_binary = mask1_flat.view(B, 1, H, W)

                k2 = max(1, int(topk_ratio2 * k1))
                topk_vals1 = flat.gather(1, topk_idx1)
                _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
                final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)

                mask2_flat = torch.zeros_like(flat)
                mask2_flat.scatter_(1, final_idx, 1.0)
                mask2_binary = mask2_flat.view(B, 1, H, W)

            # 平滑
            smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)

            final_mask1 = (smoothed_mask1 > 0).float()
            final_mask2 = (smoothed_mask2 > 0).float()

            assert (final_mask2 <= final_mask1).all(), "mask2 不是 mask1 的子集！"

            return final_mask1, smoothed_mask2

        def compute_grad_mask_nested_topk(grads, topk_ratio1=0.3, topk_ratio2=0.1):
            B, C, H, W = grads.shape

            # Compute the mean absolute gradient magnitude across the channel dimension, resulting in shape (B,1,H,W)
            grad_mag = grads.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
            flat = grad_mag.view(B, -1)  # Flatten to (B, H*W)

            # First top-k selection
            k1 = max(1, int(topk_ratio1 * H * W))
            topk_vals1, topk_idx1 = torch.topk(flat, k=k1, dim=1)

            # Create the first mask with ones at top-k positions, zeros elsewhere
            mask1 = torch.zeros_like(flat)
            mask1.scatter_(1, topk_idx1, 1.0)

            # Second top-k selection within the previously selected top-k regions
            k2 = max(1, int(topk_ratio2 * k1))
            topk_vals2, topk_idx2 = torch.topk(topk_vals1, k=k2, dim=1)

            # Map the indices of the second top-k back to the original flattened indices
            final_idx = topk_idx1.gather(1, topk_idx2)

            # Create the second mask with ones at the final selected positions
            mask2 = torch.zeros_like(flat)
            mask2.scatter_(1, final_idx, 1.0)

            # Reshape masks back to (B, 1, H, W) and return
            return mask1.view(B, 1, H, W), mask2.view(B, 1, H, W)

        def compute_grad_soft_masks_nested(grads, topk_ratio1=0.3, topk_ratio2=0.1, blur_kernel=7, blur_sigma=3):
            """
            生成两个嵌套的梯度mask（mask2 ⊆ mask1），并进行高斯平滑。

            Args:
                grads: (B, C, H, W) 的梯度张量。
                topk_ratio1: 第一阶段保留的比例（mask1）。
                topk_ratio2: 在mask1区域内进一步保留的比例（mask2）。
                blur_kernel: 高斯模糊核大小（必须为奇数）。
                blur_sigma: 高斯模糊标准差。

            Returns:
                final_mask1, final_mask2: (B, 1, H, W)，二值mask，mask2 ⊆ mask1。
            """
            B, C, H, W = grads.shape

            # 梯度幅度 (B,1,H,W)，取绝对值后在通道维平均
            grad_mag = grads.abs().mean(dim=1, keepdim=True)
            flat = grad_mag.view(B, -1)  # 展平为 (B, H*W)

            # Top-K 1
            k1 = max(1, int(topk_ratio1 * H * W))
            _, topk_idx1 = torch.topk(flat, k=k1, dim=1)

            mask1_flat = torch.zeros_like(flat)
            mask1_flat.scatter_(1, topk_idx1, 1.0)
            mask1_binary = mask1_flat.view(B, 1, H, W)

            # Top-K 2 within Top-K1
            k2 = max(1, int(topk_ratio2 * k1))
            topk_vals1 = flat.gather(1, topk_idx1)
            _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
            final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)

            mask2_flat = torch.zeros_like(flat)
            mask2_flat.scatter_(1, final_idx, 1.0)
            mask2_binary = mask2_flat.view(B, 1, H, W)

            # 可选平滑处理（保持 mask2 ⊆ mask1）
            smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)

            # 二值化处理
            final_mask1 = (smoothed_mask1 > 0).float()
            final_mask2 = (smoothed_mask2 > 0).float()

            # 可选 assert 检查子集关系
            assert (final_mask2 <= final_mask1).all(), "mask2 不是 mask1 的子集！"

            return final_mask1, final_mask2
        def compute_grad_mask_quantile(grads, topk_ratio=0.3, strong_ratio=0.05):
            """
            Generate two gradient-based binary masks using quantile thresholds.

            Args:
                grads (Tensor): Gradient tensor of shape [B, C, H, W].
                topk_ratio (float): Ratio of the most salient regions to keep in the main mask (e.g., 0.3 = top 30%).
                strong_ratio (float): Ratio for the strong mask, a subset of the topk (e.g., 0.05 = top 5%).

            Returns:
                mask (Tensor): Main binary mask of shape [B, 1, H, W] with topk_ratio preserved.
                strong_mask (Tensor): Stronger binary mask of shape [B, 1, H, W] with strong_ratio preserved.
                                     It is guaranteed that strong_mask ⊆ mask.
            """
            B, C, H, W = grads.shape

            # Compute average gradient magnitude over channels
            grad_mag = torch.abs(grads).mean(dim=1)  # Shape: [B, H, W]

            # Flatten spatial dimensions
            flat = grad_mag.view(B, -1)  # Shape: [B, H*W]

            # Compute quantile thresholds for each sample in the batch
            topk_threshold = torch.quantile(flat, 1 - topk_ratio, dim=1, keepdim=True)  # Shape: [B, 1]
            strong_threshold = torch.quantile(flat, 1 - strong_ratio, dim=1, keepdim=True)  # Shape: [B, 1]

            # Reshape thresholds to match original spatial shape
            topk_threshold = topk_threshold.expand(-1, H * W).view(B, H, W)
            strong_threshold = strong_threshold.expand(-1, H * W).view(B, H, W)

            # Create binary masks by thresholding
            mask = (grad_mag >= topk_threshold).float().unsqueeze(1)  # Shape: [B, 1, H, W]
            strong_mask = (grad_mag >= strong_threshold).float().unsqueeze(1)  # Shape: [B, 1, H, W]

            return mask, strong_mask

        def compute_grad_mask_topk_binary(grads, topk_ratio=0.3):
            B, C, H, W = grads.shape
            grad_magnitude = torch.abs(grads)
            grad_magnitude = grad_magnitude.mean(dim=1, keepdim=True)
            flat = grad_magnitude.view(B, -1)
            k = max(1, int(topk_ratio * H * W))
            topk_vals, topk_idx = torch.topk(flat, k=k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, topk_idx, 1.0)
            return mask.view(B, 1, H, W)
        def compute_grad_mask_topk_binary_cached(grads, topk_ratio_main=0.3, topk_ratio_xt=0.15):
            B, C, H, W = grads.shape
            grad_magnitude = torch.abs(grads).mean(dim=1, keepdim=True)  # (B, 1, H, W)
            flat = grad_magnitude.view(B, -1)  # (B, H*W)

            k_main = max(1, int(topk_ratio_main * H * W))
            k_xt = max(1, int(topk_ratio_xt * H * W))

            # 排序一次即可
            sorted_vals, sorted_idx = torch.sort(flat, dim=1, descending=True)

            mask_main = torch.zeros_like(flat)
            mask_xt = torch.zeros_like(flat)

            mask_main.scatter_(1, sorted_idx[:, :k_main], 1.0)
            mask_xt.scatter_(1, sorted_idx[:, :k_xt], 1.0)

            return mask_main.view(B, 1, H, W), mask_xt.view(B, 1, H, W)


        def compute_grad_masks_dilation(
                grads,
                topk_ratio1=0.3,
                topk_ratio2=0.1,
                dilate_kernel=5,
                per_channel=False,
        ):
            """
            Generate two nested gradient masks (mask2 ⊆ mask1) and apply morphological dilation.
            """
            B, C, H, W = grads.shape

            # === Step 1: gradient magnitude ===
            grad_mag = grads.abs()
            if not per_channel:
                grad_mag = grad_mag.mean(dim=1, keepdim=True)  # (B, 1, H, W)

            flat = grad_mag.view(B, -1) if not per_channel else grad_mag.view(B, C, -1)
            dim = 2 if per_channel else 1

            # === Step 2: compute top-k masks ===
            def get_topk_mask(flat, k_ratio, dim):
                k = max(1, int(k_ratio * flat.size(-1)))
                _, idx = torch.topk(flat, k=k, dim=dim)
                mask_flat = torch.zeros_like(flat)
                mask_flat.scatter_(dim, idx, 1.0)
                return mask_flat

            mask1_flat = get_topk_mask(flat, topk_ratio1, dim)
            mask1_binary = mask1_flat.view_as(grad_mag)

            # 第二层：只需要再取更高比例的 topk
            mask2_flat = get_topk_mask(flat, topk_ratio1 * topk_ratio2, dim)
            mask2_binary = mask2_flat.view_as(grad_mag)

            # === Step 3: morphological dilation ===
            pad = dilate_kernel // 2
            final_mask1 = F.max_pool2d(mask1_binary.float(), kernel_size=dilate_kernel, stride=1, padding=pad)
            final_mask2 = F.max_pool2d(mask2_binary.float(), kernel_size=dilate_kernel, stride=1, padding=pad)

            assert (final_mask2 <= final_mask1).all(), "mask2 is not a subset of mask1!"

            return final_mask1, final_mask2

        pred_xstart_steps=[]
        outmean_steps=[]
        guidedmean_steps=[]
        z_o_steps=[]
        z_t_mask_steps = []
        c_z_t_mask_steps = []
        x_t_mask_steps = []
        c_x_t_mask_steps = []
        grads_steps=[]
        z_t = diffusion.q_sample(img, t, noise=fixed_epsilon) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]
        x_t = img.clone()
        grad_mask_x_t = None


        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # 保存当前状态

            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # 预测当前步的均值和方差
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )


            pred_xstart = out["pred_xstart"].detach()
            pred_xstart_steps.append(pred_xstart.detach())
            if grad_mask_x_t is not None:
                # x_t = pred_xstart
                x_t = grad_mask_x_t * pred_xstart + (1 - grad_mask_x_t) * img.clone()

            # 计算梯度（如果需要）
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            nonzero_mask = (t_val != 0).float().view(-1, *([1] * (len(shape) - 1)))
            if (class_grad_fn is not None) and (jdx < guided_iterations):
                # === 关键修改2：使用混合后的x_t计算梯度 ===
                grads_class = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t
                # grads_image =  class_grad_fn(x_t=img, **class_grad_kwargs) / alpha_t

                # mask1,mask2 = generate_mask(img,x_t,5)
                #
                # boolmask = (mask2 < 0.15).to(out["mean"].device).float()
                # # grad_mask = grad_mask.to(out["mean"].device).float()
                # grad_mask = 1- boolmask
                # grad_mask_x_t = 1-boolmask
                grad_mask, grad_mask_x_t =compute_grad_masks_dilation(grads_class,topk_ratio, x_t_rate)
                if grads_clip==True:

                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - 3 * std
                    upper = mean + 3 * std
                    grads_class = grads_class.clamp(min=lower, max=upper)

                    # 假设 grads_class 是 torch.Tensor

                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,  # 使用混合后的x_t
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class

                outmean_steps.append(out["mean"].detach())
                out["mean"]  = out["mean"] - out["variance"] * grads
                grads_steps.append(grads)
                guidedmean_steps.append(out["mean"].detach())

                z_original = diffusion.q_sample(img, t_val,noise=fixed_epsilon)
                out["mean"] = out["mean"]  * grad_mask + z_original * (1 - grad_mask)

            z_o_steps.append(z_original.detach())
            z_t_mask_steps.append(grad_mask.detach())
            c_z_t_mask_steps.append((1-grad_mask).detach())
            x_t_mask_steps.append(grad_mask_x_t.detach())
            c_x_t_mask_steps.append((1-grad_mask_x_t).detach())

            # 采样下一步
            if not x_t_sampling:
                z_t = out["mean"]
            else:
                noise = torch.randn_like(img) if jdx < len(indices) - 1 else torch.zeros_like(img)
                z_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

        return z_t, x_t_steps, z_t_steps, outmean_steps, z_t_mask_steps,c_z_t_mask_steps, x_t_mask_steps,c_x_t_mask_steps,pred_xstart_steps,z_o_steps,guidedmean_steps,grads_steps


    return p_sample_loop


def get_FastDiME_Mask_iterative_samplingImageNet(use_sampling=False):
    '''
    Returns DiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=use_sampling,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      MAX_MASK_ACCUM_STEPS=3,
                      mask_class="binary",
                      grads_clip=True,
                      accum_method="max",
                      x_t_rate=0.05
                      ):
        fixed_epsilon = torch.randn_like(img)

        def compute_grad_mask_nested_topk(grads, topk_ratio1=0.3, topk_ratio2=0.1):
            B, C, H, W = grads.shape

            # Compute the mean absolute gradient magnitude across the channel dimension, resulting in shape (B,1,H,W)
            grad_mag = grads.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
            flat = grad_mag.view(B, -1)  # Flatten to (B, H*W)

            # First top-k selection
            k1 = max(1, int(topk_ratio1 * H * W))
            topk_vals1, topk_idx1 = torch.topk(flat, k=k1, dim=1)

            # Create the first mask with ones at top-k positions, zeros elsewhere
            mask1 = torch.zeros_like(flat)
            mask1.scatter_(1, topk_idx1, 1.0)

            # Second top-k selection within the previously selected top-k regions
            k2 = max(1, int(topk_ratio2 * k1))
            topk_vals2, topk_idx2 = torch.topk(topk_vals1, k=k2, dim=1)

            # Map the indices of the second top-k back to the original flattened indices
            final_idx = topk_idx1.gather(1, topk_idx2)

            # Create the second mask with ones at the final selected positions
            mask2 = torch.zeros_like(flat)
            mask2.scatter_(1, final_idx, 1.0)

            # Reshape masks back to (B, 1, H, W) and return
            return mask1.view(B, 1, H, W), mask2.view(B, 1, H, W)

        def compute_grad_soft_masks_nested(grads, topk_ratio1=0.3, topk_ratio2=0.1, blur_kernel=7, blur_sigma=3):
            """
            生成两个嵌套的梯度mask（mask2 ⊆ mask1），并进行高斯平滑。

            Args:
                grads: (B, C, H, W) 的梯度张量。
                topk_ratio1: 第一阶段保留的比例（mask1）。
                topk_ratio2: 在mask1区域内进一步保留的比例（mask2）。
                blur_kernel: 高斯模糊核大小（必须为奇数）。
                blur_sigma: 高斯模糊标准差。

            Returns:
                final_mask1, final_mask2: (B, 1, H, W)，二值mask，mask2 ⊆ mask1。
            """
            B, C, H, W = grads.shape

            # 梯度幅度 (B,1,H,W)，取绝对值后在通道维平均
            grad_mag = grads.abs().mean(dim=1, keepdim=True)
            flat = grad_mag.view(B, -1)  # 展平为 (B, H*W)

            # Top-K 1
            k1 = max(1, int(topk_ratio1 * H * W))
            _, topk_idx1 = torch.topk(flat, k=k1, dim=1)

            mask1_flat = torch.zeros_like(flat)
            mask1_flat.scatter_(1, topk_idx1, 1.0)
            mask1_binary = mask1_flat.view(B, 1, H, W)

            # Top-K 2 within Top-K1
            k2 = max(1, int(topk_ratio2 * k1))
            topk_vals1 = flat.gather(1, topk_idx1)
            _, topk_idx2_in_topk1 = torch.topk(topk_vals1, k=k2, dim=1)
            final_idx = topk_idx1.gather(1, topk_idx2_in_topk1)

            mask2_flat = torch.zeros_like(flat)
            mask2_flat.scatter_(1, final_idx, 1.0)
            mask2_binary = mask2_flat.view(B, 1, H, W)

            # 可选平滑处理（保持 mask2 ⊆ mask1）
            smoothed_mask1 = gaussian_blur(mask1_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)
            smoothed_mask2 = gaussian_blur(mask2_binary.float(), kernel_size=blur_kernel, sigma=blur_sigma)

            # 二值化处理
            final_mask1 = (smoothed_mask1 > 0).float()
            final_mask2 = (smoothed_mask2 > 0).float()

            # 可选 assert 检查子集关系
            assert (final_mask2 <= final_mask1).all(), "mask2 不是 mask1 的子集！"

            return final_mask1, final_mask2
        def compute_grad_mask_quantile(grads, topk_ratio=0.3, strong_ratio=0.05):
            """
            Generate two gradient-based binary masks using quantile thresholds.

            Args:
                grads (Tensor): Gradient tensor of shape [B, C, H, W].
                topk_ratio (float): Ratio of the most salient regions to keep in the main mask (e.g., 0.3 = top 30%).
                strong_ratio (float): Ratio for the strong mask, a subset of the topk (e.g., 0.05 = top 5%).

            Returns:
                mask (Tensor): Main binary mask of shape [B, 1, H, W] with topk_ratio preserved.
                strong_mask (Tensor): Stronger binary mask of shape [B, 1, H, W] with strong_ratio preserved.
                                     It is guaranteed that strong_mask ⊆ mask.
            """
            B, C, H, W = grads.shape

            # Compute average gradient magnitude over channels
            grad_mag = torch.abs(grads).mean(dim=1)  # Shape: [B, H, W]

            # Flatten spatial dimensions
            flat = grad_mag.view(B, -1)  # Shape: [B, H*W]

            # Compute quantile thresholds for each sample in the batch
            topk_threshold = torch.quantile(flat, 1 - topk_ratio, dim=1, keepdim=True)  # Shape: [B, 1]
            strong_threshold = torch.quantile(flat, 1 - strong_ratio, dim=1, keepdim=True)  # Shape: [B, 1]

            # Reshape thresholds to match original spatial shape
            topk_threshold = topk_threshold.expand(-1, H * W).view(B, H, W)
            strong_threshold = strong_threshold.expand(-1, H * W).view(B, H, W)

            # Create binary masks by thresholding
            mask = (grad_mag >= topk_threshold).float().unsqueeze(1)  # Shape: [B, 1, H, W]
            strong_mask = (grad_mag >= strong_threshold).float().unsqueeze(1)  # Shape: [B, 1, H, W]

            return mask, strong_mask

        def compute_grad_mask_topk_binary(grads, topk_ratio=0.3):
            B, C, H, W = grads.shape
            grad_magnitude = torch.abs(grads)
            grad_magnitude = grad_magnitude.mean(dim=1, keepdim=True)
            flat = grad_magnitude.view(B, -1)
            k = max(1, int(topk_ratio * H * W))
            topk_vals, topk_idx = torch.topk(flat, k=k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, topk_idx, 1.0)
            return mask.view(B, 1, H, W)
        def compute_grad_mask_topk_binary_cached(grads, topk_ratio_main=0.3, topk_ratio_xt=0.15):
            B, C, H, W = grads.shape
            grad_magnitude = torch.abs(grads).mean(dim=1, keepdim=True)  # (B, 1, H, W)
            flat = grad_magnitude.view(B, -1)  # (B, H*W)

            k_main = max(1, int(topk_ratio_main * H * W))
            k_xt = max(1, int(topk_ratio_xt * H * W))

            # 排序一次即可
            sorted_vals, sorted_idx = torch.sort(flat, dim=1, descending=True)

            mask_main = torch.zeros_like(flat)
            mask_xt = torch.zeros_like(flat)

            mask_main.scatter_(1, sorted_idx[:, :k_main], 1.0)
            mask_xt.scatter_(1, sorted_idx[:, :k_xt], 1.0)

            return mask_main.view(B, 1, H, W), mask_xt.view(B, 1, H, W)



        pred_xstart_steps=[]
        outmean_steps=[]
        guidedmean_steps=[]
        z_o_steps=[]
        z_t_mask_steps = []
        c_z_t_mask_steps = []
        x_t_mask_steps = []
        c_x_t_mask_steps = []
        grads_steps=[]
        z_t = diffusion.q_sample(img, t, noise=fixed_epsilon) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]
        x_t = img.clone()
        grad_mask_x_t = None


        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # 保存当前状态

            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # 预测当前步的均值和方差
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # === 关键修改1：使用掩码混合预测结果和原始图像 ===
            # 只在掩码区域内使用预测的x0，掩码区域外保持原始图像
            pred_xstart = out["pred_xstart"].detach()
            # pred_xstart =denoise_with_fixed_epsilon(z_t, fixed_epsilon, t_val, diffusion)
            pred_xstart_steps.append(pred_xstart.detach())
            if grad_mask_x_t is not None:
                x_t = grad_mask_x_t * pred_xstart + (1 - grad_mask_x_t) * img.clone()

            # 计算梯度（如果需要）
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            nonzero_mask = (t_val != 0).float().view(-1, *([1] * (len(shape) - 1)))
            grads = None

            # x_t_for_classifier = (x_t.clamp(-1, 1) + 1) / 2

            if (class_grad_fn is not None) and (jdx < guided_iterations):
                # === 关键修改2：使用混合后的x_t计算梯度 ===
                grads_class = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t
                if grads_clip==True:

                    mean = grads_class.mean()
                    std = grads_class.std()
                    lower = mean - 3 * std
                    upper = mean + 3 * std
                    grads_class = grads_class.clamp(min=lower, max=upper)

                    # 假设 grads_class 是 torch.Tensor

                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,  # 使用混合后的x_t
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class

            # 应用梯度引导
            if grads is not None and torch.abs(grads).sum() > 0:
                # 计算动态TopK掩码
                if mask_class == "binary":
                    grad_mask, grad_mask_x_t =compute_grad_soft_masks_nested(grads_class,topk_ratio, x_t_rate)


                outmean_steps.append(out["mean"].detach())
                out["mean"]  = out["mean"] - out["variance"] * grads
                grads_steps.append(grads)
                guidedmean_steps.append(out["mean"].detach())

                z_original = diffusion.q_sample(img, t_val,noise=fixed_epsilon)
                out["mean"] = out["mean"]  * grad_mask + z_original * (1 - grad_mask)

            z_o_steps.append(z_original.detach())
            z_t_mask_steps.append(grad_mask.detach())
            c_z_t_mask_steps.append((1-grad_mask).detach())
            x_t_mask_steps.append(grad_mask_x_t.detach())
            c_x_t_mask_steps.append((1-grad_mask_x_t).detach())

            # 采样下一步
            if not x_t_sampling:
                z_t = out["mean"]
            else:
                noise = torch.randn_like(img) if jdx < len(indices) - 1 else torch.zeros_like(img)
                z_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

        return z_t, x_t_steps, z_t_steps, outmean_steps, z_t_mask_steps,c_z_t_mask_steps, x_t_mask_steps,c_x_t_mask_steps,pred_xstart_steps,z_o_steps,guidedmean_steps,grads_steps


    return p_sample_loop




def get_DiME_Mask_iterative_sampling(use_sampling=False):
    '''
    Returns DiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=True,
                      is_x_t_sampling=False,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      MAX_MASK_ACCUM_STEPS = 10,
                      mask_class="binary",

                      ):

        def compute_grad_mask_topk_binary(grads, topk_ratio=0.3):
            B, C, H, W = grads.shape
            grad_magnitude = torch.abs(grads)
            grad_magnitude = grad_magnitude.mean(dim=1, keepdim=True)
            flat = grad_magnitude.view(B, -1)
            k = max(1, int(topk_ratio * H * W))
            topk_vals, topk_idx = torch.topk(flat, k=k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, topk_idx, 1.0)
            return mask.view(B, 1, H, W)



        mask_buffer = []

        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        for jdx, i in enumerate(indices):
            t = torch.tensor([i] * shape[0], device=device)
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, shape)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )

            grads = 0
            if (class_grad_fn is not None) and (guided_iterations > jdx):
                grads_class = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t

            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                grads_dist = dist_grad_fn(z_t=z_t, x_tau=img, x_t=x_t,
                                          alpha_t=alpha_t, **dist_grad_kargs)
                grads = grad_scale * (grads_class +  grads_dist)

                if grads is not None and torch.abs(grads).sum() > 0:
                    if mask_class=="binary":
                        grad_mask = compute_grad_mask_topk_binary(grads_class, topk_ratio=topk_ratio)

                    else:
                        print("sb!")

                    # 累积 mask
                    mask_buffer.append(grad_mask.detach())
                    if len(mask_buffer) > MAX_MASK_ACCUM_STEPS:
                        mask_buffer.pop(0)
                    accum_mask = torch.stack(mask_buffer).mean(dim=0)


                    # 应用梯度引导
                    guided_mean = (
                            out["mean"].float() -
                            out["variance"] * grads * accum_mask
                    )

                    # 区域外恢复为原始噪声图像
                    z_original = diffusion.q_sample(img, t)
                    out["mean"] = guided_mean * accum_mask + z_original * (1 - accum_mask)

            if not x_t_sampling:
                z_t = out["mean"]
            else:
                z_t = (
                        out["mean"] +
                        nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img)
                )
            # if clip_zt is True:
            #
            #     mean = z_t.mean()
            #     std = z_t.std()
            #     clip_min = mean - 2 * std
            #     clip_max = mean + 2 * std
            #     z_t = torch.clamp(z_t, clip_min, clip_max)

            if (num_timesteps - (jdx + 1) > 0) and (class_grad_fn is not None) and (dist_grad_fn is not None) and (
                    guided_iterations > jdx):
                x_t = p_sample_loop(
                    diffusion=diffusion,
                    model=model,
                    model_kwargs=model_kwargs,
                    shape=shape,
                    num_timesteps=num_timesteps - (jdx + 1),
                    img=img,
                    t=None,
                    z_t=z_t,
                    clip_denoised=True,
                    device=device,
                    x_t_sampling=use_sampling,
                    is_x_t_sampling=True,
                    grad_scale=grad_scale,
                    topk_ratio=topk_ratio,
                    MAX_MASK_ACCUM_STEPS=MAX_MASK_ACCUM_STEPS,
                    # clip_zt=clip_zt
                )[0]

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop


def get_FastDiME_Mask_iterative_sampling(use_sampling=False):
    '''
    Returns DiME's main algorithm to construct counterfactuals.
    '''

    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      is_x_t_sampling=True,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      MAX_MASK_ACCUM_STEPS=10,
                      mask_class="binary"):

        def compute_grad_mask_topk_binary(grads, topk_ratio=0.3):
            B, C, H, W = grads.shape
            grad_magnitude = torch.abs(grads)
            grad_magnitude = grad_magnitude.mean(dim=1, keepdim=True)
            flat = grad_magnitude.view(B, -1)
            k = max(1, int(topk_ratio * H * W))
            topk_vals, topk_idx = torch.topk(flat, k=k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, topk_idx, 1.0)
            return mask.view(B, 1, H, W)

        # 初始化变量
        mask_buffer = []
        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t
        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        # 初始化全1掩码（第一步使用整个图像）
        accum_mask = torch.zeros_like(img[:, :1, :, :])
        accum_mask_prev = accum_mask.clone()

        # 主采样循环
        for jdx, i in enumerate(indices):
            t_val = torch.tensor([i] * shape[0], device=device)

            # 保存当前状态
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # 预测当前步的均值和方差
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t_val,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )

            # === 关键修改1：使用掩码混合预测结果和原始图像 ===
            # 只在掩码区域内使用预测的x0，掩码区域外保持原始图像
            pred_xstart = out["pred_xstart"].detach()
            x_t = accum_mask_prev * pred_xstart + (1 - accum_mask_prev) * img

            # 计算梯度（如果需要）
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_val, shape)
            nonzero_mask = (t_val != 0).float().view(-1, *([1] * (len(shape) - 1)))
            grads = None

            if (class_grad_fn is not None) and (jdx < guided_iterations):

                grads_class = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t

                if dist_grad_fn is not None:
                    grads_dist = dist_grad_fn(
                        z_t=z_t,
                        x_tau=img,
                        x_t=x_t,  # 使用混合后的x_t
                        alpha_t=alpha_t,
                        **dist_grad_kargs
                    )
                    grads = grad_scale * (grads_class + grads_dist)
                else:
                    grads = grad_scale * grads_class

            # 应用梯度引导
            if grads is not None and torch.abs(grads).sum() > 0:
                # 计算动态TopK掩码
                if mask_class == "binary":
                    # === 关键修改3：基于当前x_t的梯度计算新掩码 ===
                    grad_mask = compute_grad_mask_topk_binary(grads_class, topk_ratio)

                # 更新掩码缓冲区
                mask_buffer.append(grad_mask.detach())
                if len(mask_buffer) > MAX_MASK_ACCUM_STEPS:
                    mask_buffer.pop(0)
                accum_mask = torch.stack(mask_buffer).mean(dim=0)

                # 保存当前掩码用于下一步
                accum_mask_prev = accum_mask.clone()

                # 应用梯度更新（仅在掩码区域）
                guided_mean = out["mean"] - out["variance"] * grads * accum_mask
                z_original = diffusion.q_sample(img, t_val)
                out["mean"] = guided_mean * accum_mask + z_original * (1 - accum_mask)

            # 采样下一步
            if not is_x_t_sampling:
                z_t = out["mean"]
            else:
                noise = torch.randn_like(img) if jdx < len(indices) - 1 else torch.zeros_like(img)
                z_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop




def get_FastDiMEnomask_iterative_sampling(use_sampling=False):
    @torch.no_grad()
    def p_sample_loop(diffusion,
                      model,
                      shape,
                      num_timesteps,
                      img,
                      t,
                      z_t=None,
                      clip_denoised=True,
                      model_kwargs=None,
                      device=None,
                      class_grad_fn=None,
                      class_grad_kwargs=None,
                      dist_grad_fn=None,
                      dist_grad_kargs=None,
                      x_t_sampling=True,
                      is_x_t_sampling=False,
                      guided_iterations=9999999,
                      grad_scale=1.5,
                      topk_ratio=0.4,
                      MAX_MASK_ACCUM_STEPS=10,
                      mask_class="binary",

                      ):



        x_t = img.clone()
        z_t = diffusion.q_sample(img, t) if z_t is None else z_t

        x_t_steps = []
        z_t_steps = []
        indices = list(range(num_timesteps))[::-1]

        for jdx, i in enumerate(indices):

            t = torch.tensor([i] * shape[0], device=device)
            x_t_steps.append(x_t.detach())
            z_t_steps.append(z_t.detach())

            # out is a dictionary with the following (self-explanatory) keys:
            # 'mean', 'variance', 'log_variance'
            out = diffusion.p_mean_variance(
                model,
                z_t,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )
            x_t =  out["pred_xstart"].detach()
            alpha_t = _extract_into_tensor(diffusion.sqrt_alphas_cumprod,
                                           t, shape)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(shape) - 1)))
            )  # no noise when t == 0

            grads = 0

            grads_class = class_grad_fn(x_t=x_t, **class_grad_kwargs) / alpha_t

            if (dist_grad_fn is not None) and (guided_iterations > jdx):
                grads_dist = dist_grad_fn(
                    z_t=z_t,
                    x_tau=img,
                    x_t=x_t,
                    alpha_t=alpha_t,
                    **dist_grad_kargs
                )
                grads = grad_scale * (grads_class + grads_dist)
            else:
                grads = grad_scale * grads_class

            out["mean"] = (
                    out["mean"].float() -
                    out["variance"] * grads
            )

            if not x_t_sampling:
                z_t = out["mean"]

            else:
                z_t = (
                        out["mean"] +
                        nonzero_mask * torch.exp(0.5 * out["log_variance"]) * torch.randn_like(img)
                )

        return z_t, x_t_steps, z_t_steps

    return p_sample_loop





class ChunkedDataset:
    def __init__(self, dataset, chunk=0, num_chunks=1):
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset)) if (i % num_chunks) == chunk]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        i = [self.indexes[idx]]
        i += list(self.dataset[i[0]])
        return i


class ImageSaver():
    def __init__(self, output_path, exp_name, extention='.jpg'):
        self.output_path = output_path
        self.exp_name = exp_name
        self.idx = 0
        self.extention = extention
        self.construct_directory()

    def construct_directory(self):

        os.makedirs(osp.join(self.output_path, 'Original', 'Correct'), exist_ok=True)
        os.makedirs(osp.join(self.output_path, 'Original', 'Incorrect'), exist_ok=True)

        for clst, cf, subf in itertools.product(['CC', 'IC'],
                                                ['CCF', 'ICF'],
                                                ['CF', 'Noise', 'Info', 'SM']):
            os.makedirs(osp.join(self.output_path, 'Results',
                                 self.exp_name, clst,
                                 cf, subf),
                        exist_ok=True)

    def __call__(self, imgs, cfs, noises, target, label,
                 pred, pred_cf, bkl, l_1, indexes=None, masks=None):

        for idx in range(len(imgs)):
            current_idx = indexes[idx].item() if indexes is not None else idx + self.idx
            mask = None if masks is None else masks[idx]
            self.save_img(img=imgs[idx],
                          cf=cfs[idx],
                          noise=noises[idx],
                          idx=current_idx,
                          target=target[idx].item(),
                          label=label[idx].item(),
                          pred=pred[idx].item(),
                          pred_cf=pred_cf[idx].item(),
                          bkl=bkl[idx].item(),
                          l_1=l_1[idx].item(),
                          mask=mask)

        self.idx += len(imgs)

    @staticmethod
    def select_folder(label, target, pred, pred_cf):
        folder = osp.join('CC' if label == pred else 'IC',
                          'CCF' if target == pred_cf else 'ICF')
        return folder

    @staticmethod
    def preprocess(img):
        '''
        remove last dimension if it is 1
        '''
        if img.shape[2] > 1:
            return img
        else:
            return np.squeeze(img, 2)

    def save_img(self, img, cf, noise, idx, target, label,
                 pred, pred_cf, bkl, l_1, mask):
        folder = self.select_folder(label, target, pred, pred_cf)
        output_path = osp.join(self.output_path, 'Results',
                               self.exp_name, folder)
        img_name = f'{idx}'.zfill(7)
        orig_path = osp.join(self.output_path, 'Original',
                             'Correct' if label == pred else 'Incorrect',
                             img_name + self.extention)

        if mask is None:
            l0 = np.abs(img.astype('float') - cf.astype('float'))
            l0 = l0.sum(2, keepdims=True)
            l0 = 255 * l0 / l0.max()
            l0 = np.concatenate([l0] * img.shape[2], axis=2).astype('uint8')
            l0 = Image.fromarray(self.preprocess(l0))
            l0.save(osp.join(output_path, 'SM', img_name + self.extention))
        else:
            mask = mask.astype('uint8') * 255
            mask = Image.fromarray(mask)
            mask.save(osp.join(output_path, 'SM', img_name + self.extention))

        img = Image.fromarray(self.preprocess(img))
        img.save(orig_path)

        cf = Image.fromarray(self.preprocess(cf))
        cf.save(osp.join(output_path, 'CF', img_name + self.extention))

        noise = Image.fromarray(self.preprocess(noise))
        noise.save(osp.join(output_path, 'Noise', img_name + self.extention))


        to_write = (f'label: {label}' +
                    f'\npred: {pred}' +
                    f'\ntarget: {target}' +
                    f'\ncf pred: {pred_cf}' +
                    f'\nBKL: {bkl}' +
                    f'\nl_1: {l_1}')
        with open(osp.join(output_path, 'Info', img_name + '.txt'), 'w') as f:
            f.write(to_write)
class VGGNormalizer(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mu', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return self.classifier(x)


class Normalizer(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mu', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def forward(self, x):
        x = (torch.clamp(x, -1, 1) + 1) / 2
        x = (x - self.mu) / self.sigma
        return self.classifier(x)

class ClassifierNormalizer(torch.nn.Module):
    '''
    normalizing module. Useful for computing the gradient
    to a x image (x in [0, 1]) when using a classifier with
    different normalization inputs (i.e. f((x - mu) / sigma))
    '''
    def __init__(self, classifier,
                 mu=[0.485, 0.456, 0.406],
                 sigma=[0.229, 0.224, 0.225]):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mu', torch.tensor(mu).view(1, -1, 1, 1))
        self.register_buffer('sigma', torch.tensor(sigma).view(1, -1, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return self.classifier(x)



class SingleLabel(ImageFolder):
    def __init__(self, query_label, **kwargs):
        super().__init__(**kwargs)
        self.query_label = query_label

        # remove those instances that do no have the
        # query label

        old_len = len(self)
        instances = [self.targets[i] == query_label
                     for i in range(old_len)]
        self.samples = [self.samples[i]
                        for i in range(old_len) if instances[i]]
        self.targets = [self.targets[i]
                        for i in range(old_len) if instances[i]]
        self.imgs = [self.imgs[i]
                     for i in range(old_len) if instances[i]]


class SlowSingleLabel():
    def __init__(self, query_label, dataset, maxlen=float('inf')):
        self.dataset = dataset
        self.indexes = []
        if isinstance(dataset, ImageFolder):
            self.indexes = np.where(np.array(dataset.targets) == query_label)[0]
            self.indexes = self.indexes[:maxlen]
        else:
            print('Slow route. This may take some time!')
            if query_label != -1:
                for idx, (_, l) in enumerate(tqdm(dataset)):

                    l = l['y'] if isinstance(l, dict) else l
                    if l == query_label:
                        self.indexes.append(idx)

                    if len(self.indexes) == maxlen:
                        break
            else:
                self.indexes = list(range(min(maxlen, len(dataset))))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]


class PerceptualLoss(nn.Module):
    def __init__(self, layer, c, dataset_name='Celeb'):
        super().__init__()
        self.c = c
        vgg19_model = vgg19(pretrained=True)
        vgg19_model = nn.Sequential(*list(vgg19_model.features.children())[:layer])

        # if dataset_name in ['CelebA','CelebAMT','CelebAMV']:
        self.model = Normalizer(vgg19_model)
        # else:
        #     self.model = VGGNormalizer(vgg19_model)

        self.model.eval()

    def forward(self, x0, x1):
        B = x0.size(0)
        l = F.mse_loss(self.model(x0).view(B, -1),
                       self.model(x1).view(B, -1),
                       reduction='none').mean(dim=1)
        return self.c * l.sum()



class extra_data_saver():
    def __init__(self, output_path, exp_name):
        self.idx = 0
        self.exp_name = exp_name

    def __call__(self, x_ts, indexes=None):
        n_images = x_ts[0].size(0)
        n_steps = len(x_ts)

        for i in range(n_images):
            current_idx = indexes[i].item() if indexes is not None else i + self.idx
            os.makedirs(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6)), exist_ok=True)

            for j in range(n_steps):
                cf = x_ts[j][i, ...]

                # renormalize the image
                cf = ((cf + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                cf = cf.permute(1, 2, 0)
                cf = cf.contiguous().cpu().numpy()
                cf = Image.fromarray(cf)
                cf.save(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6), str(j).zfill(4) + '.jpg'))

        self.idx += n_images


class X_T_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.jpg'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'x_t')


class Z_T_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.jpg'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'z_t')
class PredXstart_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.jpg'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'predxstart')
class OutMean_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.jpg'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'outmean')
class GuidedMean_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.jpg'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'guidedmean')
class Grads_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.jpg'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'grads')

class Z_O_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.jpg'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'z_original')

class Z_T_Mask_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extension='.png'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'z_t_mask')
        self.extension = extension

    def __call__(self, x_ts, indexes=None):
        n_images = x_ts[0].size(0)
        n_steps = len(x_ts)

        for i in range(n_images):
            current_idx = indexes[i].item() if indexes is not None else i + self.idx
            save_dir = osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6))
            os.makedirs(save_dir, exist_ok=True)

            for j in range(n_steps):
                cf = x_ts[j][i, ...]  # 可能是(1,H,W)或(H,W)

                if cf.dim() == 3 and cf.size(0) == 1:
                    cf = cf.squeeze(0)

                cf = (cf * 255).to(torch.uint8).contiguous().cpu().numpy()
                img = Image.fromarray(cf)
                img.save(osp.join(save_dir, f"{str(j).zfill(4)}{self.extension}"))

        self.idx += n_images

class C_Z_T_Mask_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extension='.png'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'c_z_t_mask')
        self.extension = extension

    def __call__(self, x_ts, indexes=None):
        n_images = x_ts[0].size(0)
        n_steps = len(x_ts)

        for i in range(n_images):
            current_idx = indexes[i].item() if indexes is not None else i + self.idx
            save_dir = osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6))
            os.makedirs(save_dir, exist_ok=True)

            for j in range(n_steps):
                cf = x_ts[j][i, ...]  # 可能是(1,H,W)或(H,W)

                if cf.dim() == 3 and cf.size(0) == 1:
                    cf = cf.squeeze(0)

                cf = (cf * 255).to(torch.uint8).contiguous().cpu().numpy()
                img = Image.fromarray(cf)
                img.save(osp.join(save_dir, f"{str(j).zfill(4)}{self.extension}"))

        self.idx += n_images
class X_T_Mask_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extension='.png'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'x_t_mask')
        self.extension = extension  # mask保存一般用png，避免压缩损失

    def __call__(self, x_ts, indexes=None):
        n_images = x_ts[0].size(0)
        n_steps = len(x_ts)

        for i in range(n_images):
            current_idx = indexes[i].item() if indexes is not None else i + self.idx
            save_dir = osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6))
            os.makedirs(save_dir, exist_ok=True)

            for j in range(n_steps):
                cf = x_ts[j][i, ...]  # tensor shape (1, H, W) or maybe (H, W)

                # 如果有通道维度，去掉它
                if cf.dim() == 3 and cf.size(0) == 1:
                    cf = cf.squeeze(0)  # 变成 (H, W)

                # mask一般是0/1，直接乘255变成uint8格式
                cf = (cf * 255).to(torch.uint8).contiguous().cpu().numpy()

                # 转成PIL图像保存
                img = Image.fromarray(cf)
                img.save(osp.join(save_dir, f"{str(j).zfill(4)}{self.extension}"))

        self.idx += n_images


class C_X_T_Mask_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extension='.png'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'c_x_t_mask')
        self.extension = extension  # mask保存一般用png，避免压缩损失

    def __call__(self, x_ts, indexes=None):
        n_images = x_ts[0].size(0)
        n_steps = len(x_ts)

        for i in range(n_images):
            current_idx = indexes[i].item() if indexes is not None else i + self.idx
            save_dir = osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6))
            os.makedirs(save_dir, exist_ok=True)

            for j in range(n_steps):
                cf = x_ts[j][i, ...]  # tensor shape (1, H, W) or maybe (H, W)

                # 如果有通道维度，去掉它
                if cf.dim() == 3 and cf.size(0) == 1:
                    cf = cf.squeeze(0)  # 变成 (H, W)

                # mask一般是0/1，直接乘255变成uint8格式
                cf = (cf * 255).to(torch.uint8).contiguous().cpu().numpy()

                # 转成PIL图像保存
                img = Image.fromarray(cf)
                img.save(osp.join(save_dir, f"{str(j).zfill(4)}{self.extension}"))

        self.idx += n_images



class Mask_Saver(extra_data_saver):
    def __init__(self, output_path, exp_path, extention='.jpg'):
        super().__init__(output_path, exp_path)
        self.output_path = osp.join(output_path, 'masks')

    def __call__(self, masks, indexes=None):
        '''
        Masks are non-binarized 
        '''
        n_images = masks[0].size(0)
        n_steps = len(masks)

        for i in range(n_images):
            current_idx = indexes[i].item() if indexes is not None else i + self.idx
            os.makedirs(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6)), exist_ok=True)

            for j in range(n_steps):
                cf = masks[j][i, ...]
                cf = torch.cat((cf, (cf > 0.5).to(cf.dtype)), dim=-1)

                # renormalize the image
                cf = (cf * 255).clamp(0, 255).to(torch.uint8)
                cf = cf.permute(1, 2, 0)
                cf = cf.squeeze(dim=-1)
                cf = cf.contiguous().cpu().numpy()
                cf = Image.fromarray(cf)
                cf.save(osp.join(self.output_path, self.exp_name, str(current_idx).zfill(6), str(j).zfill(4) + self.extention))

        self.idx += n_images


class ImageSaverCF():
    def __init__(self, output_path, exp_name, extention='.jpg'):
        self.output_path = output_path
        self.exp_name = exp_name
        self.idx = 0
        self.extention = extention
        self.construct_directory()

    def construct_directory(self):

        os.makedirs(osp.join(self.output_path, 'Original', 'Correct'), exist_ok=True)
        os.makedirs(osp.join(self.output_path, 'Original', 'Incorrect'), exist_ok=True)

        for clst, cf, subf in itertools.product(['CC', 'IC'],
                                                ['CCF', 'ICF'],
                                                ['CF', 'Noise', 'Info', 'SM']):
            os.makedirs(osp.join(self.output_path, 'Results',
                                 self.exp_name, clst,
                                 cf, subf),
                        exist_ok=True)

    def __call__(self, imgs, cfs, noises, target, label, task_label,
                 pred, pred_cf, bkl, l_1, indexes=None, masks=None):

        for idx in range(len(imgs)):
            current_idx = indexes[idx].item() if indexes is not None else idx + self.idx
            mask = None if masks is None else masks[idx]
            self.save_img(img=imgs[idx],
                          cf=cfs[idx],
                          noise=noises[idx],
                          idx=current_idx,
                          target=target[idx].item(),
                          label=label[idx].item(),
                          task_label=task_label[idx].item(),
                          pred=pred[idx].item(),
                          pred_cf=pred_cf[idx].item(),
                          bkl=bkl[idx].item(),
                          l_1=l_1[idx].item(),
                          mask=mask)

        self.idx += len(imgs)

    @staticmethod
    def select_folder(label, target, pred, pred_cf):
        folder = osp.join('CC' if label == pred else 'IC',
                          'CCF' if target == pred_cf else 'ICF')
        return folder

    @staticmethod
    def preprocess(img):
        '''
        remove last dimension if it is 1
        '''
        if img.shape[2] > 1:
            return img
        else:
            return np.squeeze(img, 2)

    def save_img(self, img, cf, noise, idx, target, label, task_label,
                 pred, pred_cf, bkl, l_1, mask):
        folder = self.select_folder(label, target, pred, pred_cf)
        output_path = osp.join(self.output_path, 'Results',
                               self.exp_name, folder)
        img_name = f'{idx}'.zfill(7)
        orig_path = osp.join(self.output_path, 'Original',
                             'Correct' if label == pred else 'Incorrect',
                             img_name + self.extention)

        if mask is None:
            l0 = np.abs(img.astype('float') - cf.astype('float'))
            l0 = l0.sum(2, keepdims=True)
            l0 = 255 * l0 / l0.max()
            l0 = np.concatenate([l0] * img.shape[2], axis=2).astype('uint8')
            l0 = Image.fromarray(self.preprocess(l0))
            l0.save(osp.join(output_path, 'SM', img_name + self.extention))
        else:
            mask = mask.astype('uint8') * 255
            mask = Image.fromarray(mask)
            mask.save(osp.join(output_path, 'SM', img_name + self.extention))

        img = Image.fromarray(self.preprocess(img))
        img.save(orig_path)

        cf = Image.fromarray(self.preprocess(cf))
        cf.save(osp.join(output_path, 'CF', img_name + self.extention))

        noise = Image.fromarray(self.preprocess(noise))
        noise.save(osp.join(output_path, 'Noise', img_name + self.extention))


        to_write = (f'label: {label}' +
                    f'\ntask_label: {task_label}' +
                    f'\npred: {pred}' +
                    f'\ntarget: {target}' +
                    f'\ncf pred: {pred_cf}' +
                    f'\nBKL: {bkl}' +
                    f'\nl_1: {l_1}')
        with open(osp.join(output_path, 'Info', img_name + '.txt'), 'w') as f:
            f.write(to_write)