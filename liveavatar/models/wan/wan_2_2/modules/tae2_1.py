import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out, act_func):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out),
            act_func,
            conv(n_out, n_out),
            act_func,
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = act_func

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _nt, c, h, w = x.shape
        return self.conv(x.reshape(-1, self.stride * c, h, w))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _nt, c, h, w = x.shape
        x = self.conv(x)
        return x.reshape(-1, c, h, w)


def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    n, t, c, h, w = x.shape
    if parallel:
        x = x.reshape(n * t, c, h, w)
        for block in tqdm(model, disable=not show_progress_bar):
            if isinstance(block, MemBlock):
                nt, c, h, w = x.shape
                t = nt // n
                x_reshaped = x.reshape(n, t, c, h, w)
                mem = F.pad(x_reshaped, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :t].reshape(x.shape)
                x = block(x, mem)
            else:
                x = block(x)
        nt, c, h, w = x.shape
        t = nt // n
        return x.view(n, t, c, h, w)

    out = []
    work_queue = [TWorkItem(xt, 0) for xt in x.reshape(n, t * c, h, w).chunk(t, dim=1)]
    progress_bar = tqdm(range(t), disable=not show_progress_bar)
    mem = [None] * len(model)
    while work_queue:
        xt, i = work_queue.pop(0)
        if i == 0:
            progress_bar.update(1)
        if i == len(model):
            out.append(xt)
            continue
        block = model[i]
        if isinstance(block, MemBlock):
            if mem[i] is None:
                xt_new = block(xt, xt * 0)
                mem[i] = xt
            else:
                xt_new = block(xt, mem[i])
                mem[i].copy_(xt)
            work_queue.insert(0, TWorkItem(xt_new, i + 1))
        elif isinstance(block, TPool):
            if mem[i] is None:
                mem[i] = []
            mem[i].append(xt)
            if len(mem[i]) == block.stride:
                n, c, h, w = xt.shape
                xt = block(torch.cat(mem[i], 1).view(n * block.stride, c, h, w))
                mem[i] = []
                work_queue.insert(0, TWorkItem(xt, i + 1))
        elif isinstance(block, TGrow):
            xt = block(xt)
            _nt, c_cur, h_cur, w_cur = xt.shape
            for xt_next in reversed(xt.view(n, block.stride * c_cur, h_cur, w_cur).chunk(block.stride, 1)):
                work_queue.insert(0, TWorkItem(xt_next, i + 1))
        else:
            xt = block(xt)
            work_queue.insert(0, TWorkItem(xt, i + 1))
    progress_bar.close()
    return torch.stack(out, 1)


class TAEHV(nn.Module):
    def __init__(
        self,
        checkpoint_path="taew2_1.pth",
        decoder_time_upscale=(True, True),
        decoder_space_upscale=(True, True, True),
        patch_size=1,
        latent_channels=16,
        model_type="wan21",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        self.is_cogvideox = checkpoint_path is not None and "taecvx" in checkpoint_path
        self.model_type = model_type
        if model_type == "wan22":
            self.patch_size, self.latent_channels = 2, 48
        act_func = nn.ReLU(inplace=True)

        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size**2, 64),
            act_func,
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            TPool(64, 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(),
            conv(self.latent_channels, n_f[0]),
            act_func,
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),
            act_func,
            conv(n_f[3], self.image_channels * self.patch_size**2),
        )

        if checkpoint_path is not None:
            ext = os.path.splitext(checkpoint_path)[1].lower()
            if ext == ".pth":
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            elif ext == ".safetensors":
                state_dict = load_file(checkpoint_path, device="cpu")
            else:
                raise ValueError(f"Unsupported checkpoint format: {ext}")
            self.load_state_dict(self.patch_tgrow_layers(state_dict))

    def patch_tgrow_layers(self, state_dict):
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if state_dict[key].shape[0] > new_sd[key].shape[0]:
                    state_dict[key] = state_dict[key][-new_sd[key].shape[0] :]
        return state_dict

    def encode_video(self, x, parallel=True, show_progress_bar=False):
        if self.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)
        if x.shape[1] % 4 != 0:
            n_pad = 4 - x.shape[1] % 4
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def decode_video(self, x, parallel=True, show_progress_bar=False):
        skip_trim = self.is_cogvideox and x.shape[1] % 2 == 0
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = x.clamp_(0, 1)
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        if skip_trim:
            return x
        return x[:, self.frames_to_trim :]
