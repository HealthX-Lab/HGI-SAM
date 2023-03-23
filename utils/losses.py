#!/usr/env/bin python3.9

from typing import List, cast

import torch
import numpy as np
from torch import Tensor, einsum

from utils.utils2 import simplex, probs2one_hot, one_hot
from utils.utils2 import one_hot2hd_dist


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        loss = - einsum("bkxy,bkxy->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bkxy->bk", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bkxy,bkxy->bk", pc, tc)
        union: Tensor = w * (einsum("bkxy->bk", pc) + einsum("bkxy->bk", tc))

        divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        loss = divided.mean()

        return loss


class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum("bcxy,bcxy->bc", pc, tc)
        union: Tensor = (einsum("bkxy->bk", pc) + einsum("bkxy->bk", tc))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss


class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkxy,bkxy->bkxy", pc, dc)

        loss = multipled.mean()

        return loss


BoundaryLoss = SurfaceLoss


class HausdorffLoss():
    """
    Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs)
        assert simplex(target)
        assert probs.shape == target.shape

        B, K, *xy = probs.shape  # type: ignore

        pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
        tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
        assert pc.shape == tc.shape == (B, len(self.idc), *xy)

        target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
                                              for b in range(B)], axis=0)
        assert target_dm_npy.shape == tc.shape == pc.shape
        tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)

        pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
        pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, self.idc, ...].numpy())
                                            for b in range(B)], axis=0)
        assert pred_dm_npy.shape == tc.shape == pc.shape
        pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)

        delta = (pc - tc)**2
        dtm = tdm**2 + pdm**2

        multipled = einsum("bkxy,bkxy->bkxy", delta, dtm)

        loss = multipled.mean()

        return loss


class FocalLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.gamma: float = kwargs["gamma"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        masked_probs: Tensor = probs[:, self.idc, ...]
        log_p: Tensor = (masked_probs + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        w: Tensor = (1 - masked_probs)**self.gamma
        loss = - einsum("bkxy,bkxy,bkxy->", w, mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss