import numpy as np
import os

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils.dataset import *
from utils.preprocessing import get_transform
from models.swin_weak import SwinWeak
import cv2
from utils.utils import *
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
from monai.transforms.post.array import one_hot
import numpy as np
from monai.metrics import DiceMetric, compute_meandice
from copy import deepcopy


def main():
    physio_path = "C:\\physio-ich"
    rsna_path = "C:\\rsna-ich"
    ds = PhysioNetICHDataset(physio_path, windows=[(80, 340), (700, 3200)], transform=get_transform(384))
    # ds = PhysioNetICHDataset(physio_path, windows=None, transform=get_transform(384))

    t_x, t_y, v_x, v_y = rsna_train_valid_split(rsna_path)
    windows = [(80, 200), (600, 2800)]
    transform = get_transform(384)

    train_ds = RSNAICHDataset(rsna_path, t_x, t_y, windows=windows, transform=transform)
    validation_ds = RSNAICHDataset(rsna_path, v_x, v_y, windows=windows, transform=transform)

    model = SwinWeak(3, 1)
    load_model(model, r"extra/weights/SwinWeak_refined.pt")
    model.cuda()
    model.eval()
    dice = DiceMetric(include_background=False, reduction='none')
    dice2 = DiceMetric(include_background=False, reduction='none')
    with torch.no_grad():
        for x, m, y in ds:
            if y[-1] == 1:
                x, m = x.to('cuda'), m.to('cuda')
                # x = x.to('cuda')
                p, p_mask = model.attentional_segmentation(x.unsqueeze(0))
                p_mask_copy = deepcopy(p_mask)
                p_mask = binarization_simple_thresholding(deepcopy(p_mask), 0.07)
                # # p_mask = (p_mask - p_mask.min()) / (p_mask.max() - p_mask.min())
                p_mask = one_hot(p_mask, 2, dim=0)
                refined_mask = model.refinement_segmentation(x.unsqueeze(0))
                refined_mask_copy = deepcopy(torch.softmax(refined_mask, dim=1))
                refined_mask = torch.argmax(torch.softmax(refined_mask, dim=1), dim=1)
                refined_mask = one_hot(refined_mask, 2, dim=0)
                #
                m[m > 0] = 1
                mask = one_hot(m.unsqueeze(0), 2, dim=0)

                img = np.array(x.permute(1, 2, 0).cpu())
                #
                # # p_mask2 = (p_mask2 - p_mask2.min()) / (p_mask2.max() - p_mask2.min())

                # d = dcrf.DenseCRF2D(384, 384, 2)
                # # U = unary_from_softmax(p_mask.cpu().numpy())
                # U = unary_from_labels(torch.argmax(p_mask, dim=0).cpu().numpy(), 2, gt_prob=0.9, zero_unsure=False)
                # d.setUnaryEnergy(U)
                #
                # feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
                # d.addPairwiseEnergy(feats, compat=2,
                #                     kernel=dcrf.DIAG_KERNEL,
                #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
                # feats = create_pairwise_bilateral(sdims=(16, 16), schan=(0.05, 0.1, 0.1),
                #                                   img=img, chdim=2)
                # d.addPairwiseEnergy(feats, compat=3,
                #                     kernel=dcrf.DIAG_KERNEL,
                #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
                # Q = d.inference(5)
                # crf = np.argmax(Q, axis=0)
                # crf = np.array(crf).reshape(384, 384).astype(np.float32)
                # crf = torch.FloatTensor(crf)
                # crf = one_hot(crf.unsqueeze(0), 2, dim=0).cuda()
                # # print(map.shape, map.dtype, map.min(), map.max())
                #
                # dice(p_mask.unsqueeze(0), mask.unsqueeze(0))
                # dice2(crf.unsqueeze(0), mask.unsqueeze(0))

                # print(compute_meandice(p_mask.unsqueeze(0).cuda(), mask.unsqueeze(0), include_background=False))
                # print(compute_meandice(refined_mask.unsqueeze(0), mask.unsqueeze(0), include_background=False))

                print(x.shape)
                cv2.imshow('image', img)
                cv2.imshow('mask', mask[1].cpu().numpy())
                cv2.imshow('pred', p_mask[1].cpu().numpy())
                cv2.imshow('refined pred', refined_mask[1].cpu().numpy())
                cv2.imshow('pred copy', p_mask_copy[0].cpu().numpy())
                cv2.imshow('refined copy', refined_mask_copy[0, 1].cpu().numpy())
                # cv2.imshow('crf', crf[1].cpu().numpy())
                cv2.waitKey()

    print(dice.aggregate(reduction='mean'))
    print(dice2.aggregate(reduction='mean'))


if __name__ == '__main__':
    main()
