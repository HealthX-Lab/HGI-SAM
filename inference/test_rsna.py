import numpy as np
import os

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


def main():
    physio_path = "C:\\physio-ich"
    rsna_path = "C:\\rsna-ich"
    # ds = PhysioNetICHDataset(physio_path, windows=[(80, 340), (700, 3200)], transform=get_transform(384))

    t_x, t_y, v_x, v_y = rsna_train_valid_split(rsna_path)
    windows = [(80, 200), (600, 2800)]
    transform = get_transform(384)

    train_ds = RSNAICHDataset(rsna_path, t_x, t_y, windows=windows, transform=transform)
    validation_ds = RSNAICHDataset(rsna_path, v_x, v_y, windows=windows, transform=transform)

    model = SwinWeak(3, 1)
    load_model(model.swin, r"C:\rsna-ich\Good weights\backup\Focal-100-checkpoint-0.pt")
    for x, y in train_ds:
        if y[-1] == 1:
            p, p_mask = model.segmentation(x.unsqueeze(0))
            img = np.array(x.permute(1, 2, 0))
            p_mask = binarization_simple_thresholding(p_mask, 0.07)
            # p_mask = (p_mask - p_mask.min()) / (p_mask.max() - p_mask.min())
            #
            # p_mask_labels = one_hot(p_mask, 2, dim=0).numpy()
            # print(p_mask_labels.shape)
            # d = dcrf.DenseCRF2D(384, 384, 2)
            # U = unary_from_softmax(p_mask_labels)
            # d.setUnaryEnergy(U)
            #
            # feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
            # d.addPairwiseEnergy(feats, compat=1,
            #                     kernel=dcrf.DIAG_KERNEL,
            #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
            # feats = create_pairwise_bilateral(sdims=(16, 16), schan=(0.1, 0.1, 0.1),
            #                                   img=img, chdim=2)
            # d.addPairwiseEnergy(feats, compat=2,
            #                     kernel=dcrf.DIAG_KERNEL,
            #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
            # Q = d.inference(1)
            # map = np.argmax(Q, axis=0)
            # map = np.array(map).reshape(384, 384).astype(np.float32)
            # print(map.shape, map.dtype, map.min(), map.max())

            cv2.imshow('image', img)
            # cv2.imshow('mask', m.numpy())
            cv2.imshow('pred', p_mask[0].numpy())
            # cv2.imshow('crf', map)
            cv2.waitKey()


if __name__ == '__main__':
    main()
