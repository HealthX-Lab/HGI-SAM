import torch.optim
from utils.utils import *
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
from monai.transforms.post.array import one_hot
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda'):
    model.to(device)
    model.train()
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    pbar_train.set_description('training')
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}

    for i, (sample, label) in pbar_train:
        optimizer.zero_grad()
        sample, label = sample.to(device), label.to(device)

        pred = model(sample)
        # pred = F.softmax(pred, dim=1)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        _metrics["train_cfm"].add_loss(loss.item())
        _metrics["train_cfm"].add_prediction(torch.argmax(pred, dim=1), label)

    _metrics["train_cfm"].compute_confusion_matrix()

    model.eval()
    pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
    pbar_valid.set_description('validating')

    with torch.no_grad():
        for i, (sample, label) in pbar_valid:
            sample, label = sample.to(device), label.to(device)

            pred = model(sample)
            # pred = F.softmax(pred, dim=1)
            loss = loss_fn(pred, label)

            _metrics["valid_cfm"].add_loss(loss.item())
            _metrics["valid_cfm"].add_prediction(torch.argmax(pred, dim=1), label)

        _metrics["valid_cfm"].compute_confusion_matrix()

    return _metrics


def train_one_epoch_segmentation(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda', augmentation=None):
    model.to(device)
    model.train()
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    pbar_train.set_description('training')
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}

    for i, (sample, label) in pbar_train:
        optimizer.zero_grad()
        sample, label = sample.to(device), label.to(device)
        if augmentation is not None:
            for b in range(len(label)):
                sample[b], label[b] = augmentation(sample[b], label[b])

        pred = model(sample)
        loss = loss_fn(pred, label)

        loss.backward()
        optimizer.step()

        _metrics["train_cfm"].add_loss(loss.item())
        _metrics["train_cfm"].add_number_of_samples(len(label))

    model.eval()
    pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
    pbar_valid.set_description('validating')

    with torch.no_grad():
        for i, (sample, label) in pbar_valid:
            sample, label = sample.to(device), label.to(device)

            pred = model(sample)
            loss = loss_fn(pred, label)

            _metrics["valid_cfm"].add_loss(loss.item())
            _metrics["valid_cfm"].add_number_of_samples(len(label))

    return _metrics


def train_one_epoch_refine_segmentation(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn, train_loader, valid_loader, device='cuda', augmentation=None):
    model.to(device)
    model.train()
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    pbar_train.set_description('training')
    _metrics = {"train_cfm": ConfusionMatrix(), "valid_cfm": ConfusionMatrix()}

    for i, (sample, label) in pbar_train:
        optimizer.zero_grad()
        sample = sample.to(device)

        with torch.no_grad():
            p, p_mask = model.attentional_segmentation(sample)
            p_mask = binarization_simple_thresholding(p_mask, 0.07)

        if augmentation:
            sample, p_mask = augmentation(sample, p_mask)

        p_mask = one_hot(p_mask.unsqueeze(1), 2, dim=1)
        p_mask_pp = torch.zeros_like(p_mask)
        for k in range(p_mask.shape[0]):
            p_mask_pp[k] = post_process(sample[k], p_mask[k])
        pred = model.refinement_segmentation(sample)
        loss = loss_fn(pred, p_mask_pp)

        loss.backward()
        optimizer.step()

        _metrics["train_cfm"].add_loss(loss.item())
        _metrics["train_cfm"].add_number_of_samples(len(label))

    model.eval()
    pbar_valid = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
    pbar_valid.set_description('validating')

    with torch.no_grad():
        for i, (sample, label) in pbar_valid:
            sample = sample.to(device)

            p, p_mask = model.attentional_segmentation(sample)
            p_mask = binarization_simple_thresholding(p_mask, 0.07)

            p_mask = one_hot(p_mask.unsqueeze(1), 2, dim=1)
            p_mask_pp = torch.zeros_like(p_mask)
            for k in range(p_mask.shape[0]):
                p_mask_pp[k] = post_process(sample[k], p_mask[k])
            pred = model.refinement_segmentation(sample)
            loss = loss_fn(pred, p_mask_pp)

            _metrics["valid_cfm"].add_loss(loss.item())
            _metrics["valid_cfm"].add_number_of_samples(len(label))

    return _metrics


def post_process(x, p_mask):
    img = np.array(x.permute(1, 2, 0).cpu())

    d = dcrf.DenseCRF2D(384, 384, 2)
    U = unary_from_labels(torch.argmax(p_mask, dim=0).cpu().numpy(), 2, gt_prob=0.9, zero_unsure=False)
    d.setUnaryEnergy(U)

    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=2,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = create_pairwise_bilateral(sdims=(16, 16), schan=(0.05, 0.1, 0.1),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    crf = np.argmax(Q, axis=0)
    crf = np.array(crf).reshape(384, 384).astype(np.float32)
    crf = torch.FloatTensor(crf)
    crf = one_hot(crf.unsqueeze(0), 2, dim=0).cuda()

    return crf
