from dataset import *
from train import *
from preprocessing import *
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from models.unet import *
from models.swin_unetr import *
from models.swin_weak import *
from models.swin_weak import *
import torch.optim as optim
from utils import *
import statistics
import argparse


def main(args: argparse.Namespace):
    if args.model == "swin-unetr":
        model = SwinUNetR(3, 1)
    elif args.model == "unet":
        model = UNet(3, 1)
    else:
        model = SwinWeak(3, 1)

    if args.loss == "focal":
        loss_fn = FocalLoss()
    elif args.loss == "dice":
        loss_fn = DiceLoss()
    elif args.loss == "dicebce":
        loss_fn = DiceBCELoss()
    elif args.loss == "focaldicebce":
        loss_fn = FocalDiceBCELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    if args.setup == "rsna":
        train_rsna(args.rsna_path, model, loss_fn, int(args.batch_size), int(args.num_workers))
    else:
        train_and_test_physionet(args.physio_path, model, loss_fn)


def train_rsna(root_dir, model, loss_fn, batch_size, num_workers):
    t_x, t_y, v_x, v_y = rsna_2d_train_validation_split(root_dir)
    checkpoint_name = model.__class__.__name__ + "-" + loss_fn.__class__.__name__

    windows = [(80, 200), (600, 2800)]
    transform = get_transform(384)
    train_ds = RSNAICHDataset2D(root_dir, t_x, t_y, windows=windows, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=rsna_collate_binary_label)
    validation_ds = RSNAICHDataset2D(root_dir, v_x, v_y, windows=windows, transform=transform)
    valid_loader = DataLoader(validation_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=rsna_collate_binary_label)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(model, 3, checkpoint_name + ".pth")
    epoch = 1
    while not early_stopping.early_stop:
        m = train_one_epoch(model, optimizer, loss_fn, train_loader, valid_loader)
        early_stopping(m['valid_cfm'].get_mean_loss())
        print("epoch:", epoch, "acc=", m['valid_cfm'].get_accuracy(), " F1=", m['valid_cfm'].get_f1_score())
        epoch += 1


def train_physionet(model: nn.Module, loss_fn, train_loader, valid_loader, checkpoint_name, cf, device='cuda'):
    augmentation = Augmentation(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(model, 3, f'{checkpoint_name}-fold{cf}.pth')
    while not early_stopping.early_stop:
        m = train_one_epoch_segmentation(model, optimizer, loss_fn, train_loader, valid_loader, augmentation=augmentation, device=device)
        print('\nvalid dice:', m['valid_cfm'].get_mean_dice(), ' IoU:', m['valid_cfm'].get_mean_iou(), ' Hausdorff:', m['valid_cfm'].get_mean_hausdorff_distance())
        early_stopping(m['valid_cfm'].get_mean_loss())


def train_and_test_physionet(physio_path, model, loss_fn):
    k = 10
    device = 'cuda'
    checkpoint_name = model.__class__.__name__ + "-" + loss_fn.__class__.__name__

    ds = PhysioNetICHDataset2D(physio_path,  windows=[(80, 340), (700, 3200)], transform=get_transform(384))

    indices = np.arange(0, len(ds.labels))
    encoded_labels = LabelEncoder().fit_transform([''.join(str(l)) for l in ds.labels])
    skf = StratifiedKFold(k)
    test_cfm_matrices = []
    for cf, (train_valid_indices, test_indices) in enumerate(skf.split(indices, encoded_labels)):  # dividing intro train/test based on all subtypes
        # dividing into train/valid based on any hemorrhage
        train_indices, valid_indices = train_test_split(train_valid_indices, stratify=ds.labels[train_valid_indices, -1], test_size=1. / (k - 1), random_state=42)

        train_ds = Subset(ds, train_indices)
        valid_ds = Subset(ds, valid_indices)
        test_ds = Subset(ds, test_indices)

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=physio_collate_image_mask)
        valid_loader = DataLoader(valid_ds, batch_size=1, collate_fn=physio_collate_image_mask)
        test_loader = DataLoader(test_ds, batch_size=1, collate_fn=physio_collate_image_mask)

        train_physionet(model, loss_fn, train_loader, valid_loader, checkpoint_name, cf, device)
        load_model(model, f'{checkpoint_name}-fold{cf}.pth')
        test_cfm = test_physionet(model, test_loader, False, 0.5, device)

        test_cfm_matrices.append(test_cfm)
        print(f'fold {cf+1} dice:', test_cfm.get_mean_dice(), ' iou:', test_cfm.get_mean_iou(), ' hausdorff:', test_cfm.get_mean_hausdorff_distance())

    dices = []
    IoUs = []
    hausdorff_distances = []
    for i in range(k):
        dices.append(test_cfm_matrices[i].get_mean_dice().item())
        IoUs.append(test_cfm_matrices[i].get_mean_iou().item())
        hausdorff_distances.append(test_cfm_matrices[i].get_mean_hausdorff_distance().item())

    print('dice: ', statistics.mean(dices), ' +/- ', statistics.stdev(dices))
    print('IoU: ', statistics.mean(IoUs), ' +/- ', statistics.stdev(IoUs))
    print('hausdorff: ', statistics.mean(hausdorff_distances), ' +/- ', statistics.stdev(hausdorff_distances))


def test_physionet(model, test_loader, weak_model=False, threshold=0.5, device="cuda"):
    model.to(device)
    test_cfm = ConfusionMatrix()
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            if not label.any():
                continue
            if weak_model:
                pred, pred_mask = model.segmentation(image)
            else:
                pred_mask = model(image)
                pred_mask = torch.sigmoid(pred_mask).squeeze(1)

            pred_mask = binarization_simple_thresholding(pred_mask, threshold)
            test_cfm.add_dice(dice_metric(pred_mask, label))
            test_cfm.add_iou(intersection_over_union(pred_mask, label))
            test_cfm.add_hausdorff_distance(hausdorff_distance(pred_mask, label))

    return test_cfm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--rsna_path", help="path to the rsna dataset root directory", default=r'C:\rsna-ich')
    parser.add_argument("--physio_path", help="path to the physionet dataset root directory", default=r'C:\physio-ich')
    parser.add_argument("--model", help="swin-weak, swin-unet, or unet", default="unet")
    parser.add_argument("--setup", help="rsna or physio", default="physio")
    parser.add_argument("--loss", help="focal, bce, dice, dicebce, or focaldicebce", default="focal")
    parser.add_argument("--batch_size", help="batch size", default=16)
    parser.add_argument("--num_workers", help="number of workers", default=1)

    main(parser.parse_args())
