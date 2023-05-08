import os
import torchvision
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import nibabel
import pickle
import numpy as np
import csv
import pydicom
from torchvision.transforms.functional import rotate
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from helpers.preprocessing import Augmentation
from helpers.preprocessing import window_image


class RSNAICHDataset(Dataset):
    def __init__(self, root_dir: str, filenames: [str], labels: np.array, windows: [(int, int)] = None,
                 transform: torchvision.transforms.transforms.Compose = None, augmentation: Augmentation = None):
        """
        Specific pytorch dataset designed for RSNA ICH dataset

        :param root_dir: path to RSNA ICH root directory
        :param filenames: list of filenames that is used in this dataset (helps in differentiating configs and validation sets)
        :param labels: array of labels corresponding to each filename in filenames
        :param windows: a list of (a, b) tuples whereas a: window-center, b: window-width
        :param transform: transforms applied to the windowed image such is resizing
        :param augmentation: augmentations applied to the transformed image such is random affine transform
        """
        self.train_dir = os.path.join(root_dir, 'stage_2_train')
        self.filenames = filenames
        self.labels = labels
        self.windows = windows
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image_path = os.path.join(self.train_dir, self.filenames[item])

        image, default_window_params = _read_image_2d(image_path)
        window_center, window_width, window_intercept, window_slope = default_window_params
        label = torch.LongTensor(self.labels[item])  # converting the label into a long tensor

        # taking brain window based on the windowing parameters derived from dicom headers
        default_window = _get_image_windows(image, [(window_center, window_width)], window_intercept, window_slope)
        if self.windows is not None:  # adding other windows such is brain-window or subdural-window
            image = torch.cat([default_window, _get_image_windows(image, self.windows, window_intercept, window_slope)])
        else:
            image = default_window

        if self.transform is not None:
            image = self.transform(image)

        if self.augmentation is not None:
            image = self.augmentation(image)

        return image, label


class PhysioNetICHDataset(Dataset):
    def __init__(self, root_dir: str, windows: [(int, int)] = None, transform: torchvision.transforms.transforms.Compose = None):
        """
        Specific pytorch dataset designed for PhysioNet ICH dataset. Here, because the dataset is small and also in 3D,
        first, we read all 2D slices, and then, we process them.

        :param root_dir: path to the PhysioNet ICH root directory
        :param windows: a list of (a, b) tuples whereas a: window-center, b: window-width
        :param transform: transforms applied to the windowed image such is resizing
        """
        self.scans_dir = os.path.join(root_dir, 'ct_scans')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.labels_path = os.path.join(root_dir, 'hemorrhage_diagnosis_raw_ct.csv')
        self.brains_dir = os.path.join(root_dir, 'brainmask_ero')  # this directory is created by our scripts in the root directory
        self.filenames = os.listdir(self.scans_dir)  # list of 3d nifty images

        self.slices = []  # a list containing 2D slices of the scan
        self.brains = []  # a list containing brain-masks corresponding to each slice
        self.masks = []  # a list containing hemorrhage-segmented-masks corresponding to each slice
        self.labels = []  # a list of hemorrhage-existence labels corresponding to each slice

        self.transform = transform
        self.windows = windows

        self.read_dataset()

    def read_dataset(self):
        """
        a function that reads 3D PhysioNet ICH dataset, and stores their 2D slices.
        """
        # reading the labels for each slice
        SUBTYPES = ["Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural", "No_Hemorrhage"]
        with open(self.labels_path, newline='') as labels_csv:
            reader = csv.DictReader(labels_csv)
            for row in reader:
                label = np.zeros(len(SUBTYPES))
                for i, subtype in enumerate(SUBTYPES):
                    label[i] = float(row[subtype])
                label[-1] = 1 - label[-1]  # Any hemorrhage = 1 - No Hemorrhage

                self.labels.append(label)
        self.labels = np.array(self.labels)

        # reading 3D scans, masks, and brain-masks and storing their 2D slices into lists
        pbar = tqdm(self.filenames, total=len(self.filenames))
        pbar.set_description("reading physionet dataset")
        for file in pbar:
            scan = _read_image_3d(os.path.join(self.scans_dir, file), do_rotate=True)
            mask = _read_image_3d(os.path.join(self.masks_dir, file), do_rotate=True)
            brain = _read_image_3d(os.path.join(self.brains_dir, f'{file.split(".")[0]}_mask.nii.gz'), do_rotate=True)

            num_slices = scan.shape[-1]

            for i in range(num_slices):
                self.slices.append(scan[:, :, i])
                self.masks.append(mask[:, :, i])
                self.brains.append(brain[:, :, i])

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = self.slices[item]
        mask = self.masks[item]
        brain = self.brains[item]
        label = self.labels[item]

        if mask.max() > 0:  # 0-1 normalization
            mask = (mask - mask.min()) / (mask.max() - mask.min())

        # based on PhysioNet documentations, we consider window-center=40 and window-width=120 as the default brain-window parameters
        default_window = _get_image_windows(image, [(40, 120)], 0, 1)
        if self.windows is not None:  # adding other windows such as bone-window and subdural-window
            image = torch.cat([default_window, _get_image_windows(image, self.windows, 0, 1)])
        else:
            image = default_window

        # binarization of ground-truth mask
        mask[mask > 0] = 1
        # adding intensity channel to the binary brain-mask and mask
        brain = brain.unsqueeze(0)
        mask = mask.unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)
            brain = self.transform(brain)
            mask = self.transform(mask)

        return image, mask, brain, label


def physio_collate_image_mask(batch):
    """
    collate function for PhysioNet dataset that only returns image and its ground-truth mask
    :param batch: the read batch
    :return: a tuple of image intensities and ground-truth mask
    """
    data = torch.stack([item[0] for item in batch])
    mask = torch.stack([item[1] for item in batch])

    return [data, mask]


def physio_collate_image_label(batch):
    """
    collate function for PhysioNet dataset that only returns image and its label
    :param batch: the read batch
    :return: a tuple of image intensities and hemorrhage-existence label
    """
    data = torch.stack([item[0] for item in batch])
    target = torch.stack([item[3] for item in batch])

    return [data, target]


def rsna_collate_binary_label(batch):
    """
    collate function for RSNA dataset that returns only any-hemorrhage label
    :param batch: the read batch
    :return: a tuple of image intensities and hemorrhage-existence (any) label
    """
    data = torch.stack([item[0] for item in batch])
    target = torch.stack([item[1] for item in batch])
    target = target[:, -1]
    return [data, target]


def rsna_train_valid_split(root_dir: str, extra_path: str, validation_size=0.1, random_state=42, override=False):
    """
    a method that splits the RSNA ICH 2D dicom dataset into configs and validation set randomly.
    we save the split into files for faster computation and further requirements
    :param root_dir: path to the RSNA ICH dataset root directory
    :param extra_path: path to the extra directory which contains split files
    :param validation_size: proportion of validation set
    :param random_state: the random state for splitting
    :param override: whether to compute the split again and rewrite filenames
    :return: filenames and corresponding labels for configs and validation sets.
    """
    SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]
    train_file_split_path, validation_file_split_path = os.path.join(extra_path, 'rsna_division', 'train_file_split.pt'), os.path.join(extra_path, 'rsna_division', 'validation_file_split.pt')
    train_label_split_path, validation_label_split_path = os.path.join(extra_path, 'rsna_division', 'train_label_split.pt'), os.path.join(extra_path, 'rsna_division', 'validation_label_split.pt')
    if os.path.isfile(train_file_split_path) and os.path.isfile(validation_file_split_path) and os.path.isfile(train_label_split_path) and os.path.isfile(validation_label_split_path) and not override:
        with open(train_file_split_path, "rb") as tf, open(train_label_split_path, "rb") as tl, open(validation_file_split_path, "rb") as vf, open(validation_label_split_path, "rb") as vl:
            return pickle.load(tf), pickle.load(tl), pickle.load(vf), pickle.load(vl)

    labels_path = os.path.join(root_dir, 'stage_2_train.csv')
    train_path = os.path.join(root_dir, 'stage_2_train')

    total_filenames = os.listdir(train_path)
    corrupted_files = ['ID_6431af929.dcm']  # files that have encoding problems and cause runtime errors
    for corrupted_file in corrupted_files:
        if corrupted_file in total_filenames:
            total_filenames.remove(corrupted_file)

    # reading different subtypes labels
    labels_dict = {}
    with open(labels_path, newline='') as labels_csv:
        reader = csv.reader(labels_csv, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            labels_dict[row[0]] = int(row[1])

    labels = []
    pbar = tqdm(total_filenames, total=len(total_filenames))
    pbar.set_description("reading files and labels")
    for filename in pbar:
        labels.append([int(labels_dict[key]) for key in [filename.split('.')[0] + "_" + x for x in SUBTYPES]])

    labels = np.array(labels)

    # configs-validation split
    train_filenames, validation_filenames, train_labels, validation_labels = train_test_split(total_filenames, labels, test_size=validation_size, random_state=random_state)

    #  saving configs and validation splits filenames and labels into files
    with open(train_file_split_path, "wb") as tf, open(train_label_split_path, "wb") as tl, open(validation_file_split_path, "wb") as vf, open(validation_label_split_path, "wb") as vl:
        pickle.dump(train_filenames, tf), pickle.dump(train_labels, tl), pickle.dump(validation_filenames, vf), pickle.dump(validation_labels, vl)
    return train_filenames, train_labels, validation_filenames, validation_labels


def _get_image_windows(image, windows: [(int, int)], intercept, slope):
    """
    a method that returns a stack of windowed images
    :param image: original intensities of CT scan
    :param windows: list of windowing parameters
    :param intercept: intercept of window
    :param slope: slope of window
    :return: a torch tensor of a stack of different windowed images
    """
    window_images = []
    for window in windows:
        window_images.append(window_image(image, window, intercept, slope))

    return torch.stack(window_images)


def _read_image_3d(file_path: str, do_rotate=False):
    """
    A method to read 3D nifty image
    rotation is applied because the way nifty and dicom arrays are stores is different, and we need a rotation to make them similar
    :param file_path: path to the nifty file
    :param do_rotate: whether to rotate image 90 degrees counter-clock-wise
    :return: 3D tensor of image intensities
    """
    assert file_path is not None, 'file path is needed'
    assert os.path.isfile(file_path), 'wrong file path'

    image = torch.FloatTensor(nibabel.load(filename=file_path).get_fdata())
    if do_rotate:
        image = rotate(image.permute(2, 0, 1), 90).permute(1, 2, 0)
    return image


def _get_first_of_dicom_field_as_int(data):
    # credit: https://www.kaggle.com/code/dcstang/see-like-a-radiologist-with-systematic-windowing/notebook
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(data) == pydicom.multival.MultiValue:
        return int(data[0])
    else:
        return int(data)


def _get_windowing(data):
    # credit: https://www.kaggle.com/code/dcstang/see-like-a-radiologist-with-systematic-windowing/notebook
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [_get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def _read_image_2d(file_path: str):
    """
    a method to read 2D dicom image
    :param file_path: path to the file
    :return: 2D tensor of image intensities and brain-window parameters derived from dicom headers
    """
    assert file_path is not None, 'file path is needed'
    assert os.path.isfile(file_path), 'wrong file path'

    image = pydicom.read_file(file_path)  # reading dicom image
    window_params = _get_windowing(image)  # extracting brain parameters from dicom headers
    image = torch.FloatTensor(image.pixel_array.astype(np.float32))  # creating a float tensor from image intensities
    return image, window_params


def physionet_cross_validation_split(physio_path, extra_path, k=5, override=False):
    """
    a method to create cross-validation folds for PhysioNet ICH dataset based on Stratified K fold division
    :param physio_path: path to the PhysioNet ICH dataset root directory
    :param k: number of folds
    :param override: whether to override the k-fold cross validation if files already exist
    :param extra_path: where to save the folds
    """
    # if overriden is not asked and all folds' divisions exist in the extra path, we do not redo the division.
    if not override:
        files_exist = True
        for i in range(k):
            if not os.path.isfile(os.path.join(extra_path, 'folds_division', f"fold{i}.pt")):
                files_exist = False
        if files_exist:
            return
    ds = PhysioNetICHDataset(physio_path)

    indices = np.arange(0, len(ds.labels))
    # dividing into train/test based on all subtypes
    encoded_labels = LabelEncoder().fit_transform([''.join(str(_l)) for _l in ds.labels])
    skf = StratifiedKFold(k)
    for cf, (train_valid_indices, test_indices) in enumerate(skf.split(indices, encoded_labels)):
        with open(os.path.join(extra_path, 'folds_division', f"fold{cf}.pt"), 'wb') as fold_indices_file:
            pickle.dump(test_indices, fold_indices_file)
