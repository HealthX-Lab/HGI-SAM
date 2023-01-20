import os
from torch.utils.data import Dataset
import torch
import pandas as pd
from preprocessing import window_image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import nibabel
import pickle


def rsna_3d_train_validation_split(root_dir: str, validation_size=0.05, random_state=42, override=False):
    """
       a method that splits the 3D nifti dataset into train and validation set based on the number of slices containing hemorrhage
       we save the split into files for faster computation and further requirements
    """
    train_split_path, validation_split_path = root_dir + '\\' + 'train_split', root_dir + '\\' + 'validation_split'
    if os.path.isfile(train_split_path) and os.path.isfile(validation_split_path) and not override:
        with open(train_split_path, "rb") as tf, open(validation_split_path, "rb") as vf:
            return pickle.load(tf), pickle.load(vf)

    labels_dir = os.path.join(root_dir, 'train_labels')
    filenames, hemorrhages_counts = [], []
    labels = os.listdir(labels_dir)
    for label_filename in tqdm(labels):
        file_path = labels_dir + '\\' + label_filename
        df = pd.read_csv(file_path)
        filenames.append(label_filename.removesuffix('.csv')), hemorrhages_counts.append(sum(df["any"]))

    train_filenames, validation_filenames, train_counts, validation_counts = train_test_split(filenames, hemorrhages_counts, test_size=validation_size, random_state=random_state)
    with open(train_split_path, "wb") as tf, open(validation_split_path, "wb") as vf:
        pickle.dump(train_filenames, tf), pickle.dump(validation_filenames, vf)

    return train_filenames, validation_filenames


def _get_image_windows(image, windows: [(int, int)]):
    window_images = []
    for window in windows:
        window_images.append(window_image(image, window))

    return torch.stack(window_images)


def _read_image(file_path: str):
    assert file_path is not None, 'file path is needed'
    assert os.path.isfile(file_path), 'wrong file path'

    image = torch.FloatTensor(nibabel.load(filename=file_path).get_fdata())
    return image


class RSNAICHDataset3D(Dataset):
    def __init__(self, root_dir, files, windows=None, transform=None):
        """
        Specific pytorch dataset designed for RSNA ICH dataset
        :param root_dir: root directory of PhysioNet dataset
        :param files: filename corresponding to training or validation set images
        :param windows: a list of window_center and window_width
        :param transform: optional transform to be applied on a sample.
        """
        self.train_dir = os.path.join(root_dir, 'stage_2_train')
        self.labels_dir = os.path.join(root_dir, 'train_labels')
        self.files = files
        self.transform = transform
        self.windows = windows

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image_path = os.path.join(self.train_dir, self.files[item] + '.nii')
        label_path = os.path.join(self.labels_dir, self.files[item] + '.csv')

        image = _read_image(image_path)  # x, y, z
        label = pd.read_csv(label_path)
        label = torch.LongTensor(label["any"])

        if self.windows is not None:
            image = _get_image_windows(image, self.windows)  # c, x, y, z

        image = torch.movedim(image, image.dim() - 1, 0)
        if self.transform is not None:
            image = self.transform(image)

        return image, label


class PhysioNetICHDataset3D(Dataset):
    def __init__(self, root_dir, windows=None, transform=None):
        """
        Specific pytorch dataset designed for PhysioNet ICH dataset
        :param root_dir: root directory of PhysioNet dataset
        :param windows: a list of window_center and window_width
        :param transform: optional transform to be applied on a sample.
        """
        self.scans_dir = os.path.join(root_dir, 'ct_scans')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.filenames = os.listdir(self.scans_dir)

        self.transform = transform
        self.windows = windows

        labels_path = os.path.join(root_dir, 'hemorrhage_diagnosis_raw_ct.csv')
        self.labels = pd.read_csv(labels_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        patient_number = int(self.filenames[item].removesuffix('.nii'))
        image_path = os.path.join(self.scans_dir, self.filenames[item])
        mask_path = os.path.join(self.masks_dir, self.filenames[item])

        image = _read_image(image_path)
        mask = _read_image(mask_path)
        label = self.labels[self.labels['PatientNumber'] == patient_number]
        label = 1 - torch.LongTensor(label['No_Hemorrhage'])

        if self.windows is not None:
            image = _get_image_windows(image, self.windows)
        if mask.max() > 0:
            mask = (mask - mask.min()) / (mask.max() - mask.min())

        image = torch.movedim(image, image.dim() - 1, 0)
        mask = torch.movedim(image, mask.dim() - 1, 0)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask.unsqueeze(1)).squeeze()

        return image, mask, label
