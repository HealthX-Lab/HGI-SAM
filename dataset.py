import os
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import nibabel as nib
import cv2
import csv
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

    labels_dir = root_dir + r'\train_labels'
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
    # WINDOWS_CENTER_WIDTH = {
    #     'brain': (40, 80),
    #     'subdural': (80, 200),
    #     'bone': (600, 2800),
    #     'soft': (40, 380),
    #
    #     'brain2': (40, 120),
    #     'bone2': (700, 3200),
    #     'subdural2': (80, 280),
    #
    #     'brain3': (40, 100),
    #     'subdural3': (80, 200),
    #     'bone3': (400, 500),
    window_images = []
    for window in windows:
        window_images.append(window_image(image, window))

    return torch.stack(window_images)


def _read_image(file_path: str):
    assert file_path is not None, 'file path is needed'
    assert os.path.isfile(file_path), 'wrong file path'

    image = torch.tensor(nibabel.load(filename=file_path).get_fdata())
    return image


class RSNAICHDataset(Dataset):
    def __init__(self, root_dir, files, windows=None, transform=None):
        """
        param files (list of string): list of paths to the nifti files.
        param transform (callable, optional): optional transform to be applied on a sample.
        param window (list of tuple): a list of window_center and window_width
        """
        self.train_dir = root_dir + r'\stage_2_train'
        self.labels_dir = root_dir + r'\train_labels'
        self.files = files
        self.transform = transform
        self.windows = windows

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image_path = self.train_dir + '\\' + self.files[item] + '.nii'
        label_path = self.labels_dir + '\\' + self.files[item] + '.csv'

        print(image_path, label_path)
        image = _read_image(image_path)
        label = pd.read_csv(label_path)

        if self.windows is not None:
            image = _get_image_windows(image, self.windows)
        if self.transform is not None:
            image = self.transform(image)

        return image, label


class PhysioNetICHDataset(Dataset):
    def __init__(self, src_dir, labels_state, with_masks_only=False, transform=None, windows=None):
        if windows is None:
            self.windows = ['brain', 'subdural', 'bone']

        self.scans_dir = os.path.join(src_dir, 'ct_scans')
        self.masks_dir = os.path.join(src_dir, 'masks')
        self.labels_path = os.path.join(src_dir, 'hemorrhage_diagnosis_raw_ct.csv')
        self.filenames = os.listdir(self.scans_dir)
        self.slices = []
        self.scans_num_slices = []
        self.masks = []
        self.labels = []
        self.has_masks = []
        self.transform = transform
        self.windows = windows
        self.with_masks_only = with_masks_only

        if labels_state == 0:
            self.SUBTYPES = ["No_Hemorrhage"]
        elif labels_state == 1:
            self.SUBTYPES = ["Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural"]
        else:
            self.SUBTYPES = ["Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural",
                             "No_Hemorrhage"]

        self.read_dataset()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        sample_img = self.slices[item]
        sample_mask = self.masks[item]

        if sample_mask.max() > 0:
            sample_mask = (sample_mask - sample_mask.min()) / (sample_mask.max() - sample_mask.min())

        sample_img = cv2.rotate(sample_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        sample_mask = cv2.rotate(sample_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

        sample_img = get_slice_image(self.windows, file=sample_img)
        sample_mask = torch.tensor(sample_mask.astype(np.float32))

        if self.transform:
            sample_img = self.transform(sample_img)
            sample_mask = self.transform(sample_mask.unsqueeze(0)).squeeze()

        return sample_img, sample_mask, self.labels[item]

    def read_dataset(self):
        with open(self.labels_path, newline='') as labels_csv:
            reader = csv.DictReader(labels_csv)
            for row in reader:
                label = torch.zeros(len(self.SUBTYPES))
                for i, subtype in enumerate(self.SUBTYPES):
                    label[i] = float(row[subtype])
                if 'No_Hemorrhage' in self.SUBTYPES:
                    label[-1] = 1 - label[-1]

                self.labels.append(label)
        k = 0
        for file in self.filenames:
            scan = nib.load(os.path.join(self.scans_dir, file))
            mask = nib.load(os.path.join(self.masks_dir, file))

            scan_data = scan.get_fdata()
            mask_data = mask.get_fdata()
            num_slices = scan_data.shape[2]
            self.scans_num_slices.append(num_slices)

            for i in range(num_slices):
                is_with_mask = mask_data[:, :, i].any()
                if is_with_mask:
                    self.has_masks.append(k)
                elif self.with_masks_only:
                    continue

                self.slices.append(scan_data[:, :, i])
                self.masks.append(mask_data[:, :, i])
                k += 1


def get_files_and_labels(src_path: str, labels_path: str,
                         sub_dataset_percentage=1., labels_state=2, random_state=42):
    assert os.path.isdir(src_path) and os.path.isfile(labels_path), 'wrong src or labels path'
    assert 0 < sub_dataset_percentage <= 1, 'sub dataset percentage must be between 0 and 1'
    assert labels_state in [0, 1, 2], 'labels state must be 0, 1, or 2'
    files, labels = [], []

    if labels_state == 0:
        SUBTYPES = ["any"]
    elif labels_state == 1:
        SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
    else:
        SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]

    total_filenames = os.listdir(src_path)
    corrupted_files = ['ID_6431af929.dcm']
    for corrupted_file in corrupted_files:
        if corrupted_file in total_filenames:
            total_filenames.remove(corrupted_file)

    random.seed(random_state)
    filenames = random.sample(total_filenames, k=int(sub_dataset_percentage * len(total_filenames)))

    labels_dict = {}
    with open(labels_path, newline='') as labels_csv:
        reader = csv.reader(labels_csv, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            labels_dict[row[0]] = float(row[1])

    pbar = tqdm(filenames, total=len(filenames))
    pbar.set_description("reading files and labels")
    for filename in pbar:
        file_path = os.path.join(src_path, filename)
        file_label = []
        for subtype in SUBTYPES:
            key = filename.split('.dcm')[0] + "_" + subtype
            file_label.append(float(labels_dict[key]))

        files.append(file_path)
        labels.append(file_label)

    return files, np.array(labels)


def get_files(src_path: str, sub_dataset_percentage=1., random_state=42):
    assert os.path.isdir(src_path), 'wrong src path'
    total_filenames = os.listdir(src_path)
    corrupted_files = ['ID_6431af929.dcm']
    for corrupted_file in corrupted_files:
        if corrupted_file in total_filenames:
            total_filenames.remove(corrupted_file)

    random.seed(random_state)
    filenames = random.sample(total_filenames, k=int(sub_dataset_percentage * len(total_filenames)))
    files_paths = [os.path.join(src_path, filename) for filename in filenames]

    return files_paths


def create_output_csv(files, predictions, output_path, labels_state=2):
    if labels_state == 0:
        SUBTYPES = ["any"]
    elif labels_state == 1:
        SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
    else:
        SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]

    output = {}
    for file, pred in zip(files, predictions):
        filename = file.split('\\')[-1].split('.')[0]
        for i, subtype in enumerate(SUBTYPES):
            key = filename + '_' + subtype
            value = pred[i].item()
            output[key] = value

    with open(output_path, 'w', newline='') as output_csv:
        writer = csv.writer(output_csv)
        writer.writerow(["ID", "Label"])
        for key, value in output.items():
            writer.writerow([key, value])
