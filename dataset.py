import os
from torch.utils.data import Dataset
import torch
from preprocessing import get_slice_image
import random
from tqdm import tqdm
import csv
import numpy as np
import nibabel as nib
import cv2


class RSNAICHDataset(Dataset):
    def __init__(self, files, labels=None, transform=None, windows=None):
        """
        param root_dir (string): Directory with all samples. Path to RSNA train or test directories.
        param transform (callable, optional): optional transform to be applied on a sample.
        """
        if windows is None:
            windows = ['brain', 'subdural', 'bone']
        self.files = files
        self.labels = labels
        self.transform = transform
        self.windows = windows

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        sample_path = self.files[item]

        sample_img = get_slice_image(self.windows, file_path=sample_path)

        if self.transform:
            sample_img = self.transform(sample_img)

        if self.labels is None:
            return sample_img
        else:
            return sample_img, 0, self.labels[item]

    def get_first_thousand_samples(self):
        return torch.stack([self.__getitem__(i)[0] for i in range(1000)])


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