import os
import nibabel as nib


def threshold():
    ct_dir = r'C:\physio-ich\ct_scans'
    th_dir = r'C:\physio-ich\windowed'

    for f in os.listdir(ct_dir):
        file_path = os.path.join(ct_dir, f)
        out_path = os.path.join(th_dir, f'{f}.gz')
        file = nib.load(file_path)
        data = file.get_fdata()
        brain_w = 120
        brain_l = 40
        lb = brain_l - brain_w // 2
        up = brain_l + brain_w // 2
        data[data < lb] = lb
        data[data > up] = up
        out_img = nib.Nifti1Image(data, affine=file.affine, header=file.header)
        nib.save(out_img, out_path)


if __name__ == '__main__':
    threshold()
