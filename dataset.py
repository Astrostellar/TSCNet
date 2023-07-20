import os
from tqdm import tqdm
import numpy as np
import random
import nibabel as nib
from torch.utils import data
from scipy import ndimage as nd
from utils.resample import downsample

def norm_01(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def determine_axis(orientation, direction='axial'):
    if direction == 'sagittal':
        return orientation.index('R') if 'R' in orientation else orientation.index('L')
    elif direction == 'coronal':
        return orientation.index('A') if 'A' in orientation else orientation.index('P')
    elif direction == 'axial':
        return orientation.index('S') if 'S' in orientation else orientation.index('I')

def read_img(in_path, direction='axial', simulate_lr=True):
    img_list = []
    axis_list = []
    filenames = os.listdir(in_path)
    for f in tqdm(filenames):
        img = nib.load(os.path.join(in_path, f))
        orientation = nib.aff2axcodes(img.affine)
        axis = determine_axis(orientation, direction=direction)
        img_vol = np.array(img.dataobj)
        img_vol = norm_01(img_vol)
        if simulate_lr:
            img_vol = downsample(img_vol, current_res=1, downsample_res=2, axis=axis)
        img_list.append(img_vol)
        axis_list.append(axis)
    return img_list, axis_list

# -----------------------
# Training data
# -----------------------

class ImgTrain(data.Dataset):
    def __init__(self, in_path, sample_size, thick_direction, simulate_lr=True, is_train=True, stage=1):
        self.sample_size = sample_size
        self.patch, self.axis = read_img(in_path=in_path, direction=thick_direction, simulate_lr=simulate_lr)
        self.is_train = is_train
        self.stage = stage

    def __len__(self):
        return len(self.patch)

    def __getitem__(self, item):
        subject_img = self.patch[item]
        thick_axis = self.axis[item]
        # randomly choice a slice to resample
        if self.is_train and self.stage == 1:
            down_axis = random.choice(np.delete(np.arange(3), thick_axis))
        else:
            down_axis = thick_axis
        # randomly choice a slice along the down_axis
        slice_idx_0 = random.randint(0, subject_img.shape[down_axis] - 3)
        # get the 2D slice
        slice_img_0 = np.take(subject_img, slice_idx_0, axis=down_axis)

        slice_idx_1 = slice_idx_0 + 1 # the median slice
        slice_img_1 = np.take(subject_img, slice_idx_1, axis=down_axis)

        slice_idx_2 = slice_idx_0 + 2 # the next slice
        slice_img_2 = np.take(subject_img, slice_idx_2, axis=down_axis)
        
        # zero_pad to 256
        if slice_img_0.shape[0] < self.sample_size:
            slice_img_0 = np.pad(slice_img_0, ((0, self.sample_size - slice_img_0.shape[0]), (0, 0)), 'constant', constant_values=0)
            slice_img_1 = np.pad(slice_img_1, ((0, self.sample_size - slice_img_1.shape[0]), (0, 0)), 'constant', constant_values=0)
            slice_img_2 = np.pad(slice_img_2, ((0, self.sample_size - slice_img_2.shape[0]), (0, 0)), 'constant', constant_values=0)
        if slice_img_0.shape[1] < self.sample_size:
            slice_img_0 = np.pad(slice_img_0, ((0, 0), (0, self.sample_size - slice_img_0.shape[1])), 'constant', constant_values=0)
            slice_img_1 = np.pad(slice_img_1, ((0, 0), (0, self.sample_size - slice_img_1.shape[1])), 'constant', constant_values=0)
            slice_img_2 = np.pad(slice_img_2, ((0, 0), (0, self.sample_size - slice_img_2.shape[1])), 'constant', constant_values=0)
        
        # random crop
        h, w = slice_img_0.shape
        x = random.randint(0, h - self.sample_size)
        y = random.randint(0, w - self.sample_size)
        slice_img_0 = slice_img_0[np.newaxis, x:x + self.sample_size, y:y + self.sample_size]
        slice_img_1 = slice_img_1[np.newaxis, x:x + self.sample_size, y:y + self.sample_size]
        slice_img_2 = slice_img_2[np.newaxis, x:x + self.sample_size, y:y + self.sample_size]

        return slice_img_0, slice_img_1, slice_img_2


def loader_train(in_path, batch_size, thick_direction, sample_size, is_train, stage=1):
    """
    :param in_path_hr: the path of HR patches
    :param batch_size: N in Equ. 3
    :param sample_size: K in Equ. 3
    :param is_train:
    :return:
    """
    return data.DataLoader(
        dataset=ImgTrain(in_path=in_path, sample_size=sample_size, thick_direction=thick_direction, simulate_lr=is_train, is_train=is_train, stage=stage),
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train
    )


# -----------------------
# Testing data
# -----------------------

class ImgTest(data.Dataset):
    def __init__(self, in_path_lr, scale):
        self.img_lr = []
        self.xyz_hr = []
        # load lr image
        lr_vol = sitk.GetArrayFromImage(sitk.ReadImage(in_path_lr))
        lr_vol = lr_vol[:-10,:,:]
        print(lr_vol.shape)
        lr_vol = nd.interpolation.zoom(lr_vol, [1 / scale, 1, 1], order=3)
        print(lr_vol.shape)
        self.img_lr.append(lr_vol)
        for img_lr in self.img_lr:
            temp_size = np.array(img_lr.shape).astype(float)
            temp_size[0] = temp_size[0] * scale
            temp_size = list(temp_size.astype(int))
            self.xyz_hr.append(utils.make_coord(temp_size, flatten=True))

    def __len__(self):
        return len(self.img_lr)

    def __getitem__(self, item):
        return self.img_lr[item], self.xyz_hr[item]


def loader_test(in_path_lr, scale):
    return data.DataLoader(
        dataset=ImgTest(in_path_lr=in_path_lr, scale=scale),
        batch_size=1,
        shuffle=False
    )
