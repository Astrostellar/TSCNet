import torch
import numpy as np
from scipy import ndimage

def blurring_sigma_for_downsampling(current_res, downsample_res, mult_coef=None, thickness=None):
    """Compute standard deviations of 1d gaussian masks for image blurring before downsampling.
    :param downsample_res: slice spacing to downsample to. 
    :param current_res: slice spacing of the volume before downsampling.
    :param mult_coef: (optional) multiplicative coefficient for the blurring kernel. Default is 0.75.
    :param thickness: (optional) slice thickness in each dimension. Must be the same type as downsample_res.
    :return: standard deviation of the blurring masks given as the same type as downsample_res (list or tensor).
    """

    # get blurring resolution (min between downsample_res and thickness)
    
    if thickness is not None:
        downsample_res = min(downsample_res, thickness)
    # get std deviation for blurring kernels
    if downsample_res==current_res:
        sigma = 0.5
    elif mult_coef is None: 
        sigma = 0.75 * downsample_res / current_res
    else:
        sigma = mult_coef * downsample_res / current_res
    return sigma

def gaussian_kernel(sigma):
    windowsize = int(round(2.5 * sigma) / 2) * 2 + 1
    locations = np.arange(0, windowsize) - (windowsize - 1) / 2          
    exp_term = -(locations**2) / (2 * sigma**2)
    kernel = np.exp(exp_term)/(np.sqrt(2 * np.pi) * sigma)
    kernel = kernel / np.sum(kernel)
    return kernel

def gaussian_blur_3d(volume,kernel,axis=2):
    volume=torch.FloatTensor(volume).unsqueeze(1).unsqueeze(1)  # (1,1,w,h,d)
    k=kernel.shape[0]
    if axis==2:
        kernel=torch.FloatTensor(kernel).view(1,1,1,1,k)  # (1,1,1,1,k)
    elif axis==1:
        kernel=torch.FloatTensor(kernel).view(1,1,1,k,1)  # (1,1,1,k,1)
    elif axis==0:
        kernel=torch.FloatTensor(kernel).view(1,1,k,1,1) # (1,1,k,1,1)
    volume_blur=torch.nn.functional.conv3d(volume,kernel,padding=(0,0,k//2))
    return volume_blur.squeeze().numpy()

def gaussian_blur_2d(volume,kernel,axis=2):
    volume=torch.FloatTensor(volume).unsqueeze(1).unsqueeze(1)  # (1,1,w,h)
    k=kernel.shape[0]
    if axis==1:
        kernel=torch.FloatTensor(kernel).view(1,1,1,k)
    elif axis==1:
        kernel=torch.FloatTensor(kernel).view(1,1,k,1)
    volume_blur=torch.nn.functional.conv2d(volume,kernel,padding=(0,k//2))
    return volume_blur.squeeze().numpy()

def downsample(img,current_res=1,downsample_res=3,axis=2):        
    # minic down-sampling process
    sigma=blurring_sigma_for_downsampling(current_res, downsample_res)
    kernel=gaussian_kernel(sigma)
    if len(img.shape)==3:  
        img_blur=gaussian_blur_3d(img,kernel,axis=axis)
        if axis==2:
            img_down=ndimage.zoom(img_blur,(1,1,current_res/downsample_res))
        elif axis==1:
            img_down=ndimage.zoom(img_blur,(1,current_res/downsample_res,1))
        elif axis==0:
            img_down=ndimage.zoom(img_blur,(current_res/downsample_res,1,1))
    elif len(img.shape)==2:  
        img_blur=gaussian_blur_2d(img,kernel,axis=axis)
        if axis==1:
            img_down=ndimage.zoom(img_blur,(1,current_res/downsample_res))  # check if (gaussian blur) and (zoom) applied to the same axis   
        elif axis==0:
            img_down=ndimage.zoom(img_blur,(current_res/downsample_res,1))
    return img_down