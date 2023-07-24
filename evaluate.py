import torch
import os
import time
import numpy as np
import nibabel as nib
from dataset import loader_test, ImgTest, norm_01
from collections import OrderedDict
from network import InterpolationNetwork, DiscriminatorForVGG
from utils.util import print_current_losses, evaluate_2D, evaluate_3D

def evaluate(args):
    save_dir = os.path.join('results', args.project_name)
    os.makedirs(os.path.join(save_dir, 'evaluate'), exist_ok=True)
    
    model = InterpolationNetwork()
    model.load_state_dict(torch.load(args.resume, map_location='cpu'))
    model = model.cuda()

    model.eval()

    psnr = []
    ssim = []
    mae = []
    
    for subject in os.listdir(args.data_path):
        data_path = os.path.join(args.data_path, subject)
        subject_dataset = ImgTest(data_path, 2, args.thick_direction, simulate_lr=True)
        subject_loader = loader_test(subject_dataset)
        
        upsampled_data = []
        input_data = []
        for data in subject_loader:
            slice_img_0, slice_img_1 = data
            with torch.no_grad():
                generated_slice = model(slice_img_0.unsqueeze(0).cuda(), slice_img_1.unsqueeze(0).cuda())
            upsampled_data.append(slice_img_0.squeeze().unsqueeze(subject_dataset.axis).cpu().numpy())
            upsampled_data.append(np.clip(generated_slice.squeeze().unsqueeze(subject_dataset.axis).cpu().numpy(), 0, 1))
            input_data.append(slice_img_0.squeeze().unsqueeze(subject_dataset.axis).cpu().numpy())
            input_data.append(slice_img_0.squeeze().unsqueeze(subject_dataset.axis).cpu().numpy())
        # the last slice should be duplicated
        upsampled_data.append(slice_img_1.squeeze().unsqueeze(subject_dataset.axis).cpu().numpy())
        upsampled_data.append(slice_img_1.squeeze().unsqueeze(subject_dataset.axis).cpu().numpy())
        input_data.append(slice_img_1.squeeze().unsqueeze(subject_dataset.axis).cpu().numpy())
        input_data.append(slice_img_1.squeeze().unsqueeze(subject_dataset.axis).cpu().numpy())
        upsampled_data = np.concatenate(upsampled_data, axis=subject_dataset.axis)
        input_data = np.concatenate(input_data, axis=subject_dataset.axis)

        gt_img = subject_dataset.img
        gt_data = np.array(gt_img.dataobj)
        gt_data = norm_01(gt_data)
        out_input_img = nib.Nifti1Image(input_data, gt_img.affine, gt_img.header)
        out_img = nib.Nifti1Image(upsampled_data, gt_img.affine, gt_img.header)
        nib.save(out_input_img, os.path.join(save_dir, 'evaluate', subject.replace('.nii.gz', '_input.nii.gz')))
        nib.save(out_img, os.path.join(save_dir, 'evaluate', subject.replace('.nii.gz', '_prediction.nii.gz')))
        print(upsampled_data.shape, gt_data.shape)
        c_psnr, c_ssim, c_mae = evaluate_3D(upsampled_data, gt_data, data_range=1)
        psnr.append(c_psnr)
        ssim.append(c_ssim)
        mae.append(c_mae)
        print('Subject: {}, PSNR: {:.4f}, SSIM: {:.4f}, MAE: {:.4f}'.format(subject, c_psnr, c_ssim, c_mae))

    print('Average PSNR: {:.4f}, Average SSIM: {:.4f}, Average MAE: {:.4f}'.format(np.mean(psnr), np.mean(ssim), np.mean(mae))




    



        


