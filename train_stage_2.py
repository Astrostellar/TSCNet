import torch
import os
import time
import numpy as np
from dataset import loader_train
from collections import OrderedDict
from network import InterpolationNetwork, DiscriminatorForVGG
from utils.util import print_current_losses, evaluate_2D

def train_stage_2(args):
    save_dir = os.path.join('results', args.project_name)
    lambda_adv = 0.1
    os.makedirs(os.path.join(save_dir, 'checkpoint'), exist_ok=True)

    if not args.eval_only:
        train_dataloader = loader_train(in_path=args.data_path, sample_size=args.sample_size, 
                                        thick_direction=args.thick_direction, batch_size=args.batch_size, is_train=True, stage=2)
        val_dataloader = loader_train(in_path=args.data_path, sample_size=args.sample_size, 
                                    thick_direction=args.thick_direction, batch_size=args.batch_size, is_train=False, stage=2)
    else:
        test_dataloader = loader_train(in_path=args.data_path, sample_size=args.sample_size,
                                       thick_direction=args.thick_direction, batch_size=args.batch_size, is_train=False, stage=2)
    
    model = InterpolationNetwork()
    discriminator = DiscriminatorForVGG(in_channels=1, out_channels=1, channels=64)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)
    model = model.cuda()
    discriminator = discriminator.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=100, gamma=0.5)

    criterionMSE = torch.nn.MSELoss()
    adversarial_criterion = torch.nn.BCEWithLogitsLoss()
    real_label = torch.full([args.batch_size, 1], 1.0, dtype=torch.float, device='cuda')
    fake_label = torch.full([args.batch_size, 1], 0.0, dtype=torch.float, device='cuda')

    total_iters = 0                # the total number of training iterations
    iter_data_time = time.time()

    model.train()
    discriminator.train()
    if not args.eval_only:
        for epoch in range(args.epochs):
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            for i, data in enumerate(train_dataloader):
                iter_start_time = time.time()
                if i % 10 == 0:
                    t_data = iter_start_time - iter_data_time
                
                slice_img_0, slice_img_1, slice_img_2 = data
                slice_img_0 = slice_img_0.cuda()
                slice_img_1 = slice_img_1.cuda()
                slice_img_2 = slice_img_2.cuda()
                total_iters += args.batch_size
                epoch_iter += args.batch_size

                for d_parameters in discriminator.parameters():
                    d_parameters.requires_grad = False
                model.zero_grad(set_to_none=True)

                generated_slice = model(slice_img_0, slice_img_2)
                loss_mse = criterionMSE(generated_slice, slice_img_1)  # not sure here

                generated_slice_01 = model(slice_img_0, slice_img_1)
                generated_slice_12 = model(slice_img_1, slice_img_2)

                generated_slice_1 = model(generated_slice_01, generated_slice_12)
                adversarial_loss = adversarial_criterion(discriminator(generated_slice_1), real_label)

                loss_cycle = criterionMSE(generated_slice_1, slice_img_1)
                loss = loss_mse + loss_cycle + lambda_adv * adversarial_loss
                loss = loss_cycle + lambda_adv * adversarial_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update the parameters of discriminator
                for d_parameters in discriminator.parameters():
                    d_parameters.requires_grad = True
                discriminator.zero_grad(set_to_none=True)
                gt_output = discriminator(slice_img_1)
                d_loss_gt = adversarial_criterion(gt_output, real_label)
                fake_output = discriminator(generated_slice_1.detach().clone())
                d_loss_fake = adversarial_criterion(fake_output, fake_label)
                d_loss = (d_loss_gt + d_loss_fake) / 2

                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()

                if i % 10 == 0:
                    errors_ret = OrderedDict()
                    errors_ret['loss_mse'] = loss_mse.item()
                    errors_ret['loss_cycle'] = loss_cycle.item()
                    errors_ret['adversarial_loss'] = adversarial_loss.item()
                    errors_ret['d_loss'] = d_loss.item()
                    t_comp = (time.time() - iter_start_time) / args.batch_size
                    print_current_losses(epoch, epoch_iter, errors_ret, t_comp, t_data)

                iter_data_time = time.time()
                
            scheduler.step()
            scheduler_d.step()
            if epoch % 10 == 0:
                model.eval()
                val_loss = []
                c_psnr = 0
                c_ssim = 0
                c_mae = 0
                with torch.no_grad():
                    for i, data in enumerate(val_dataloader):
                        slice_img_0, slice_img_1, slice_img_2 = data
                        generated_slice = model(slice_img_0.cuda(), slice_img_2.cuda())
                        loss = criterionMSE(generated_slice, slice_img_1.cuda())
                        val_loss.append(loss.item())

                        predictions = generated_slice.cpu().numpy()
                        real_B = slice_img_1.cpu().numpy()
                        predictions = np.clip(predictions, 0, 1)
                        real_B = np.clip(real_B, 0, 1)
                        oneBEva = evaluate_2D(predictions, real_B)
                        if oneBEva is None:
                            continue
                        else:
                            c_psnr += oneBEva[0]
                            c_ssim += oneBEva[1]
                            c_mae += oneBEva[2]

                    # save the model
                    if args.num_gpus > 1:
                        torch.save(model.module.state_dict(), os.path.join(save_dir, f'checkpoint/model_stage_2_{epoch}.pth'))
                    else:
                        torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint/model_stage_2_{epoch}.pth'))

                print('Epoch: {}, Val Loss: {:.6}, psnr: {:.6}, ssim: {:.6}, mae" {:.6}'.format(epoch, np.mean(val_loss), c_psnr/(i+1), c_ssim/(i+1), c_mae/(i+1)))
                model.train()




        


