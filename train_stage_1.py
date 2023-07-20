import torch
import os
import numpy as np
from dataset import loader_train
from network import InterpolationNetwork
from utils.util import print_current_losses

def train_stage_1(args):
    save_dir = os.path.join('results', args.project_name)
    os.makedirs(os.path.join(save_dir, 'checkpoint'), exist_ok=True)

    train_dataloader = loader_train(in_path=args.data_path, sample_size=args.sample_size, 
                                    thick_direction=args.thick_direction, batch_size=args.batch_size, is_train=True)
    val_dataloader = loader_train(in_path=args.data_path, sample_size=args.sample_size, 
                                  thick_direction=args.thick_direction, batch_size=args.batch_size, is_train=False)

    model = InterpolationNetwork()
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterionMSE = torch.nn.MSELoss()

    total_iters = 0                # the total number of training iterations

    for epoch in range(args.epochs):
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(train_dataloader):
            slice_img_0, slice_img_1, slice_img_2 = data
            total_iters += args.batch_size
            epoch_iter += args.batch_size

            generated_slice = model(slice_img_0.cuda(), slice_img_2.cuda())
            loss = criterionMSE(generated_slice, slice_img_1.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        if epoch % 10 == 0:
            model.eval()
            val_loss = []
            with torch.no_grad():
                for i, data in enumerate(val_dataloader):
                    slice_img_0, slice_img_1, slice_img_2 = data
                    generated_slice = model(slice_img_0.cuda(), slice_img_2.cuda())
                    loss = criterionMSE(generated_slice, slice_img_1.cuda())
                    val_loss.append(loss.item())

                # save the model
                if args.num_gpus > 1:
                    torch.save(model.module.state_dict(), os.path.join(save_dir, f'checkpoint/model_stage_1_{epoch}.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint/model_stage_1_{epoch}.pth'))

            print('Epoch: {}, Val Loss: {}'.format(epoch, np.mean(val_loss)))
            model.train()




        


