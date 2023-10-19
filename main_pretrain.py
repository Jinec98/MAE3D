from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ShapeNet55
from model import MAE3D
from torch.utils.data import DataLoader
from util import cal_loss_cd, IOStream, write_plyfile
from tqdm import tqdm
import time


def _init_():
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + file_name):
        os.makedirs('checkpoints/' + file_name)
    if not os.path.exists('checkpoints/' + file_name + '/' + 'models'):
        os.makedirs('checkpoints/' + file_name + '/' + 'models')
    os.system('cp main_pretrain.py checkpoints' + '/' + file_name + '/' + 'main_pretrain.py.backup')
    os.system('cp model.py checkpoints' + '/' + file_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + file_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + file_name + '/' + 'data.py.backup')


def train(args, io):
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
    elif args.dataset == 'shapenet55':
        train_loader = DataLoader(ShapeNet55(num_points=args.num_points), num_workers=8,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = MAE3D(args, encoder_dims=1024, decoder_dims=1024).to(device)
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    if args.resume:
        model.load_state_dict(torch.load(args.model_path))
        print("Loaded model: %s" % args.model_path)

    for epoch in range(args.epochs):
        train_loss = 0.0
        chamfer_dist = 0.0
        count = 0.0
        model.train()

        idx = 0
        total_time = 0.0
        for batch, (data, index) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            opt.zero_grad()
            start_time = time.time()

            pred_pc, pred_center, gt_center, vis_pos, crop_pos = model(data)

            loss_cd_center = cal_loss_cd(pred_center, gt_center.permute(0, 2, 1))
            loss_cd_pc = cal_loss_cd(pred_pc, data)
            chamfer_dist_pc = cal_loss_cd(pred_pc, data, mode='mean')

            loss = loss_cd_pc * 0.1 + loss_cd_center
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)

            count += batch_size
            train_loss += loss.item() * batch_size
            chamfer_dist += chamfer_dist_pc.item() * batch_size
            idx += 1

            if epoch == args.epochs - 1 and args.test_visualize:
                for i in range(batch_size):
                    write_plyfile(visualize_path + 'train_' + str(index[i].item()) + '_vis', vis_pos[i].view(-1, 3))
                    write_plyfile(visualize_path + 'train_' + str(index[i].item()) + '_mask', crop_pos[i].view(-1, 3))
                    write_plyfile(visualize_path + 'train_' + str(index[i].item()) + '_gt', data[i].permute(1, 0))
                    write_plyfile(visualize_path + 'train_' + str(index[i].item()) + '_pred', pred_pc[i].permute(1, 0))

        scheduler.step()
        print('train total time is', total_time)

        outstr = 'Train %d\nloss: %.6f\nchamfer distance: %.6f\n' % (
            epoch,
            train_loss * 1.0 / count,
            chamfer_dist * 1.0 / count,)
        io.cprint(outstr)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_%s.t7' % (file_name, str(epoch)))
            print('Model saved.')
        if epoch == args.epochs - 1:
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_pretrain.t7' % file_name)
            print('Last model saved.')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Completion Pre-training')
    parser.add_argument('--exp_name', type=str, default='exp_shapenet55_block', metavar='N', help='Name of the experiment')
    parser.add_argument('--mask_ratio', type=float, default=0.7, help='masking ratio')
    parser.add_argument('--random', type=bool, default=False, metavar='N', help='random masking')
    parser.add_argument('--visualize', type=bool, default=False, help='visualize the point cloud at last epoch')
    parser.add_argument('--num_points', type=int, default=2048, help='num of points to use')
    parser.add_argument('--patch_size', type=int, default=64, help='patch size')

    parser.add_argument('--dataset', type=str, default='shapenet55', metavar='N', choices=['modelnet40', 'shapenet55'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=301, metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001, 0.01 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--model_path', type=str, default='./checkpoints/mask_ratio_0.7_exp' + '/models/model_300.t7', metavar='N', help='restore model path')
    parser.add_argument('--resume', type=bool, default=False, metavar='N', help='Restore model from path')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    file_name = 'mask_ratio_' + str(args.mask_ratio) + '/' + args.exp_name
    visualize_path = './visualize/mask_ratio_' + str(args.mask_ratio) + '/' + args.exp_name + '/'
    _init_()

    io = IOStream('checkpoints/' + file_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(args, io)
