from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ScanObjectNN, ScanObjectNN_hardest
from model import MAE3D, MAE3D_cls
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss,  IOStream
import sklearn.metrics as metrics
from tqdm import tqdm
import time


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + file_name):
        os.makedirs('checkpoints/' + file_name)
    if not os.path.exists('checkpoints/' + file_name + '/' + 'models'):
        os.makedirs('checkpoints/' + file_name + '/' + 'models')
    os.system('cp main_cls.py checkpoints' + '/' + file_name + '/' + 'main_cls.py.backup')
    os.system('cp model.py checkpoints' + '/' + file_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + file_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + file_name + '/' + 'data.py.backup')


def train(args, io):
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            ModelNet40(limited_ratio=args.limited_ratio, partition='train', num_points=args.num_points), num_workers=0,
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    elif args.dataset == 'ScanObjectNN_objectonly':
        train_loader = DataLoader(
            ScanObjectNN(root='data/ScanObjectNN/main_split_nobg', partition='train', num_points=args.num_points), num_workers=8,
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(root='data/ScanObjectNN/main_split_nobg', partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'ScanObjectNN_objectbg':
        train_loader = DataLoader(
            ScanObjectNN(root='data/ScanObjectNN/main_split', partition='train', num_points=args.num_points), num_workers=8,
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(root='data/ScanObjectNN/main_split',partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'ScanObjectNN_hardest':
        train_loader = DataLoader(
            ScanObjectNN_hardest(root='data/ScanObjectNN/main_split', partition='train', num_points=args.num_points), num_workers=8,
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN_hardest(root='data/ScanObjectNN/main_split', partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")


    model_pretrain = MAE3D(args, encoder_dims=1024, decoder_dims=1024).to(device)
    model_cls = MAE3D_cls(args, encoder_dims=1024).to(device)
    print(str(model_pretrain))
    model_pretrain = nn.DataParallel(model_pretrain)
    print(str(model_cls))
    model_cls = nn.DataParallel(model_cls)


    if args.pretrained:
        model_pretrain.load_state_dict(torch.load(pretrain_model_path))
        print("Loaded model: %s" % (pretrain_model_path))
        p_keys = []
        f_keys = []
        for p_key in model_pretrain.state_dict():
            if len(p_key.split('.')) >= 3 and p_key.split('.')[2] == 'patch_embed':
                p_keys.append(p_key)
        for f_key in model_cls.state_dict():
            if len(f_key.split('.')) >= 2 and f_key.split('.')[1] == 'patch_embed':
                f_keys.append(f_key)
        for (p_key, f_key) in zip(p_keys, f_keys):
            model_cls.state_dict()[f_key].copy_(model_pretrain.state_dict()[p_key])

    if args.linear_classifier:
        for n, p in model_cls.named_parameters():
            if n.split('.')[1] == 'patch_embed':
                p.requires_grad = False

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD([p for p in model_cls.parameters() if p.requires_grad], lr=args.lr * 100,
                        momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam([p for p in model_cls.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    best_test_acc = 0

    for epoch in range(args.epochs):
        # scheduler.step()
        train_loss = 0.0
        count = 0.0
        model_cls.train()
        train_pred = []
        train_true = []
        total_time = 0.0

        for batch, (data, label) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            opt.zero_grad()
            start_time = time.time()
            logits = model_cls(data)
            loss = cal_loss(logits, label)
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)

            preds = logits.max(dim=1)[1]

            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())


        scheduler.step()
        print('train total time is', total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (
            epoch,
            train_loss * 1.0 / count,
            metrics.accuracy_score(
                train_true, train_pred),
            metrics.balanced_accuracy_score(
                train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model_cls.eval()
        test_pred = []
        test_true = []

        total_time = 0.0
        for data, label in tqdm(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            start_time = time.time()
            logits = model_cls(data)
            loss = cal_loss(logits, label)
            end_time = time.time()
            total_time += (end_time - start_time)
            preds = logits.max(dim=1)[1]

            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        print('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        outstr = 'Test %s, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (
            epoch,
            test_loss * 1.0 / count,
            test_acc,
            avg_per_class_acc)
        io.cprint(outstr)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model_cls.state_dict(), 'checkpoints/%s/models/model_cls.t7' % args.exp_name)
        outstr = 'Best test acc: %.6f' % best_test_acc
        io.cprint(outstr)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")


    model_cls = MAE3D_cls(args, encoder_dims=1024).to(device)
    model_cls = nn.DataParallel(model_cls)
    model_cls.load_state_dict(torch.load(model_path))
    model_cls = model_cls.eval()

    test_loss = 0.0
    count = 0.0
    model_cls.eval()
    test_pred = []
    test_true = []

    total_time = 0.0
    for data, label in tqdm(test_loader):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]

        start_time = time.time()
        logits = model_cls(data)
        loss = cal_loss(logits, label)
        end_time = time.time()
        total_time += (end_time - start_time)
        preds = logits.max(dim=1)[1]

        count += batch_size
        test_loss += loss.item() * batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    print('test total time is', total_time)
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

    outstr = 'Test :: loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (
        test_loss * 1.0 / count,
        test_acc,
        avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Completion Pre-training')
    parser.add_argument('--exp_name', type=str, default='exp_shapenet55_block', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--mask_ratio', type=float, default=0.7, help='masking ratio')
    parser.add_argument('--random', type=bool, default=False, metavar='N', help='random masking')
    parser.add_argument('--patch_size', type=int, default=64, help='patch size')

    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'ScanObjectNN_objectonly', 'ScanObjectNN_objectbg', 'ScanObjectNN_hardest'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=251, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001, 0.1 if using sgd)')  # 0.00000001
    parser.add_argument('--momentum', type=float, default=0.7, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--limited_ratio', type=float, default=1.0,
                        help='dropout rate')

    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--pretrained', type=bool, default=False, metavar='N',
                        help='Restore model from path')

    parser.add_argument('--finetune', type=bool, default=False, metavar='N',
                        help='Restore model from path')
    parser.add_argument('--linear_classifier', type=bool, default=False, metavar='N',
                        help='random mask')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    file_name = 'mask_ratio_' + str(args.mask_ratio) + '/' + args.exp_name
    pretrain_model_path = './checkpoints/mask_ratio_' + str(args.mask_ratio) + '/' + args.exp_name + '/models/model_pretrain.t7'
    model_path = './checkpoints/mask_ratio_' + str(args.mask_ratio) + '/' + args.exp_name + '/models/model_cls.t7'
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

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
