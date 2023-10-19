import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
import numpy as np


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts, mode='sum'):
        P = self.batch_pairwise_dist(gts, preds)
        mins_1, _ = torch.min(P, 1)
        mins_2, _ = torch.min(P, 2)
        if mode == 'mean':
            loss_1 = torch.mean(mins_1)
            loss_2 = torch.mean(mins_2)
        elif mode == 'sum':
            loss_1 = torch.sum(mins_1)
            loss_2 = torch.sum(mins_2)
        # print(mins.shape)
        # exit(0)

        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.long()
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def cal_loss_cd(pred, label, mode='sum'):
    pred = pred.permute(0, 2, 1)
    label = label.permute(0, 2, 1)
    chamfer_dist = ChamferLoss()
    loss = chamfer_dist(pred, label, mode)

    return loss


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def write_plyfile(file_name, point_cloud):
    f = open(file_name + '.ply', 'w')
    init_str = "ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex " + str(len(point_cloud)) + \
               "\nproperty float x\nproperty float y\nproperty float z\n" \
               "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
    f.write(init_str)
    for i in range(len(point_cloud)):
        f.write(str(round(float(point_cloud[i][0]), 6)) + ' ' + str(round(float(point_cloud[i][1]), 6)) + ' ' +
                str(round(float(point_cloud[i][2]), 6)) + '\n')
    f.close()


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint
    xyz = xyz.contiguous()

    fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long() # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)

    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


def split_knn_patches(xyz, mask_ratio=0.7, nsample=32, random=True):
    B, N, C = xyz.shape
    npoint = N // nsample
    device = xyz.device

    num_patches = npoint
    num_mask = int(mask_ratio * num_patches)
    num_vis = num_patches - num_mask

    center_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long()
    center_pos = index_points(xyz, center_idx)

    if random:
        shuffle_idx = torch.rand(num_patches, device=device).argsort()
        vis_patch_idx, mask_patch_idx = shuffle_idx[:num_vis], shuffle_idx[num_vis:]

        mask_center_idx = center_idx[:, mask_patch_idx]
        vis_center_idx = center_idx[:, vis_patch_idx]
        mask_center_pos = index_points(xyz, mask_center_idx)
        vis_center_pos = index_points(xyz, vis_center_idx)

        all_patch_idx = knn_point(nsample, xyz, center_pos)  # [B, num_patches, nsample]
        mask_idx = all_patch_idx[:, mask_patch_idx]
        vis_idx = all_patch_idx[:, vis_patch_idx]
        mask_pos = index_points(xyz, mask_idx)  # [B, num_mask, nsample, C]
        vis_pos = index_points(xyz, vis_idx)  # [B, num_vis, nsample, C]
    else:
        mask_point_idx = np.random.randint(num_patches)
        mask_point_pos = center_pos[:, mask_point_idx]
        mask_patch_idx = knn_point(num_mask, center_pos, mask_point_pos.unsqueeze(1)).squeeze()
        vis_patch_idx = torch.empty((B, num_vis), device=device, dtype=int)
        for b in range(B):
            idx_all = set(np.arange(num_patches, dtype=int))
            mask_idx = set(mask_patch_idx[b].tolist())
            vis_idx = idx_all - mask_idx
            vis_patch_idx[b] = torch.tensor(list(vis_idx), device=device, dtype=torch.long)

        shuffle_idx = torch.cat((vis_patch_idx, mask_patch_idx), dim=1).to(device)

        batch_idx = torch.arange(B, device=device).unsqueeze(-1)
        mask_center_idx = center_idx[batch_idx, mask_patch_idx]
        vis_center_idx = center_idx[batch_idx, vis_patch_idx]
        mask_center_pos = index_points(xyz, mask_center_idx)
        vis_center_pos = index_points(xyz, vis_center_idx)

        all_patch_idx = knn_point(nsample, xyz, center_pos)  # [B, num_patches, nsample]
        mask_idx = all_patch_idx[batch_idx, mask_patch_idx]
        vis_idx = all_patch_idx[batch_idx, vis_patch_idx]
        mask_pos = index_points(xyz, mask_idx)  # [B, num_mask, nsample, C]
        vis_pos = index_points(xyz, vis_idx)  # [B, num_vis, nsample, C]

    return mask_pos,  vis_pos,  mask_center_pos, vis_center_pos, mask_patch_idx, vis_patch_idx, shuffle_idx

