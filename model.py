import torch
import torch.nn as nn
import torch.nn.functional as F
from util import split_knn_patches
import numpy as np
from einops import rearrange


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class PatchEmbed_DGCNN(nn.Module):
    def __init__(self, k=20, output_channels=512):
        super(PatchEmbed_DGCNN, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(output_channels)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, output_channels, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        return x



class HeadProj_DGCNN(nn.Module):
    def __init__(self, args, encoder_dims=1024, output_dims=40):
        super(HeadProj_DGCNN, self).__init__()

        self.linear1 = nn.Linear(encoder_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_dims)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class MAE3D_cls(nn.Module):
    def __init__(self, args, encoder_dims=1024):
        super(MAE3D_cls, self).__init__()
        self.patch_embed = PatchEmbed_DGCNN(output_channels=encoder_dims)
        self.head_projection = HeadProj_DGCNN(args)

        self.args = args

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.head_projection(x)
        return x


class MAE3D(nn.Module):
    def __init__(self, args, encoder_dims=1024, decoder_dims=1024):
        super(MAE3D, self).__init__()
        self.args = args
        self.random = args.random
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.patch_num = self.args.num_points // self.args.patch_size
        self.mask_patch_num = int(self.patch_num * self.args.mask_ratio)
        self.vis_patch_num = self.patch_num - self.mask_patch_num

        self.encoder = MAE3DEncoder(args, encoder_dims=encoder_dims)
        self.decoder = MAE3DDecoder(args, decoder_dims=decoder_dims)

        self.token_init = nn.Parameter(torch.randn(decoder_dims, requires_grad=True))
        self.cls_token_encoder = nn.Parameter(torch.randn(1, 1, encoder_dims))
        self.cls_token_decoder = nn.Parameter(torch.randn(1, 1, decoder_dims))

        self.pos_embed_encoder = nn.Sequential(
            nn.Linear(3, encoder_dims),
            nn.ReLU(),
            nn.Linear(encoder_dims, encoder_dims)
        )

        self.pos_embed_decoder = nn.Sequential(
            nn.Linear(3, decoder_dims),
            nn.ReLU(),
            nn.Linear(decoder_dims, decoder_dims)
        )

        self.encoder_to_decoder = nn.Linear(encoder_dims, decoder_dims, bias=False)


        # Point Cloud Reconstruction
        self.num_patch = self.args.patch_size
        self.num_center = int(self.args.num_points / self.num_patch)

        self.mlp_center = nn.Sequential(
            nn.Linear(decoder_dims, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * self.num_center)
        )

        self.mlp_folding = nn.Sequential(
            nn.Conv1d(3 + 2 + 1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )

        # 2D grid
        grids = np.meshgrid(np.linspace(-0.1, 0.1, 8, dtype=np.float32), np.linspace(-0.1, 0.1, 8, dtype=np.float32))
        self.grids = torch.Tensor(np.array(grids)).view(2, -1)  # (2, 4, 4) -> (2, 16)


    def forward(self, x):
        xyz = x.permute(0, 2, 1)  # (32, 1024, 3)
        batch_size, _, _ = x.size()

        mask_pos, vis_pos, mask_center_pos, vis_center_pos, mask_patch_idx, vis_patch_idx, shuffle_idx = split_knn_patches(
            xyz, mask_ratio=self.args.mask_ratio, nsample=self.args.patch_size, random=self.random)
        batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(-1)

        # reshuffle
        center_pos = torch.cat((vis_center_pos, mask_center_pos), dim=1)
        center_pos_reshuffle = torch.empty_like(center_pos, device=center_pos.device)
        center_pos_reshuffle[batch_idx, shuffle_idx] = center_pos
        center_pos = center_pos_reshuffle
        point_pos = torch.cat((vis_pos, mask_pos), dim=1)
        point_pos_reshuffle = torch.empty_like(point_pos, device=point_pos.device)
        point_pos_reshuffle[batch_idx, shuffle_idx] = point_pos
        point_pos = point_pos_reshuffle


        mask_pos = point_pos[batch_idx, mask_patch_idx]
        vis_pos = point_pos[batch_idx, vis_patch_idx]
        vis_pos_input = vis_pos.view(batch_size, self.vis_patch_num * self.args.patch_size, 3)

        x_pos_embed_encoder = self.pos_embed_encoder(center_pos)
        x_pos_embed_decoder = self.pos_embed_decoder(center_pos)
        x_pos_embed_encoder = x_pos_embed_encoder[batch_idx, vis_patch_idx].view(batch_size, -1, self.encoder_dims)
        x_pos_embed_encoder = torch.cat((self.cls_token_encoder.repeat(batch_size, 1, 1), x_pos_embed_encoder), dim=1)
        x_pos_embed_decoder = x_pos_embed_decoder.view(batch_size, -1, self.decoder_dims)
        x_pos_embed_decoder = torch.cat((self.cls_token_decoder.repeat(batch_size, 1, 1), x_pos_embed_decoder), dim=1)

        x_vis = self.encoder(vis_pos_input, x_pos_embed_encoder)
        x_vis = self.encoder_to_decoder(x_vis)

        x_full_init = self.token_init[None, None, :].repeat(batch_size, self.patch_num, 1)
        x_mask_init = x_full_init[batch_idx, mask_patch_idx]

        x_full = torch.cat((x_vis, x_mask_init), dim=1)
        x_full_reshuffle = torch.empty_like(x_full, device=x_full.device)
        x_full_reshuffle[batch_idx, shuffle_idx] = x_full

        x_full = self.decoder(x_full_reshuffle, x_pos_embed_decoder)
        
        # reconstruction
        x_center = self.mlp_center(x_full)
        x_center = x_center.view(-1, 3, self.num_center)
        center = x_center.unsqueeze(3).repeat(1, 1, 1, self.num_patch).view(batch_size, 3, -1)
        feature = x_full.unsqueeze(2).repeat(1, 1, self.num_patch * self.num_center)
        grids = self.grids.to(feature.device)
        grids = grids.unsqueeze(0).repeat(batch_size, 1, self.num_center)

        x = torch.cat((feature, grids, center), dim=1)
        x = self.mlp_folding(x)
        x_outputs = x + center

        return x_outputs, x_center, center_pos, vis_pos, mask_pos


class MAE3DEncoder(nn.Module):
    def __init__(self, args, encoder_dims=1024, mlp_dim=2048, depth=6, num_heads=8, dim_per_head=64, dropout=0.):
        super(MAE3DEncoder, self).__init__()
        self.args = args
        self.encoder_dims = encoder_dims

        # Patch Feature Embedding
        self.patch_embed = PatchEmbed_DGCNN(output_channels=encoder_dims)

        # MAE3D Transformers
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_dims))
        self.transformer = Transformer(
            encoder_dims, mlp_dim, depth=depth, num_heads=num_heads,
            dim_per_head=dim_per_head, dropout=dropout
        )

    def forward(self, x, x_pos_embed):
        batch_size, num_points, _ = x.size()

        x = self.patch_embed(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = rearrange(x, 'b (p n) c -> (b p) n c', b=batch_size, n=self.args.patch_size, c=self.encoder_dims)
        x = torch.max(x, 1)[0]
        x = x.view(batch_size, -1, self.encoder_dims)

        # transformer
        x = torch.cat((self.cls_token.repeat(batch_size, 1, 1), x), dim=1)
        x_encoder = self.transformer(x, x_pos_embed)  # ([32, 11, 256])
        return x_encoder[:, 1:]


class MAE3DDecoder(nn.Module):
    def __init__(self, args, decoder_dims=1024, mlp_dim=2048, depth=6, num_heads=8, dim_per_head=64, dropout=0.):
        super(MAE3DDecoder, self).__init__()
        self.args = args
        self.decoder_dims = decoder_dims
        self.patch_num = self.args.num_points // self.args.patch_size
        self.mask_patch_num = int(self.patch_num * self.args.mask_ratio)
        self.vis_patch_num = self.patch_num - self.mask_patch_num

        self.cls_token = nn.Parameter(torch.randn(1, 1, decoder_dims))
        self.transformer = Transformer(
            decoder_dims, mlp_dim, depth=depth, num_heads=num_heads,
            dim_per_head=dim_per_head, dropout=dropout
        )

    def forward(self, x, x_pos_embed):
        # transformer
        batch_size, _, _ = x.size()
        x = torch.cat((self.cls_token.repeat(batch_size, 1, 1), x), dim=1)
        x_decoder = self.transformer(x, x_pos_embed)
        x = x_decoder[:, 1:]

        x = x.permute(0, 2, 1)
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = net

    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, l, d = x.shape

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv = self.to_qkv(x)
        # (3,b,num_heads,l,dim_per_head)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        q, k, v = qkv.chunk(3)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        '''ii. Attention computation'''
        attn = self.attend(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale
        )

        '''iii. Put attention on Value & reshape'''
        # (b,num_heads,l,dim_per_head)
        z = torch.matmul(attn, v)
        # (b,num_heads,l,dim_per_head)->(b,l,num_heads,dim_per_head)->(b,l,dim_all_heads)
        z = z.transpose(1, 2).reshape(b, l, -1)
        # assert z.size(-1) == q.size(-1) * self.num_heads

        '''iv. Project out'''
        # (b,l,dim_all_heads)->(b,l,dim)
        out = self.out(z)
        # assert out.size(-1) == d

        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth=6, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, pos_emd):
        for norm_attn, norm_ffn in self.layers:
            x = x + pos_emd
            x = x + norm_attn(x)
            x = x + norm_ffn(x)

        return x