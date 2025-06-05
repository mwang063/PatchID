import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, N, _ = query.size()

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = query.view(B, N, self.num_heads, self.head_dim)
        key = key.view(B, N, self.num_heads, self.head_dim)
        value = value.view(B, N, self.num_heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        score = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attention_weights = F.softmax(score, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(B, N, self.embed_dim)

        output = self.out(output)
        return output


class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads, mlp_ratio=2, dropout=0.1):
        super().__init__()

        # Intro-patch attention
        self.iro_norm1 = nn.LayerNorm(model_dim)
        self.iro_norm2 = nn.LayerNorm(model_dim)
        self.iro_drop1 = nn.Dropout(dropout)
        self.iro_drop2 = nn.Dropout(dropout)
        self.iro_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout)
        self.iro_mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_ratio * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * model_dim, model_dim)
        )

        # Inter-patch attention
        self.ier_norm1 = nn.LayerNorm(model_dim)
        self.ier_norm2 = nn.LayerNorm(model_dim)
        self.ier_drop1 = nn.Dropout(dropout)
        self.ier_drop2 = nn.Dropout(dropout)
        self.ier_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout)
        self.ier_mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_ratio * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * model_dim, model_dim)
        )

    def forward(self, x, mask):
        B, T, R, P, D = x.shape

        # Intro-patch attention
        mask_ = mask.view(B * T * R, P) == 0
        x_ = self.iro_norm1(x.reshape(B * T * R, P, D))
        x = x + self.iro_drop1(self.iro_attn(x_, x_, x_, mask=mask_)).reshape(B, T, R, P, D)
        x = x + self.iro_drop2(self.iro_mlp(self.iro_norm2(x)))

        # Inter-patch attention
        mask_ = mask.transpose(2, 3).contiguous().view(B * T * P, R) == 0
        x_ = self.ier_norm1(x.transpose(2, 3).reshape(B * T * P, R, D))
        x = x + self.ier_drop1(self.ier_attn(x_, x_, x_, mask=mask_)).reshape(B, T, P, R, D).transpose(2, 3)
        x = x + self.ier_drop2(self.ier_mlp(self.ier_norm2(x)))

        return x


def group_by_patch(embedded_x, patch_ids, num_patches):
    """
    :param embedded_x: [B, T, N, D]
    :param patch_ids: [N]
    :param num_patches: R
    :return: patch_feat [B, T, R, P, D], patch_mask [B, T, R, P]
    """
    B, T, N, D = embedded_x.shape
    R = num_patches

    device = embedded_x.device
    patch_masks = [(patch_ids == r) for r in range(R)]
    idx_groups = [torch.nonzero(mask, as_tuple=False).squeeze(1) for mask in patch_masks]

    max_p = max(len(idx) for idx in idx_groups)
    patch_feat = torch.zeros(B, T, R, max_p, D, device=device)
    patch_mask = torch.zeros(B, T, R, max_p, device=device)

    for r in range(R):
        idx = idx_groups[r]
        p_len = len(idx)

        feat_r = embedded_x[:, :, idx, :]
        patch_feat[:, :, r, :p_len, :] = feat_r
        patch_mask[:, :, r, :p_len] = 1.0

    return patch_feat, patch_mask  # [B, T, R, P, D], [B, T, R, P]


def ungroup_patch(patch_feat, patch_ids, N):
    """
    :param patch_feat: [B, T, R, P, D]
    :param patch_ids:  [N]
    :param N:
    :return: embedded_x ∈ [B, T, N, D]
    """
    B, T, R, P, D = patch_feat.shape
    device = patch_feat.device

    idx_groups = [(patch_ids == r).nonzero(as_tuple=False).squeeze(1) for r in range(R)]

    embedded_x = torch.zeros(B, T, N, D, device=device)

    for r, idx in enumerate(idx_groups):
        p_len = len(idx)
        if p_len == 0:
            continue
        embedded_x[:, :, idx, :] = patch_feat[:, :, r, :p_len, :]

    return embedded_x  # [B, T, N, D]


class PatchClassifier(nn.Module):
    def __init__(self, in_dim, num_patches):
        super().__init__()
        self.patch_emb = nn.Parameter(
            torch.empty(num_patches, in_dim))
        nn.init.xavier_uniform_(self.patch_emb)

    def forward(self, node_emb):
        logits = node_emb @ self.patch_emb.transpose(0, 1)
        probs = F.softmax(logits, dim=-1)
        node_emb_ = probs @ self.patch_emb

        index = probs.max(dim=-1, keepdim=True)[1]
        probs_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        probs_ = probs_hard - probs.detach() + probs  # Straight-through trick

        return probs_, node_emb_


class PatchID(nn.Module):
    def __init__(self, input_len, output_len, node_num, tod, dow,
                 layers, patch_num, input_dims, node_dims, tod_dims, dow_dims):
        super(PatchID, self).__init__()
        self.node_num = node_num
        self.tod, self.dow = tod, dow
        self.patch_num = patch_num
        dims = input_dims + node_dims + tod_dims + dow_dims

        self.time_in_day_emb = nn.Parameter(
            torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
            torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.node_emb = nn.Parameter(
            torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)

        self.input_embedding = nn.Conv2d(in_channels=3, out_channels=input_dims, kernel_size=(1, input_len), stride=(1, input_len), bias=True)
        self.patch_classifier = PatchClassifier(node_dims, patch_num)
        self.encoder = nn.ModuleList([
            AttentionLayer(dims, num_heads=1, mlp_ratio=2, dropout=0.1) for _ in range(layers)
        ])
        self.output_layer = nn.Conv2d(in_channels=dims, out_channels=output_len, kernel_size=(1, 1), bias=True)  # 224->12

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def embedding(self, x, node_emb, te):
        x1 = torch.cat([x, (te[..., 0:1]/self.tod), (te[..., 1:2]/self.dow)], -1).float()
        input_data = self.input_embedding(x1.transpose(1, 3)).transpose(1, 3)
        b, t = input_data.shape[0], input_data.shape[1]

        t_i_d_data = te[:, -input_data.shape[1]:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)

        d_i_w_data = te[:, -input_data.shape[1]:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)

        node_emb = node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)
        return input_data

    def forward(self, x, te):
        probs, node_emb = self.patch_classifier(self.node_emb)
        patch_ids = probs.argmax(dim=-1)

        embedded_x = self.embedding(x, node_emb, te)

        patch_feat, patch_mask = group_by_patch(embedded_x, patch_ids, num_patches=self.patch_num)

        for layer in self.encoder:
            patch_feat = layer(patch_feat, patch_mask)

        embedded_x = ungroup_patch(patch_feat, patch_ids, N=self.node_num)

        pred_y = self.output_layer(embedded_x.permute(0, 3, 2, 1))

        return pred_y, probs
