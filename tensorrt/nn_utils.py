import torch
import torch.nn.functional as F

# Minimal ViT model components for inference

def unravel_index(indices, shape):
    indices = indices.long()  # Use long for integer operations
    shape = torch.tensor(shape)
    
    result = []
    indices_copy = indices.clone()  # Don't modify the original
    
    for i in range(len(shape)):
        # Divisor is the product of all remaining dimensions
        div = shape[i+1:].prod() if i < len(shape) - 1 else torch.tensor(1)
        result.append(indices_copy // div)
        indices_copy = indices_copy % div
    
    return torch.stack(result, dim=-1)

class Mlp(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=torch.nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = torch.nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(all_head_dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=torch.nn.GELU, 
                 norm_layer=torch.nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = torch.nn.Identity()  # Simplified for inference
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                                   stride=(patch_size // ratio), padding=4 + 2 * (ratio//2-1))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)

class PartHOEDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Keypoint detection components
        self.conv_for_heatmap1 = torch.nn.ConvTranspose2d(384, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(256)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv_for_heatmap2 = torch.nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.kpts_final = torch.nn.Conv2d(256, 23, kernel_size=1, stride=1, padding=0)
        
        # Orientation estimation components
        self.fc_norm = torch.nn.LayerNorm(384)
        self.hoe_head = torch.nn.Linear(384, 72)
        
        # Confidence estimation components
        self.conv_for_conf1 = torch.nn.Conv2d(384, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conf_bn1 = torch.nn.BatchNorm2d(128)
        self.conf_relu1 = torch.nn.ReLU(inplace=True)
        self.conv_for_conf2 = torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conf_bn2 = torch.nn.BatchNorm2d(128)
        self.conf_relu2 = torch.nn.ReLU(inplace=True)
        self.conf_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.conf_head = torch.nn.Linear(128, 1)
        self.conf_sig = torch.nn.Sigmoid()

    def forward(self, x, Hp, Wp):
        # Orientation estimation
        hoe_feature = self.fc_norm(x.mean(1))
        hoe = self.hoe_head(hoe_feature)
        hoe = F.softmax(hoe, dim=1)
        
        # Reshape for spatial operations
        x = x.permute(0, 2, 1).reshape(hoe_feature.shape[0], -1, Hp, Wp).contiguous()
        
        # Confidence estimation
        conf = self.conv_for_conf1(x)
        conf = self.conf_bn1(conf)
        conf = self.conf_relu1(conf)
        conf = self.conv_for_conf2(conf)
        conf = self.conf_bn2(conf)
        conf = self.conf_relu2(conf)
        conf = self.conf_pool(conf)
        conf = conf.view(conf.size(0), -1)
        conf = self.conf_head(conf)
        conf = self.conf_sig(conf)
        
        # Keypoint heatmaps
        heatmap = self.conv_for_heatmap1(x)
        heatmap = self.bn1(heatmap)
        heatmap = self.relu1(heatmap)
        heatmap = self.conv_for_heatmap2(heatmap)
        heatmap = self.bn2(heatmap)
        heatmap = self.relu2(heatmap)
        kpts = self.kpts_final(heatmap)
        
        return kpts, hoe, conf

class MinimalPartHOE(torch.nn.Module):
    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=384, depth=12, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                     in_chans=in_chans, embed_dim=embed_dim, ratio=1)
        
        # Position embedding (will be loaded from checkpoint)
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = torch.nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True)
            for _ in range(depth)
        ])
        
        self.last_norm = torch.nn.LayerNorm(embed_dim)
        self.decoder = PartHOEDecoder()

    def forward(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.last_norm(x)
        kpts, hoe, conf = self.decoder(x, Hp, Wp)
        
        return kpts, hoe, conf
    
    