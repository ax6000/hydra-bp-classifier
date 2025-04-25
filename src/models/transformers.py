import torch
import torch.nn as nn
import torch.optim as optim

class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(StemBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PatchEmbedding_Conv(nn.Module):
    def __init__(self, *, patch_size, stride, in_channels, emb_dim):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=stride
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # x: [B, C, L]
        x = self.proj(x)  # → [B, emb_dim, num_patches]
        x = x.transpose(1, 2)  # → [B, num_patches, emb_dim]
        return self.norm(x)
    
    
    
    
    
def get_PatchEmbedding_Conv(patch_size, in_channels, emb_dim):
    return PatchEmbedding_Conv(
        patch_size=patch_size,inchannels=in_channels, emb_dim=emb_dim, stride=patch_size)

def get_PatchEmbedding(patch_size, in_channels, emb_dim):
    return nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, emb_dim),
            nn.LayerNorm(emb_dim),
        )
class ViT1D(nn.Module):
    def __init__(self, input_length, patch_size, embed_dim, depth, num_heads, num_classes,get_patch_embed=False):
        super().__init__()
        self.patch_embed = PatchEmbedding_Conv(patch_size, in_dim=1, embed_dim=embed_dim)
        num_patches = input_length // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(num_patches + 1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # → [num_patches, B, embed_dim]

        B = x.shape[1]
        cls_tokens = self.cls_token.expand(-1, B, -1)  # → [1, B, embed_dim]
        x = torch.cat((cls_tokens, x), dim=0)  # prepend CLS token
        x = x + self.pos_embed  # add position encoding

        x = self.transformer(x)  # → [num_patches + 1, B, embed_dim]

        cls_output = x[0]  # CLS token出力を使う
        return self.head(cls_output)  # → [B, num_classes]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.stem = StemBlock(input_dim, model_dim)
        # 1次元信号を受け入れるための線形変換
        self.input_linear = nn.Linear(input_dim, model_dim)
        
        # Transformerのエンコーダ
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
    
        # 最後の出力層
        self.output_linear = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        # 入力信号をモデル次元に変換
        src = self.stem(src)

        
        # Transformerエンコーダに入力
        memory = self.transformer_encoder(src)
    
        # 出力を線形変換
        output = self.output_linear(output)
        
        return output
    
    
    
    