
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        channels = x.shape[1]
        x = x.view(-1, channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, compress=1, device="cuda"):
        super().__init__()
        c_0 = 64 // compress
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, c_0)
        self.down1 = Down(c_0, c_0*2)
        self.sa1 = SelfAttention(c_0*2, 32) # 2nd arg is layer input size
        self.down2 = Down(c_0*2, c_0*4)
        self.sa2 = SelfAttention(c_0*4, 16)
        self.down3 = Down(c_0*4, c_0*4)
        self.sa3 = SelfAttention(c_0*4, 8)

        self.bot1 = DoubleConv(c_0*4, c_0*8)
        self.bot2 = DoubleConv(c_0*8, c_0*8)
        self.bot3 = DoubleConv(c_0*8, c_0*4)

        # Up takes half channels from prev layers and half from Down+SA channels
        self.up1 = Up(c_0*8, c_0*2)
        self.sa4 = SelfAttention(c_0*2, 16)
        self.up2 = Up(c_0*4, c_0)
        self.sa5 = SelfAttention(c_0, 32)
        self.up3 = Up(c_0*2, c_0)
        self.sa6 = SelfAttention(c_0, 64)
        self.outc = nn.Conv2d(c_0, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, compress=1, device="cuda"):
        super().__init__()
        c_0 = 64 // compress
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, c_0)
        self.down1 = Down(c_0, c_0*2)
        self.sa1 = SelfAttention(c_0*2, 32)
        self.down2 = Down(c_0*2, c_0*4)
        self.sa2 = SelfAttention(c_0*4, 16)
        self.down3 = Down(c_0*4, c_0*4)
        self.sa3 = SelfAttention(c_0*4, 8)

        self.bot1 = DoubleConv(c_0*4, c_0*8)
        self.bot2 = DoubleConv(c_0*8, c_0*8)
        self.bot3 = DoubleConv(c_0*8, c_0*4)

        self.up1 = Up(c_0*8, c_0*2)
        self.sa4 = SelfAttention(c_0*2, 16)
        self.up2 = Up(c_0*4, c_0)
        self.sa5 = SelfAttention(c_0, 32)
        self.up3 = Up(c_0*2, c_0)
        self.sa6 = SelfAttention(c_0, 64)
        self.outc = nn.Conv2d(c_0, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            # Ensure y has the correct batch size
            if y.size(0) != x.size(0):
                y = y[:x.size(0)]
            label_embedding = self.label_emb(y)
            # Ensure label embedding size matches t
            if label_embedding.size(0) != t.size(0):
                raise ValueError(f"Label embedding size {label_embedding.size(0)} does not match t size {t.size(0)}")

            t += label_embedding

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=10, device="cpu")
    for name, param in net.named_parameters():
        print(f"{name}: {param.size()}")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)
    print(summary(net, [(3, 64, 64), (1,), (1,)]))
