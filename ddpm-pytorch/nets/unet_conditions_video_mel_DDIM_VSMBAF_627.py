import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):  
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")
    
#------------------------------------------#
#   计算时间步长的位置嵌入。
#   一半为sin，一半为cos。
#------------------------------------------#
class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device      = x.device
        half_dim    = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # x * self.scale和emb外积
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

#------------------------------------------#
#   下采样层，一个步长为2x2的卷积
#------------------------------------------#
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x, x_f,time_emb, y, video):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x),self.downsample(x_f)

#------------------------------------------#
#   上采样层，Upsample+卷积
#------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
        
    def forward(self, x, x_f,time_emb, y,video):
        return self.upsample(x),self.upsample(x_f)

#------------------------------------------#
#   使用Self-Attention注意力机制
#   做一个全局的Self-Attention
#------------------------------------------#
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()

        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        
        self.to_q = nn.Conv2d(in_channels, in_channels, 1)
        self.to_kv = nn.Conv2d(in_channels, in_channels * 2, 1)
        
        self.v_c =  nn.Conv2d(1, in_channels, 1)
        self.v_l = nn.Linear(512, 100)
        
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)
        

    def forward(self, x,x_f,video):
        b, c, h, w  = x.shape
        k, v     = torch.split(self.to_kv(self.norm(x_f)), self.in_channels, dim=1)
        qf = self.to_q(self.norm(x))
      
        qf = qf.permute(0, 2, 3, 1).view(b, h * w, c)
        #vf = x_f.permute(0, 2, 3, 1).view(b, h * w, c)
        

        k = k.permute(0, 1, 2, 3).view(b, c, h * w)
        
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        
        

        #dot_products = torch.bmm(q, k) * (c ** (-0.5))
        #assert dot_products.shape == (b, h * w, h * w)
        
        dot_products1 = torch.bmm(qf, k) * (c ** (-0.5))
        
        attention1   = torch.softmax(dot_products1, dim=-1)
        out         = torch.bmm(attention1, v)
        #out_f         = torch.bmm(attention1, vf)
        
        assert out.shape == (b, h * w, c)
        out         = out.view(b, h, w, c).permute(0, 3, 1, 2)
        #out_f         = out_f.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out)+x,x_f
    
#------------------------------------------#
#   用于特征提取的残差结构
#------------------------------------------#
class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dropout, time_emb_dim=None, num_classes=None, activation=SiLU(),
        norm="gn", num_groups=32, use_attention=False,video_dim=512,
    ):
        super().__init__()

        self.activation = activation

        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout), 
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias  = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        
        
        #self.video_bias = nn.Linear(video_dim, out_channels)
        
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        self.residual_connection    = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention              =  AttentionBlock(out_channels, norm, num_groups)#nn.Identity() if not use_attention else
    
    def forward(self, x, x_f,time_emb=None, y=None,video=None):
        out = self.activation(self.norm_1(x))
        # 第一个卷积
        out = self.conv_1(out)
        
        out_f = self.activation(self.norm_1(x_f))
        # 第一个卷积
        out_f = self.conv_1(out_f)
        
        # 对时间time_emb做一个全连接，施加在通道上
        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]


        # 对种类y_emb做一个全连接，施加在通道上
        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")

            out += self.class_bias(y)[:, :, None, None]

        out = self.activation(self.norm_2(out))
        # 第二个卷积+残差边
        out = self.conv_2(out) + self.residual_connection(x)
        out_f = self.activation(self.norm_2(out_f))
        # 第二个卷积+残差边
        out_f = self.conv_2(out_f) + self.residual_connection(x_f)
        # 最后做个Attention
        
        out,_ = self.attention(out,out,video)
        out,out_f = self.attention(out,out_f,video)
        
        return out,out_f

#------------------------------------------#
#   Unet模型
#------------------------------------------#
class UNet(nn.Module):
    def __init__(
        self, img_channels, base_channels=128, channel_mults=(1, 2, 4, 8),
        num_res_blocks=3, time_emb_dim=128 * 4, time_emb_scale=1.0, num_classes=None, activation=SiLU(),
        dropout=0.1, attention_resolutions=(0,1,2,3), norm="gn", num_groups=8, initial_pad=0,base_video_channels=512,video_dim=512
    ):
        super().__init__()
        # 使用到的激活函数，一般为SILU
        n_filters=16
        self.activation = activation
        # 是否对输入进行padding
        self.initial_pad = initial_pad
        # 需要去区分的类别数
        self.num_classes = num_classes
        
        self.base_channels = base_channels
        # 对时间轴输入的全连接层
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(base_channels,
                            time_emb_dim),
            torch.nn.Linear(time_emb_dim,
                            time_emb_dim),
        ])

        self.Vconv1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=3, stride=1, padding=1)
        
        # 第二个卷积层，使用1x1卷积减少特征图数量到 n/2
        self.Vconv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters//2, kernel_size=1, stride=1)
        
        self.Vconv1x1 = nn.Conv2d(in_channels=512, out_channels=80, kernel_size=1, stride=1, padding=0)
        
        # 第三个卷积层，再次使用1x1卷积减少特征图数量到 1
        self.Vconv3 = nn.Conv2d(in_channels=n_filters//2, out_channels=1, kernel_size=1, stride=1)
        
        self.time_bias  = nn.Linear(time_emb_dim, base_channels) if time_emb_dim is not None else None
        
    
        # 对输入图片的第一个卷积
        self.init_conv  = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # self.downs用于存储下采样用到的层，首先利用ResidualBlock提取特征
        # 然后利用Downsample降低特征图的高宽
        self.downs      = nn.ModuleList()
        self.ups        = nn.ModuleList()
        
        # channels指的是每一个模块处理后的通道数
        # now_channels是一个中间变量，代表中间的通道数
        channels        = [base_channels]
        now_channels    = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                
                self.downs.append(
                    ResidualBlock(
                        now_channels, out_channels, dropout,
                        time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
                        norm=norm, num_groups=num_groups, use_attention=i in attention_resolutions,video_dim=video_dim,
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)
            
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        # 可以看作是特征整合，中间的一个特征提取模块
        self.mid = nn.ModuleList(
            [
                ResidualBlock(
                    now_channels, now_channels, dropout,
                    time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
                    norm=norm, num_groups=num_groups, use_attention=True,video_dim=video_dim,
                ),
                ResidualBlock(
                    now_channels, now_channels, dropout,
                    time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation, 
                    norm=norm, num_groups=num_groups, use_attention=False,video_dim=video_dim,
                ),
            ]
        )

        # 进行上采样，进行特征融合
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels, out_channels, dropout, 
                    time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation, 
                    norm=norm, num_groups=num_groups, use_attention=i in attention_resolutions,video_dim=video_dim,
                ))
                now_channels = out_channels
            
            if i != 0:
                self.ups.append(Upsample(now_channels))
        
        assert len(channels) == 0
        
        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)
        self.attention              =  AttentionBlock(base_channels, norm, 1)
    
    def forward(self, x,x_now=None,time=None, y=None,video=None):
        # 是否对输入进行padding
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        # 对时间轴输入的全连接层
        temb = get_timestep_embedding(time, self.base_channels)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        time_emb = self.temb.dense[1](temb)
#------------------------------------------------------------------------------------
        video = video.unsqueeze(1)

# 步骤2：使用repeat来复制数据，形状变为 b*3*1*512
        video = video.repeat(1, 1, 80, 1)
    
        video = self.activation(self.Vconv1(video))
        
        video = video.permute(0, 3, 1, 2)  # 现在形状是 [10, 512, 3, 80]

# 应用1x1卷积层
        output_tensor = self.activation(self.Vconv1x1(video))

# 再次交换轴，将通道数放回最后
        video = output_tensor.permute(0, 2, 3, 1)  # 形状变回 [10, 3, 80, 80]


        # 第二个卷积层
        video = self.activation(self.Vconv2(video))
        # 第三个卷积层
        video_emb = self.Vconv3(video)
        
        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")
        
        # 对输入图片的第一个卷积
        x = self.init_conv(x)
        x_t = x+self.time_bias(self.activation(time_emb))[:, :, None, None]
        
        video_emb = self.init_conv(video_emb)
        video_at,_ = self.attention(video_emb,x_t,None)
        
        x_f = video_at+ self.init_conv(x_now)
        
        

        # skips用于存放下采样的中间层
        skips = [x]
        skips_f = [x_f]
        
        for layer in self.downs:
            x,x_f = layer(x, x_f,time_emb, y,video_emb)
            skips.append(x)
            skips_f.append(x_f)
        
        # 特征整合与提取
        for layer in self.mid:
            x,x_f = layer(x, x_f,time_emb, y,video_emb)
        
        # 上采样并进行特征融合
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x_f = torch.cat([x_f, skips_f.pop()], dim=1)
            x,x_f = layer(x,x_f, time_emb, y,video_emb)

        # 上采样并进行特征融合
        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)
        
        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x,video_at