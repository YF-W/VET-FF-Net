import torch
from torch import nn


class PatchEmbedding(nn.Module):

    def __init__(self,image_size,in_channels,patch_size,norm_layer=None):
        super(PatchEmbedding, self).__init__()
        image_size=(image_size,image_size)
        patch_size=(patch_size,patch_size)

        self.image_size=image_size
        self.patch_size=patch_size
        self.dim_patch=in_channels*self.patch_size[0]*self.patch_size[1]
        self.grid_size=(self.image_size[0]//self.patch_size[0],self.image_size[1]//self.patch_size[1])
        self.num_patchs=self.grid_size[0]*self.grid_size[1]

        self.process=nn.Conv2d(in_channels,self.dim_patch,kernel_size=patch_size,stride=patch_size)
        self.norm=norm_layer(self.dim_patch) if norm_layer else nn.Identity()


    def forward(self,x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."

        x=self.process(x).flatten(2).transpose(2,1)
        x=self.norm(x)
        return x

class VE_MultiHead_SelfAttention(nn.Module):
    def __init__(self,dim_patch,num_heads,qkv_bias=False,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super(VE_MultiHead_SelfAttention, self).__init__()
        self.num_heads=num_heads

        head_dim=dim_patch//self.num_heads
        self.scale=qk_scale or head_dim**-0.5

        self.qkv=nn.Linear(dim_patch,dim_patch*3,bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)

        self.proj=nn.Linear(dim_patch,dim_patch)
        self.proj_drop=nn.Dropout(proj_drop)

        self.globalMaxpool=nn.AdaptiveMaxPool2d((1,1))
        self.NMaxpool=nn.AdaptiveMaxPool2d((196,1))
        self.globalAveragepool=nn.AdaptiveAvgPool2d((1,1))
        self.NAveragepool=nn.AdaptiveAvgPool2d((196,1))

    def forward(self,x,x_Exm):
        N,D=x.shape[1:]

        maxg=self.globalMaxpool(x)
        maxn=self.NMaxpool(x)
        aveg=self.globalAveragepool(x)
        aven=self.NAveragepool(x)

        qkv=self.qkv(x)
        qkv[:,:,2*D:]=qkv[:,:,2*D:].clone() * nn.functional.softmax(maxg*maxn,dim=1) * nn.functional.softmax(aveg*aven)         #V_improve         加上clone()是遇到报错了
        qkv[:, :, 2 * D:]=qkv[:,:,2*D:].clone()+x_Exm        #V_improve

        qkv=qkv.reshape((-1,N,3,self.num_heads,D//self.num_heads)).permute((2,0,3,1,4))

        q,k,v=qkv[0],qkv[1],qkv[2]

        #K_improve
        kv_Ex=v.matmul(k.permute(0,1,3,2))

        # Scaled Dot-Product Attention
        # Matmul + Scale
        attention=((q.matmul(k.permute(0,1,3,2))+kv_Ex))*self.scale

        # SoftMax
        attention = nn.functional.softmax(attention,dim=-1)
        attention = self.attn_drop(attention)

        # Matmul
        x = (attention.matmul(v)).permute((0, 2, 1, 3)).reshape((-1, N, D))

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop_rate=0.):
        super(MLP, self).__init__()
        self.out_features=out_features or hidden_features
        self.hidden_features=hidden_features or in_features

        self.fc1=nn.Linear(in_features,self.hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(self.hidden_features,self.out_features)
        self.drop=nn.Dropout(drop_rate)

    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x

class Depthwise_Separable_PatchEmbedding(nn.Module):
    def __init__(self,in_channels,out_channels,dilation=1,kernel_size=16,stride=16):
        super(Depthwise_Separable_PatchEmbedding, self).__init__()

        self.depth_conv=nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=dilation,
                                    dilation=dilation,
                                    groups=in_channels)


        self.point_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self,x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        out=out.flatten(2).transpose(2,1)

        return out


class VE_ViT_Block(nn.Module):
    def __init__(self):
        super(VE_ViT_Block, self).__init__()

        #position_embed
        self.pos_embedding = nn.Parameter(torch.randn(1, 196, 3*16*16))             #初始化为0还是范围随机初始化

        #MHSA_improve
        self.MHSA_improve=VE_MultiHead_SelfAttention(dim_patch=16*16*3,num_heads=8)

        #Mlp
        self.Mlp=MLP(in_features=3*16*16,hidden_features=4*3*16*16,out_features=3*16*16)

        #Layer_norm
        self.norm_1=nn.LayerNorm(16*16*3)
        self.norm_2=nn.LayerNorm(16*16*3)

        #Extended extraction module
        self.Eem=Depthwise_Separable_PatchEmbedding(3,16*16*3,kernel_size=16,stride=16)


        self.drop_1=nn.Dropout(0.)          #使用的drop方式和超参数大小
        self.drop_2=nn.Dropout(0.)

    def forward(self,x,x_Eem):
        x_Eem=self.Eem(x_Eem)
        x=x+self.pos_embedding
        x=self.drop_1(x+self.MHSA_improve(self.norm_1(x),x_Eem))
        x=self.drop_2(x+self.Mlp(self.norm_2(x)))

        return x


class Channel_Concatenation_Attention_Branch(nn.Module):
    def __init__(self, in_channels):
        super(Channel_Concatenation_Attention_Branch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // 16, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class Spatial_Addition_Attention_Branch(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Addition_Attention_Branch, self).__init__()

        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        min_out, _ = torch.min(x,dim=1,keepdim=True)
        x = torch.cat([max_out, avg_out, min_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class TGAD_FM(nn.Module):
    def __init__(self,channels):
        super(TGAD_FM, self).__init__()

        self.Channel_A=Channel_Concatenation_Attention_Branch((2*channels))
        self.Spatial_A=Spatial_Addition_Attention_Branch()
        self.trans_1=nn.Conv2d((2*channels),channels,kernel_size=1)

    def forward(self,x_t,x_c):

        x_1=torch.cat((x_t,x_c),dim=1)
        x_A_1=x_1*self.Channel_A(x_1)
        x_A_1=self.trans_1(x_A_1)

        x_2=x_t.mul(x_c)            #Information Reinforcement and Inhibition Branch
        x_A_2=x_2*nn.functional.softmax(x_2,dim=1)

        x_3=x_t+x_c
        x_A_3=x_3*self.Spatial_A(x_3)

        return x_A_1 + x_A_2 + x_A_3



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Final_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Final_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)








# def test():
#     model=VE_ViT_Block()
#     x=torch.Tensor(3,3,224,224)
#     preds=model(x)
#     print("shape of preds :",preds.shape)
#
# if __name__ =='__main__':
#     test()
    