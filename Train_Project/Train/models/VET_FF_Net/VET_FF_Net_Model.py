import torch
from torch import nn
from torchvision import models as resnet_model
from models.VET_FF_Net.VET_FF_Net_Configs import PatchEmbedding,VE_ViT_Block,TGAD_FM,DoubleConv,Final_conv
from thop import profile
import warnings

warnings.filterwarnings("ignore" , category=UserWarning)

class VET_FF_Net(nn.Module):
    def __init__(self):
        super(VET_FF_Net, self).__init__()

        #Encoder
        #patch_embed
        self.patch_embed = PatchEmbedding(in_channels=3,image_size=224,patch_size=16)

        #Transformer_encoder    depth=4
        self.Transformer_encoder = nn.ModuleList()
        for i in range(4):
            self.Transformer_encoder.append(VE_ViT_Block())

        #CNN_encoder
        resnet = resnet_model.resnet34(pretrained=True)
        self.CNN_encoder = nn.ModuleList()
        for i in ['conv1','bn1','relu','layer1','layer2','layer3','layer4']:
            self.CNN_encoder.append(eval('resnet.{}'.format(i)))

        #Fusion
        self.Fusion_modules=nn.ModuleList()
        for i in [64,128,256,512]:
            self.Fusion_modules.append(TGAD_FM(i))

        #Trans_modules
        self.Trans_modules=nn.ModuleList()
        for i,j in [[64,8],[128,4],[256,2]]:
            self.Trans_modules.append(nn.ConvTranspose2d(768, i, kernel_size=j, stride=j))
        self.Trans_modules.append(nn.Conv2d(768,512,kernel_size=1))

        #Decoder
        self.decoders=nn.ModuleList()
        self.decoders.append(DoubleConv(3 * 512, 512))
        self.decoders.append(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        for i in [256,128,64]:
            self.decoders.append(DoubleConv(4*i,i))
            self.decoders.append(nn.ConvTranspose2d(i, i//2, kernel_size=2, stride=2))

        self.final_conv=Final_conv(32,1)


    def forward(self,x):

        #Encoder
        #(1)CNN_encoder
        CNN_features=[]
        x_C=self.CNN_encoder[0](x)
        x_C=self.CNN_encoder[1](x_C)
        x_C=self.CNN_encoder[2](x_C)

        for CNN_Block in self.CNN_encoder[3:]:
            x_C=CNN_Block(x_C)
            CNN_features.append(x_C)

        #(2)Transformer_encoder
        nums=0
        Transformer_features=[]
        x_p=self.patch_embed(x)
        for Transformer_Block in self.Transformer_encoder:
            x_p=Transformer_Block(x_p,x)
            Transformer_features.append(self.Trans_modules[nums](x_p.reshape(-1,768,14,14)))
            nums=nums+1

        #TGAD_FM
        F_out=[]
        for i in range(4):
            F_out.append(self.Fusion_modules[i](Transformer_features[i],CNN_features[i]))

        #Decoder
        out=self.decoders[0](torch.cat((Transformer_features[-1],
                                        CNN_features[-1],
                                        F_out[-1]),dim=1))
        out=self.decoders[1](out)

        nums_2=-2
        for i in range(2,8,2):
            out=self.decoders[i](torch.cat((Transformer_features[nums_2],
                                           CNN_features[nums_2],
                                           F_out[nums_2],
                                           out),dim=1))
            nums_2=nums_2-1
            out=self.decoders[i+1](out)

        out=self.final_conv(out)

        return  out





# def test():
#     x = torch.Tensor(1, 3, 224, 224)
#     model=VET_FF_Net()
#     preds=model(x)
#     print(preds.shape)
#
#
# test()


