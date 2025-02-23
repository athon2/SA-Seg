import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=1, bias=True)
        self.norm = nn.InstanceNorm3d(num_features=output_channels, affine=True)
        self.nonlin = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))
    
class StackedConvLayers(nn.Module):
    def __init__(self, input_features, output_features, num_convs, downsample=True) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        if downsample:
            stride = 2 
        else:
            stride = 1
        self.blocks = nn.Sequential(
            *([ConvBlock(input_features, output_features, stride=stride, padding=1)] + 
              [ConvBlock(output_features, output_features, stride=1, padding=1) for _ in range(num_convs - 1)])
        )

    def forward(self, x):
        return self.blocks(x)



class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class nnUNet3D(nn.Module):
    def __init__(self, input_channels, num_classes, inital_num_features, 
                 do_ds, num_convs=2, final_nonlin=lambda x: x) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([
            StackedConvLayers(input_channels, inital_num_features, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features, inital_num_features*2, num_convs=num_convs,),
            StackedConvLayers(inital_num_features*2, inital_num_features*4, num_convs=num_convs,),
            StackedConvLayers(inital_num_features*4, inital_num_features*8, num_convs=num_convs,),
        ]) 
        self.bottleneck = nn.Sequential(
            StackedConvLayers(inital_num_features*8, 320, 2)
            )
        self.upsample = nn.ModuleList([
            nn.ConvTranspose3d(320, inital_num_features*8, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*8, inital_num_features*4, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*4, inital_num_features*2, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*2, inital_num_features, kernel_size=2, stride=2, bias=False)
        ])

        self.decoders = nn.ModuleList([
            StackedConvLayers(inital_num_features*16, inital_num_features*8, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*8, inital_num_features*4, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*4, inital_num_features*2, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*2, inital_num_features, num_convs=num_convs, downsample=False)
        ])
        self.seg_outputs = []
        for ds in range(len(self.decoders)):
            self.seg_outputs.append(nn.Conv3d(self.decoders[ds].output_features, num_classes, 
                                              1, 1, 0, 1, 1, False))
            
        self.seg_outputs = nn.ModuleList(self.seg_outputs)          
        self.final_nonlin = final_nonlin
        self.do_ds = do_ds

        self.apply(InitWeights_He(1e-2))
        
    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.encoders)):
            x = self.encoders[d](x)
            skips.append(x)

        x = self.bottleneck(x)

        for u in range(len(self.decoders)):
            x = self.upsample[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.decoders[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.do_ds:
            return seg_outputs
        else:
            return seg_outputs[-1]

class nnUNet3D_MLPs(nn.Module):
    def __init__(self, input_channels, num_classes, inital_num_features, 
                 do_ds, num_convs=2, final_nonlin=lambda x: x, em=True, dims=[]) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([
            StackedConvLayers(input_channels, inital_num_features, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features, inital_num_features*2, num_convs=num_convs,),
            StackedConvLayers(inital_num_features*2, inital_num_features*4, num_convs=num_convs,),
            StackedConvLayers(inital_num_features*4, inital_num_features*8, num_convs=num_convs,),
        ]) 
        self.bottleneck = nn.Sequential(
            StackedConvLayers(inital_num_features*8, 320, num_convs=num_convs)
            )
        self.upsample = nn.ModuleList([
            nn.ConvTranspose3d(320, inital_num_features*8, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*8, inital_num_features*4, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*4, inital_num_features*2, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*2, inital_num_features, kernel_size=2, stride=2, bias=False)
        ])

        self.decoders = nn.ModuleList([
            StackedConvLayers(inital_num_features*16, inital_num_features*8, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*8, inital_num_features*4, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*4, inital_num_features*2, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*2, inital_num_features, num_convs=num_convs, downsample=False)
        ])
        self.posterior_y_x = nn.Sequential(
            *([ConvBlock(dims[i], dims[i+1], kernel_size=1, stride=1, padding=0) for i in range(len(dims)-1)]+
            [nn.Conv3d(dims[-1], num_classes, 1, 1, 0, 1, 1, False)])) 
        self.label_bias_s_xy = nn.Sequential(
            *([ConvBlock(dims[i], dims[i+1], kernel_size=1, stride=1, padding=0) for i in range(len(dims)-1)]+
            [nn.Conv3d(dims[-1], num_classes, 1, 1, 0, 1, 1, False)])) 
        self.em = em     
        self.final_nonlin = final_nonlin
        self.do_ds = do_ds

        self.apply(InitWeights_He(1e-2))
        
    def forward(self, x):
        skips = []
        for d in range(len(self.encoders)):
            x = self.encoders[d](x)
            skips.append(x)

        x = self.bottleneck(x)

        for u in range(len(self.decoders)):
            x = self.upsample[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.decoders[u](x)

        if self.em:
            return self.final_nonlin(self.posterior_y_x(x)), self.final_nonlin(self.label_bias_s_xy(x))
        else:
            return self.final_nonlin(self.posterior_y_x(x))


class nnUNet3D_branch(nn.Module):
    def __init__(self, input_channels, num_classes, inital_num_features, 
                 do_ds, num_convs=2, final_nonlin=lambda x: x) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([
            StackedConvLayers(input_channels, inital_num_features, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features, inital_num_features*2, num_convs=num_convs,),
            StackedConvLayers(inital_num_features*2, inital_num_features*4, num_convs=num_convs,),
            StackedConvLayers(inital_num_features*4, inital_num_features*8, num_convs=num_convs,),
        ]) 
        self.bottleneck = nn.Sequential(
            StackedConvLayers(inital_num_features*8, 320, 2)
            )
        self.upsample1 = nn.ModuleList([
            nn.ConvTranspose3d(320, inital_num_features*8, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*8, inital_num_features*4, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*4, inital_num_features*2, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*2, inital_num_features, kernel_size=2, stride=2, bias=False)
        ])
        self.upsample2 = nn.ModuleList([
            nn.ConvTranspose3d(320, inital_num_features*8, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*8, inital_num_features*4, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*4, inital_num_features*2, kernel_size=2, stride=2, bias=False),
            nn.ConvTranspose3d(inital_num_features*2, inital_num_features, kernel_size=2, stride=2, bias=False)
        ])

        self.decoders1 = nn.ModuleList([
            StackedConvLayers(inital_num_features*16, inital_num_features*8, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*8, inital_num_features*4, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*4, inital_num_features*2, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*2, inital_num_features, num_convs=num_convs, downsample=False)
        ])
        self.decoders2 = nn.ModuleList([
            StackedConvLayers(inital_num_features*16, inital_num_features*8, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*8, inital_num_features*4, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*4, inital_num_features*2, num_convs=num_convs, downsample=False),
            StackedConvLayers(inital_num_features*2, inital_num_features, num_convs=num_convs, downsample=False)
        ])
 
        self.seg_output1 = nn.Conv3d(inital_num_features, num_classes, 1, 1, 0, 1, 1, False)    
        self.seg_output2 = nn.Conv3d(inital_num_features, num_classes, 1, 1, 0, 1, 1, False)       
        self.final_nonlin = final_nonlin
        self.do_ds = do_ds
        self.inference = False
        self.apply(InitWeights_He(1e-2))
        self.feats_hook = False
        self.ex_hook = False

    def enc_forward(self, x):
        skips = []
        for d in range(len(self.encoders)):
            x = self.encoders[d](x)
            skips.append(x)

        x = self.bottleneck(x)

        return x, skips
    
    def dec1_forward(self, x, skips):
        for u in range(len(self.decoders1)):
            x = self.upsample1[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.decoders1[u](x)
        return x 

    def dec2_forward(self, x, skips):
        for u in range(len(self.decoders2)):
            x = self.upsample2[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.decoders2[u](x)

        return x 
    
    def forward(self, x):
        feats, skips = self.enc_forward(x)
        main_seg = self.dec1_forward(feats, skips)
        aux_seg = self.dec2_forward(feats, skips)

        if not self.inference:
            return self.final_nonlin(self.seg_output1(main_seg)),self.final_nonlin(self.seg_output2(aux_seg))
        else:
           return self.final_nonlin(self.seg_output1(main_seg))
        
class nnUNet3D_DualNet(nn.Module):
    def __init__(self, input_channels, num_classes, inital_num_features, 
                 do_ds, num_convs=2, final_nonlin=lambda x: x, dropout_p=0) -> None:
        super().__init__()
        self.net1 = nnUNet3D(input_channels, num_classes, inital_num_features, do_ds, num_convs, final_nonlin, dropout_p)
        self.net2 = nnUNet3D(input_channels, num_classes, inital_num_features, do_ds, num_convs, final_nonlin, dropout_p)
        self.inference = False

    def forward(self,x):
        f_x_logits = self.net1(x)
        if not self.inference:
            e_x_logits = self.net2(x)
            return f_x_logits, e_x_logits
        else:
            return f_x_logits