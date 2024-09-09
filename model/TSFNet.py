#from . import common
if __name__!="__main__":
    from . import MST
else:
    import MST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops


def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                    padding=(kernel_size // 2), bias=bias, groups=groups)


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size  = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride  = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        offset    = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x         = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x
    
class MSTFusionBlock(nn.Module):
    def __init__(self, dim_imgfeat, dim_dctfeat, kernel_size=3):
        super(MSTFusionBlock, self).__init__()
        conv = default_conv
        self.conv_img    = nn.Sequential(conv(dim_imgfeat, dim_imgfeat, kernel_size=kernel_size),
                                      nn.ReLU(True),
                                      conv(dim_imgfeat, dim_imgfeat, kernel_size=kernel_size))
        
        self.conv_dct    = nn.Sequential(conv(dim_dctfeat, dim_dctfeat, kernel_size=kernel_size),
                                      nn.ReLU(True),
                                      conv(dim_dctfeat, dim_dctfeat, kernel_size=kernel_size))
        
        self.stage_tconv = nn.ConvTranspose2d(dim_dctfeat, dim_dctfeat, kernel_size=kernel_size, stride=2, padding=(kernel_size//2))
        self.msab        = MST.MSAB(dim_in=(dim_imgfeat+dim_dctfeat), dim_head=(dim_imgfeat+dim_dctfeat), dim_out=dim_imgfeat, heads=2, num_blocks=1)
    
    def forward(self, in_pix, in_dct):    
        out_pix = self.conv_img(in_pix)
        out_dct = self.conv_dct(in_dct)
        out_pix = self.msab(out_pix, self.stage_tconv(out_dct, output_size=in_pix.shape[2:]))
        return out_pix+in_pix, out_dct+in_dct
        

class TSFNet(nn.Module):
    def __init__(self, args):
        super(TSFNet, self).__init__()

        in_channel1  = args.downsample_1 * args.n_colors#--> (4*2)    #12 --> (4x3)
        in_channel2  = args.downsample_2 * args.n_colors #48 --> (16x3)
        out_channel  = args.downsample_1 * args.o_colors
        
        dim_x1       = args.n_feats_1
        dim_x2       = args.n_feats_2
        
        kernel_size  = 3
        n_basicblock = args.n_resblocks
        
        # define head module for pixel input
        conv         = default_conv if args.deformable==False else DeformableConv2d
        self.head1   = nn.Sequential(conv(in_channels=in_channel1, out_channels=dim_x1//2, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                                nn.PReLU(dim_x1//2),
                                conv(in_channels=dim_x1//2, out_channels=dim_x1, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                                )
        self.head2   = nn.Sequential(conv(in_channels=in_channel2, out_channels=dim_x2//2, kernel_size=kernel_size, padding=(kernel_size//2), stride=1),
                               nn.PReLU(dim_x2//2),
                               conv(in_channels=dim_x2//2, out_channels=dim_x2, kernel_size=kernel_size, padding=(kernel_size//2), stride=1)
                               )
        self.body    = nn.ModuleList([ MSTFusionBlock(dim_x1, dim_x2, kernel_size) for _ in range(int(n_basicblock)) ])
        
        # define tail module
        self.tail    = default_conv(dim_x1, out_channel, kernel_size)
        
      
    def forward(self, x):
        x1 = F.pixel_unshuffle(x, 2)
        x2 = F.pixel_unshuffle(x, 4)
        
        x1 = self.head1(x1)
        x2 = self.head2(x2)
        
        for i, layer in enumerate(self.body):
            if i == 0:
                res_x1, res_x2 = layer(x1, x2)
            else:
                res_x1, res_x2 = layer(res_x1, res_x2)
            
        res_x1 += x1
        res_x1 = self.tail(res_x1)
        res_x1 = F.pixel_shuffle(res_x1, 2)
        res_x1 += x
        return res_x1


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__=="__main__":
    import os
    import sys
    home_dir = os.getenv("HOME")
    sys.path.append(os.path.join(home_dir, "PythonDir/TSFNet/"))
    from option import args
    model = TSFNet(args).cuda()
    print(model)
    img = torch.tensor(torch.ones(1,3,256,256)).cuda()
    print(img.shape)
    out = model(img)
    print(out.shape)
    print("Done...")