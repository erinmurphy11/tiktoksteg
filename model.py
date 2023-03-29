import torch.nn as nn
import torch

# custom weights initialization based on WGAN paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02) # mean, std
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def dw_conv(
        in_c: int, out_c: int, kernel_size: int, stride: int, padding: int
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_c,
                in_c,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_c,
            ),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_c,
                out_c,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

class HideNet(nn.Module):
    def get_channels(self, in_channels=6, out_channels=3, init_channels=64, max_channels=512, num_conv=6):
        # Initialize lists
        encoder_in = [in_channels]  # input channels for encoder
        encoder_out = []  # output channels for encoder
        decoder_in = []  # input channels for decoder
        decoder_out = [max_channels]  # output channels for decoder

        # Build encoder
        for i in range(num_conv):
            encoder_out.append(min(init_channels * 2 ** i, max_channels))
            encoder_in.append(encoder_out[-1])

        # Build decoder
        for i in range(num_conv):
            decoder_in.append(encoder_out[-1 - i] * 2)
            decoder_out.append(min(init_channels * 2 ** (num_conv - i - 2), max_channels))

        # Reverse the decoder lists to match the U-Net architecture
        decoder_in
        decoder_out

        # Adjust input and output channels to match given values
        encoder_in[0] = in_channels
        decoder_out[-1] = out_channels

        encoder_in = encoder_in[:-1]
        decoder_out = decoder_out[1:]

        return encoder_in, encoder_out, decoder_in, decoder_out
    
    def down_block(self, in_c: int, out_c: int, conv: nn.Module=nn.Conv2d, kernel_size: int=4, stride: int=2):
        return nn.Sequential(
            conv(in_c, out_c, kernel_size, stride, 1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2)
        )

    def up_block(
        self, in_c: int, out_c: int, conv: nn.Module, act=nn.ReLU, mode: str = "nearest"
    ):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode),
            conv(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            act(),
        )

    def __init__(
        self,
        first_c: int = 64,
        n_depthwise: int = 2,
        upsampling_mode: str = "nearest",
        n_conv: int = 6,
        max_c: int = 512,
    ):
        super().__init__()
        
        self.down_in, self.down_out, self.up_in, self.up_out = self.get_channels(
            init_channels=first_c, 
            max_channels=max_c,
            num_conv=n_conv,
        )

        down_layers = []
        up_layers = []

        for i in range(len(self.down_in)):
            
            if i < n_depthwise:
                conv = nn.Conv2d
            else:
                conv = dw_conv 
                
            down_layers.append(
                self.down_block(self.down_in[i], self.down_out[i], conv)
            )
             
            
        for i in range(len(self.up_in) - 1):
            
            if i < n_depthwise:
                conv = dw_conv
            else:
                conv = nn.Conv2d
                
            up_layers.append(
                self.up_block(self.up_in[i], self.up_out[i], conv, mode=upsampling_mode)
            )
            
        up_layers.append(
            self.up_block(self.up_in[-1], self.up_out[-1], nn.Conv2d, act=nn.Tanh)
        )
        
        self.down_layers = nn.ModuleList(down_layers)
        self.bottleneck = self.down_block(self.down_out[-1], self.down_out[-1], kernel_size=3, stride=1)
        self.up_layers = nn.ModuleList(up_layers)
        
        

    def forward(self, x):
        down_out = [x]

        for i in range(len(self.down_in)):
            down_out.append(self.down_layers[i](down_out[-1]))

        
        up_out = self.bottleneck(down_out[-1])
        up_out += down_out[-1]

        for i in range(1, len(self.up_in)):
            up_out = self.up_layers[i - 1](torch.concat([up_out, down_out[-i]], dim=1))
            
        up_out = self.up_layers[-1](torch.concat([up_out, down_out[1]], dim=1))

        return up_out
    
class RevealNet(nn.Module):
    def conv_block(self, in_c: int, out_c: int, conv: nn.Module=nn.Conv2d, kernel_size: int=3, stride: int=1, padding: int=1):
        return nn.Sequential(
            conv(in_c, out_c, kernel_size, stride, padding), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2)
        )
    
    def __init__(self, nc=3, nhf=64, output_function=nn.Tanh):

        super(RevealNet, self).__init__()
        self.main = nn.Sequential(
            self.conv_block(nc, nhf),
            self.conv_block(nhf, nhf * 2, conv=dw_conv),
            self.conv_block(nhf * 2, nhf * 4, conv=dw_conv),
            self.conv_block(nhf * 4, nhf * 2, conv=dw_conv),
            self.conv_block(nhf * 2, nhf, conv=dw_conv),
            nn.Conv2d(nhf, nc, 3, 1, 1),
            output_function(),
        )

    def forward(self, input):
        output = self.main(input)
        return output