import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=2, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        return out
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(InceptionBlock, self).__init__()

        self.conv1 = conbr_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.conv2 = conbr_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.conv3 = conbr_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Adjust the convolutional layer when input and output channels differ
        if in_channels != out_channels:
            self.conv_res = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        
        # Apply the residual connection if the input and output channels differ
        if residual.shape[1] != out.shape[1]:
            residual = self.conv_res(residual)
        
        out += residual
        out = self.relu(out)
        
        return out
    
class UNET_1D_simp(nn.Module):
    def __init__(self, input_dim, output_dim, layer_n, kernel_size, depth, args):
        super(UNET_1D_simp, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        self.output_dim = output_dim
        self.args = args

        self.AvgPool1D0 = nn.AvgPool1d(kernel_size=int(self.args.fs/4), stride=None) if not args.data_type == 'ppg' else nn.AvgPool1d(kernel_size=int(args.fs/5), stride=1)
        self.AvgPool1D1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.AvgPool1D2 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=2)
        self.AvgPoolOut = nn.AvgPool1d(kernel_size=6, stride=2, padding=2)

        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size, 1, 1)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n * 2), self.kernel_size, 2, 2)
        self.layer3 = self.down_layer(int(self.layer_n * 2) + int(self.input_dim), int(self.layer_n * 3),
                                      self.kernel_size, 2, 2)
        self.layer4 = self.down_layer(int(self.layer_n * 3) + int(self.input_dim), int(self.layer_n * 4),
                                      self.kernel_size, 2, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n * 7), int(self.layer_n * 3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n * 5), int(self.layer_n * 2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n * 3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.outcov = nn.Conv1d(self.layer_n, 1, kernel_size=self.kernel_size, stride=1, padding=2)
        self.outcov2 = nn.Conv1d(in_channels=128, out_channels=181, kernel_size=1)
        self.fc = nn.Linear(output_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, 181)
        self.out_act = nn.ReLU()

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        return nn.Sequential(*block)
    
    def forward(self, x): # x -> (batch_size, channels, time steps)
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        #############Encoder#####################

        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)

        x1 = torch.cat([out_1, pool_x1], 1)
        out_2 = self.layer3(x1)

        x2 = torch.cat([out_2, pool_x2], 1)
        x3 = self.layer4(x2)

        #############Decoder####################
        up = self.upsample(x3)
        up = torch.cat([up, out_2], 1)
        up = self.cbr_up1(up)

        up = self.upsample(up)
        up = torch.cat([up, out_1], 1)
        up = self.cbr_up2(up)

        up = self.upsample(up)
        up = torch.cat([up, out_0], 1)
        up = self.cbr_up3(up)

        out = self.outcov(up)
        #out1 = self.fc(torch.flatten(out,start_dim=1))
        out1 = torch.tanh(out.squeeze())
        #filtered_signal = torch.mul(x,out1.unsqueeze(1)).squeeze()
        #out2 = torch.nn.Softmax(dim=1)(self.fc2(torch.flatten(out,start_dim=1)))
        #mapped_freq = map_freq(out2, self.args.cuda)
        return out1, None
    
################## convnet ###################
class convnet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, linear_unit, args):
        super(convnet, self).__init__()

        self.lin_unit = linear_unit
        self.args = args

        self.AvgPool1D0 = nn.AvgPool1d(kernel_size=int(self.args.fs/4), stride=None) if not args.data_type == 'ppg' else nn.AvgPool1d(kernel_size=int(args.fs/5), stride=1)

        self.conv1 = conbr_block(in_channels, out_channels, kernel_size, stride, dilation=1)
        self.incept1 = InceptionBlock(out_channels, 12, kernel_size, stride, dilation=1)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=3)
        self.incept2 = InceptionBlock(36, 36, kernel_size, stride, dilation=1)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=3)
        self.incept3 = InceptionBlock(108, 108, kernel_size, stride, dilation=1)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=3)        
        self.conv_out = conbr_block(324, 324, 1, stride=1, dilation=1)
        self.fc1 = nn.Linear(3564, linear_unit)

    def forward(self, x):
        x = self.AvgPool1D0(x)
        x = self.conv1(x)
        x = self.incept1(x)
        x = self.pool1(x)
        x = self.incept2(x)
        x = self.pool2(x)
        x = self.incept3(x)
        x = self.pool3(x)      
        x = self.conv_out(x) 
        x = self.fc1(torch.flatten(x,start_dim=1))
        out = torch.tanh(x)
        return out.squeeze(), None

############################################### Analyze Layer ############################
class analyze_layer(nn.Module):
    def __init__(self, input_layer, out_layer, kernel_size, stride, linear_unit):
        super(analyze_layer, self).__init__()

        self.first_conv = conbr_block(1, out_layer, kernel_size, stride, dilation=1)
        self.second_conv = conbr_block(out_layer, out_layer*2, kernel_size, stride, dilation=1)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=3)
        self.fc1 = nn.Linear(linear_unit, input_layer)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.pool1(x)
        x = self.fc1(torch.flatten(x,start_dim=1))
        out = 0.01+F.sigmoid(x)
        return out.squeeze()
    
############################################### DCL Arch ############################

class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, data_type='ppg', conv_kernels=64, kernel_size=5, LSTM_units=128):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.out_dim = LSTM_units
        
        if data_type == 'ppg':
            self.fc1 = nn.Linear(128, 200)

        self.activation = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]
        x = self.fc1(x)
        x = torch.tanh(x.squeeze())
        return x, None
    
############## Setup models #################

def setup_model(args, DEVICE):
    if args.model == 'unet':
        return UNET_1D_simp(input_dim=1, output_dim=args.out_dim, layer_n=32, kernel_size=5, depth=1, args=args).cuda(DEVICE)
    elif args.model == 'resunet':
        return resunet(args=args).cuda(DEVICE)
    elif args.model == 'convnet':
        return convnet(in_channels=1, out_channels=8, kernel_size=5, stride=1, linear_unit=args.out_dim, args=args).cuda(DEVICE)
    elif args.model == 'dcl':
        return DeepConvLSTM(n_channels=1, data_type=args.data_type, conv_kernels=64, kernel_size=5, LSTM_units=128).cuda(DEVICE)
    elif args.model == 'resnet1d':
        args.model == ResNet1D(in_channels=1, base_filters=32, kernel_size=5, stride=1, groups=1, n_block=3, n_classes=args.out_dim, downsample_gap=2, increasefilter_gap=4, use_do=True).cuda(DEVICE)
    else:
        NotImplementedError


############## Setup the other model ################
class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm1d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x
      
class res_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        #conv layer
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        #Shortcut Connection (Identity Mapping)
        self.s = nn.Conv1d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip
      
class decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.r = res_block(in_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x
      
class resunet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        """ Encoder 1 """
        self.c11 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(64)
        self.c12 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv1d(1, 64, kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        self.r2 = res_block(64, 128, stride=2)
        self.r3 = res_block(128, 256, stride=2)

        """ Bridge """
        self.r4 = res_block(256, 512, stride=2)

        """ Decoder """
        self.d1 = decoder(512, 256)
        self.d2 = decoder(256, 128)
        self.d3 = decoder(128, 64)

        """ Output """
        self.output = nn.Conv1d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(200, 181)
    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        b = self.r4(skip3)

        """ Decoder """
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        """ output """
        output = self.output(d3)
        output1 = self.sigmoid(output)
        out2 = torch.nn.Softmax(dim=1)(self.fc2(torch.flatten(output,start_dim=1)))
        mapped_freq = map_freq(out2, self.args.cuda)
        return output1.squeeze(), mapped_freq
    
############ Normalize ###################
def normalize_tensors(tensor1, tensor2, tensor3, tensor4, tensor5):
    # Stack the tensors along the second dimension
    stacked_tensors = torch.stack((tensor1, tensor2, tensor3, tensor4, tensor5), dim=1)

    # Calculate the sum of each batch
    batch_sum = stacked_tensors.sum(dim=1, keepdim=True)

    # Normalize each batch by dividing by the sum
    normalized_tensors = stacked_tensors / batch_sum

    return normalized_tensors

def diff_with_last_element(tensor):
    last_element = tensor[:, :, -1].unsqueeze(2)  # Extract the last element and unsqueeze to match dimensions
    tensor = torch.cat((tensor, last_element), dim=2)  # Append the last element to the tensor
    diff_result = torch.diff(tensor, dim=2)  # Apply torch.diff along the third dimension
    return diff_result

############ Map value ###########
def map_freq_phase(freq, phase, DEVICE):
    max_bpm, min_bpm, fs = 210, 30, 100
    phase_values = torch.linspace(0, torch.tensor(np.pi / 2), int(10), dtype=torch.float32, requires_grad=True).cuda(DEVICE) 
    freq_values = (fs/60)*torch.linspace(min_bpm, torch.tensor(max_bpm), int(max_bpm-min_bpm+1), dtype=torch.float32, requires_grad=True).cuda(DEVICE) 
    ###
    freq.requires_grad_(True)
    weighted_freq = torch.sum(freq_values[None,:,None]*freq,1)
    weighted_phase = torch.sum(phase_values[None,:,None]*phase,1)
    return weighted_freq, weighted_phase

############ Map single value ###########
def map_freq(freq, DEVICE):
    max_bpm, min_bpm, fs = 210, 30, 100
    freq_values = (fs*60)/torch.linspace(min_bpm, torch.tensor(max_bpm), int(max_bpm-min_bpm+1), dtype=torch.float32, requires_grad=True).cuda(DEVICE) 
    ###
    weighted_freq = torch.zeros(freq.size(0), requires_grad=True)
    weighted_freq = torch.sum(freq_values[None,:]*freq,1)
    return weighted_freq

############ parameter count ###########
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

############## RES-NET1D ################
"""
resnet for 1-d signal data, pytorch version
 
Shenda Hong, Oct 2019
"""
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False, backbone=False, output_dim=200):
        super(ResNet1D, self).__init__()
        
        self.out_dim = output_dim
        self.backbone = backbone
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        self.dense2 = nn.Linear(out_channels, self.out_dim)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.transpose(-1,-2) # RESNET 1D takes channels first
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.backbone:
            out = self.dense2(out)
            return None, out
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out_class = self.dense(out)
        if self.verbose:
            print('dense', out_class.shape)
        # out = self.softmax(out)
        if self.verbose:
            print('softmax', out_class.shape)
        
        return out_class, out    
    


################################################################################
############## multi-rate model ################
class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction

        self.conv_even = lambda x: x[:, ::2, :]
        self.conv_odd = lambda x: x[:, 1::2, :]


    def forward(self, x):
        '''Returns the odd and even part'''
        # if not even, pad the input with last item
        if x.size(1) % 2 != 0:
            x = torch.cat((x, x[:,-1:, :]), dim=1)
        return (self.conv_even(x), self.conv_odd(x))


class LiftingScheme(nn.Module):
    def __init__(self, in_planes, modified=False, size=[], splitting=True, k_size=4, dropout=0, simple_lifting=False):
        super(LiftingScheme, self).__init__()
        self.modified = True

        # kernel_size = k_size
        kernel_size = k_size
        dilation = 1

        # pad = int(np.ceil(dilation * (kernel_size - 1) / 2)) + 1
        pad = kernel_size - 1 if kernel_size != 1 else 0
        # pad = k_size // 2 # 2 1 0 0

        self.splitting = splitting
        self.split = Splitting()

        # Dynamic build sequential network
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:
            modules_P = [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_U = [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
        else:
            size_hidden = 2

            modules_P = [
                nn.ReflectionPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                          kernel_size=kernel_size, dilation=dilation,stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #    nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(in_planes * size_hidden, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_U = [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                          kernel_size=kernel_size, dilation=dilation, stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
             #   nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(in_planes * size_hidden, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            if self.modified:
                modules_phi = [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                          kernel_size=kernel_size, dilation=dilation, stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                #nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(in_planes * size_hidden, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
                modules_psi = [
                nn.ReplicationPad1d(pad),
                nn.Conv1d(in_planes * prev_size, in_planes * size_hidden,
                          kernel_size=kernel_size, dilation=dilation, stride=1),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
#                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(in_planes * size_hidden, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]

            self.phi = nn.Sequential(*modules_phi)
            self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)
#        self.phi = nn.Sequential(*modules_phi)
#        self.psi = nn.Sequential(*modules_psi)


    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if not self.modified:

            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            # x_odd = self.ptemp(x_odd)
            # x_odd =self.U(x_odd) #18 65
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c) #         Todo: +  -> * -> sigmod
#            d = x_odd - self.P(x_even)
#            c = x_even + self.U(d)

            # c = x_even + self.seNet_P(x_odd)
            # d = x_odd - self.seNet_P(c)
            return (c, d)
        else:

            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            # a = self.phi(x_even)

            d = x_odd.mul(torch.exp(self.phi(x_even))) - self.P(x_even)
            c = x_even.mul(torch.exp(self.psi(d))) + self.U(d)
            return (c, d)

class LiftingSchemeLevel(nn.Module):
    def __init__(self, in_planes, share_weights, modified=False, size=[2, 1], kernel_size=4, simple_lifting=False):
        super(LiftingSchemeLevel, self).__init__()
        self.level = LiftingScheme(
             in_planes=in_planes, modified=modified,
            size=size, k_size=kernel_size, simple_lifting=simple_lifting)


    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (L, H) = self.level(x)  

        return (L, H)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, disable_conv=True):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
#        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.disable_conv = disable_conv
        if not self.disable_conv:
            self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        if self.disable_conv:
            return self.relu(self.bn1(x)) # It was like that
        else:
            return self.conv1(x) # Then to this
            return self.conv1((self.bn1(x))) # First, changed to this

class LevelTWaveNet(nn.Module):
    def __init__(self, in_planes, out_planes, lifting_size, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx):
        super(LevelTWaveNet, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        if self.regu_approx + self.regu_details > 0.0:

            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingSchemeLevel(in_planes, share_weights,
                                       size=lifting_size, kernel_size=kernel_size,
                                       simple_lifting=simple_lifting)
        self.share_weights = share_weights
        if no_bottleneck:
            self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv=True)
        else:
            # out_planes = in_planes if in_planes > 1 else out_planes
            self.bootleneck = BottleneckBlock(in_planes, out_planes, disable_conv=False)

    def forward(self, x):
        (L, H) = self.wavelet(x) 
        approx = L
        details = H
        r = None
        if(self.regu_approx + self.regu_details != 0.0):  #regu_details=0.01, regu_approx=0.01

            if self.regu_details:
                rd = self.regu_details * \
                     H.abs().mean()

            # Constrain on the approximation
            if self.regu_approx:
                rc = self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)

            if self.regu_approx == 0.0:
                # Only the details
                r = rd
            elif self.regu_details == 0.0:
                # Only the approximation
                r = rc
            else:
                # Both
                r = rd + rc
        if self.bootleneck:
            return self.bootleneck(approx).permute(0, 2, 1), r, self.bootleneck(details).permute(0, 2, 1)
        else:
            return approx.permute(0, 2, 1), r, details

class multi_rate(nn.Module):
    def __init__(self, num_classes, first_conv=9, backbone=False, extend_channel=128,
                 number_levels=7, conv_kernels=16, lifting_size=[2, 1], kernel_size=4, no_bootleneck=False,
                 classifier="mode1", share_weights=False, simple_lifting=False,
                  regu_details=0.01, regu_approx=0.01):
        super(multi_rate, self).__init__()
        self.backbone = backbone
        self.share_weights = share_weights
        self.nb_channels_in = first_conv
        # First convolution
        if first_conv != 1 and first_conv != 3 and first_conv != 9 and first_conv != 22 :
            self.first_conv = True
            self.conv1 = nn.Sequential(
                nn.Conv1d(first_conv, extend_channel,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(extend_channel),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(extend_channel, extend_channel,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(extend_channel),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(0.5),
            )
            in_planes = extend_channel
            out_planes = extend_channel * (number_levels + 1)
        else:
            self.first_conv = False
            in_planes = first_conv
            out_planes = first_conv * (number_levels + 1)


        self.levels = nn.ModuleList()

        for i in range(number_levels):
            # bootleneck = True
            # if no_bootleneck and i == number_levels - 1:
            #     bootleneck = False
            out_planes = conv_kernels
            if i == 0:
                in_planes = 1
                self.levels.add_module(
                    'level_' + str(i),
                    LevelTWaveNet(in_planes, out_planes,
                                lifting_size, kernel_size, no_bootleneck,
                                share_weights, simple_lifting, regu_details, regu_approx)
                )
            else:
                self.levels.add_module(
                    'level_' + str(i),
                    LevelTWaveNet(in_planes, out_planes,
                                lifting_size, kernel_size, no_bootleneck,
                                share_weights, simple_lifting, regu_details, regu_approx)
                )
            in_planes = conv_kernels

        if no_bootleneck:
            in_planes *= 1

        self.num_planes = out_planes

        if classifier == "mode1":
            self.fc = nn.Linear(128, num_classes)
        elif classifier == "mode2":

            self.fc = nn.Sequential(
                nn.Linear(in_planes*(number_levels + 1), 1024),  # Todo:  extend channels
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(1024, num_classes)
            )
        else:
            raise "Unknown classifier"

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.atten = MultiHeadAttention(n_head=2, d_model=conv_kernels, d_k=32, d_v=32, share_weights=share_weights, dropout=0)
        self.count_levels = 0
    def forward(self, x):
        if self.first_conv:
            x = x.permute(0, 2, 1)
            x = self.conv1(x)
            x = x.permute(0, 2, 1)
        rs = []  # List of constrains on details and mean
        det = []  # List of averaged pooled details

        input = [x, ]
        for l in self.levels:
            low, r, details = l(input[0])

            input.append(low)
            input.append(details)
            del input[0]

            rs += [r]
            self.count_levels = self.count_levels + 1
        # import pdb;pdb.set_trace();
        for aprox in input:
            aprox = aprox.permute(0, 2, 1)
            aprox = self.avgpool(aprox)
            det += [aprox]

        self.count_levels = 0
        # We add them inside the all GAP detail coefficients

        x = torch.cat(det, 2) #[b, 77, 8]
        x = x.permute(0, 2, 1)
        x = x.reshape(x.size(0), -1)
        # q, att = self.atten(x, x, x, mask=None)
        # x = q
        # b, c, l = x.size()
        # x = x.view(-1, c * l)
        #
        # det += [aprox]
        # x = torch.cat(det, 2)
        # b, c, l = x.size()
        # x = x.view(-1, c * l)
        return self.fc(x), rs
    


############## Multi-Head Attention ################
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, share_weights=False, dropout=0):
        super().__init__()
        self.share_weight = share_weights
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        temp = True
        if share_weights:
            print('Attention block share weight!')
            self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        else:
            # print('Attention block NOT share weight!')
            self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        if temp:
            # print('Attention temporature < 1')
            self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        else:
            print('Attention temporature = 1')
            self.attention = ScaledDotProductAttention(temperature=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        if  self.share_weight:
            k = self.w_qs(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_qs(v).view(sz_b, len_v, n_head, d_v)
        else: 
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn       