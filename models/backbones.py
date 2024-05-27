import torch
from torch import nn
from .attention import *
from .MMB import *

class FCN(nn.Module):
    def __init__(self, n_channels, n_classes, in_dim=800, out_channels=128, backbone=True):
        super(FCN, self).__init__()

        self.backbone = backbone

        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                                         nn.Dropout(0.35))
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        if in_dim == 800: 
            self.out_len = 102
        elif in_dim == 640:
            self.out_len = 82
        elif in_dim == 200:
            self.out_len = 27
        elif in_dim == 512:
            self.out_len = 66
        elif in_dim == 1600:
            self.out_len = 202
        elif in_dim == 480:
            self.out_len = 62
        elif n_channels == 3: 
            self.out_len = 21

        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels

        if backbone == False:
            self.logits = nn.Linear(self.out_len * out_channels, n_classes)

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        if self.backbone:
            return None, x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.logits(x_flat)
            return logits, x


class FCN_2(nn.Module):
    def __init__(self, n_channels, n_classes, in_dim=800, out_channels=128, backbone=True):
        super(FCN_2, self).__init__()

        self.backbone = backbone

        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.Conv1d(32, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),                                         
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                                         nn.Dropout(0.2))
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=6, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(),
                                         nn.Conv1d(64, 64, kernel_size=6, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(),                                         
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=4, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        if in_dim == 800: 
            self.out_len = 108
        elif in_dim == 640:
            self.out_len = 88
        elif in_dim == 200:
            self.out_len = 33
        elif in_dim == 512:
            self.out_len = 66
        elif in_dim == 1600:
            self.out_len = 208
        elif in_dim == 480:
            self.out_len = 68
        elif n_channels == 3: 
            self.out_len = 21

        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels

        if backbone == False:
            self.logits = nn.Linear(self.out_len * out_channels, n_classes)

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        if self.backbone:
            return None, x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.logits(x_flat)
            return logits, x


class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True, regress=False):
        super(DeepConvLSTM, self).__init__()

        self.backbone = backbone
        self.n_classes = n_classes
        self.reg = regress
        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.out_dim = LSTM_units

        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes)
        if regress == True:
            self.regressor = nn.Linear(LSTM_units, 1)

        self.activation = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
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

        if self.backbone:
            return None, x
        elif self.reg:
            out = self.regressor(x)
            return out, x
        else:
            out = self.classifier(x)
            return out, x
        

class cnn_lstm(nn.Module):
    def __init__(self, n_channels, n_classes, backbone=True, regress=True):
      super(cnn_lstm, self).__init__()

      self.backbone = backbone
      self.n_classes = n_classes
      self.reg = regress

      self.conv1 = conbr_block(n_channels, 32, kernel_size=32, stride=1, dilation=1)
      self.pool1 = nn.MaxPool1d(2)
      self.conv2 = conbr_block(32, 64, kernel_size=16, stride=1, dilation=1)
      self.pool2 = nn.MaxPool1d(2)
      self.dropout1 = nn.Dropout1d(0.2)
      self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
      self.fc1 = nn.Linear(128, 1)
      self.dropout2 = nn.Dropout1d(0.2)
      self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        #x = torch.transpose(x, 0, 1)
        x = torch.transpose(x, 1, 2)
        x, (hidden, cell) = self.lstm1(x)
        x = x[:,-1,:]
        #x = x[-1,:,:]
        out = self.fc1(x)
        return out, x

class DeepConvLSTM_2(nn.Module):
    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True):
        super(DeepConvLSTM_2, self).__init__()

        self.backbone = backbone

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv1_2 = conbr_block(1, conv_kernels, kernel_size-2, 1, 1)
        self.conv2_2 = conbr_block(conv_kernels, 2*conv_kernels, kernel_size-2, 1, 1)
        self.conv3_2 = conbr_block(2*conv_kernels, 2*conv_kernels, kernel_size-2, 1, 1)
        self.conv4_2 = conbr_block(2*conv_kernels, 2*conv_kernels, kernel_size-2, 1, 1)

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * 2 *conv_kernels, LSTM_units, num_layers=2)

        self.out_dim = LSTM_units

        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes)

        self.activation = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(0, 2, 1)
        x = self.conv1_2(x)
        x = self.conv2_2(x)
        x = self.conv3_2(x)
        x = self.conv4_2(x)

        x = x.unsqueeze(3)
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x



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

class UNET_1D_simp_ssl(nn.Module):
    def __init__(self, input_dim, output_dim, layer_n, kernel_size, depth, args, backbone=True, n_class=1):
        super(UNET_1D_simp_ssl, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        self.output_dim = output_dim
        self.backbone = backbone
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
        self.fc = nn.Linear(output_dim, output_dim)
        self.out_act = nn.ReLU()
        self.out_dim = output_dim

        if backbone == False:
            self.classifier = nn.Linear(output_dim, n_class) 

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        return nn.Sequential(*block)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, channel, length)
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
        out1 = torch.tanh(out.squeeze())
        if self.backbone:
            return None, out1
        else:
            out = self.classifier(out1)
            return out, x




class LSTM(nn.Module):
    def __init__(self, n_channels, n_classes, LSTM_units=128, backbone=True):
        super(LSTM, self).__init__()

        self.backbone = backbone
        self.lstm = nn.LSTM(n_channels, LSTM_units, num_layers=2)
        self.out_dim = LSTM_units

        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(1, 0, 2)
        x, (h, c) = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x

class AE(nn.Module):
    def __init__(self, n_channels, len_sw, n_classes, outdim=128, backbone=True):
        super(AE, self).__init__()

        self.backbone = backbone
        self.len_sw = len_sw

        self.e1 = nn.Linear(n_channels, 8)
        self.e2 = nn.Linear(8 * len_sw, 2 * len_sw)
        self.e3 = nn.Linear(2 * len_sw, outdim)

        self.d1 = nn.Linear(outdim, 2 * len_sw)
        self.d2 = nn.Linear(2 * len_sw, 8 * len_sw)
        self.d3 = nn.Linear(8, n_channels)

        self.out_dim = outdim

        if backbone == False:
            self.classifier = nn.Linear(outdim, n_classes)

    def forward(self, x):
        x_e1 = self.e1(x)
        x_e1 = x_e1.reshape(x_e1.shape[0], -1)
        x_e2 = self.e2(x_e1)
        x_encoded = self.e3(x_e2)

        x_d1 = self.d1(x_encoded)
        x_d2 = self.d2(x_d1)
        x_d2 = x_d2.reshape(x_d2.shape[0], self.len_sw, 8)
        x_decoded = self.d3(x_d2)

        if self.backbone:
            return x_decoded, x_encoded
        else:
            out = self.classifier(x_encoded)
            return out, x_decoded

class CNN_AE(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE, self).__init__()

        self.backbone = backbone
        self.n_channels = n_channels

        self.e_conv1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU())
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.dropout = nn.Dropout(0.35)

        self.e_conv2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU())
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)

        self.e_conv3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU())
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)

        self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
        self.d_conv1 = nn.Sequential(nn.ConvTranspose1d(out_channels, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU())

        if n_channels == 9: # ucihar
            self.lin1 = nn.Linear(33, 34)
        elif n_channels == 6: # hhar
            self.lin1 = nn.Identity()
        elif n_channels == 3: # shar
            self.lin1 = nn.Linear(39, 40)

        self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
        self.d_conv2 = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(32),
                                     nn.ReLU())

        self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
        self.d_conv3 = nn.Sequential(nn.ConvTranspose1d(32, n_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(n_channels),
                                     nn.ReLU())

        if n_channels == 9: # ucihar
            self.lin2 = nn.Linear(127, 128)
            self.out_dim = 18 * out_channels
        elif n_channels == 6:  # hhar
            self.lin2 = nn.Linear(99, 100)
            self.out_dim = 15 * out_channels
        elif n_channels == 3: # shar
            self.out_dim = 21 * out_channels

        if backbone == False:
            self.classifier = nn.Linear(self.out_dim, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, indice1 = self.pool1(self.e_conv1(x))
        x = self.dropout(x)
        x, indice2 = self.pool2(self.e_conv2(x))
        x_encoded, indice3 = self.pool3(self.e_conv3(x))
        x = self.d_conv1(self.unpool1(x_encoded, indice3))
        x = self.lin1(x)
        x = self.d_conv2(self.unpool2(x, indice2))
        x = self.d_conv3(self.unpool1(x, indice1))
        if self.n_channels == 9: # ucihar
            x_decoded = self.lin2(x)
        elif self.n_channels == 6 : # hhar
            x_decoded =self.lin2(x)
        elif self.n_channels == 3: # shar
            x_decoded = x
        x_decoded = x_decoded.permute(0, 2, 1)
        x_encoded = x_encoded.reshape(x_encoded.shape[0], -1)

        if self.backbone:
            return x_decoded, x_encoded
        else:
            out = self.classifier(x_encoded)
            return out, x_decoded

class Transformer(nn.Module):
    def __init__(self, n_channels, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True):
        super(Transformer, self).__init__()

        self.backbone = backbone
        self.out_dim = dim
        self.transformer = Seq_Transformer(n_channel=n_channels, len_sw=len_sw, n_classes=n_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        if backbone == False:
            self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.transformer(x)
        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x

class Classifier(nn.Module):
    def __init__(self, bb_dim, n_classes):
        super(Classifier, self).__init__()

        self.classifier = nn.Linear(bb_dim, n_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.classifier(x)

        return out


class Projector(nn.Module):
    def __init__(self, model, bb_dim, prev_dim, dim):
        super(Projector, self).__init__()
        if model == 'SimCLR' or model == 'SimPer':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim))
        elif model == 'byol':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim, affine=False))
        elif model == 'NNCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim))
        elif model == 'TS-TCC':
            self.projector = nn.Sequential(nn.Linear(dim, bb_dim // 2),
                                           nn.BatchNorm1d(bb_dim // 2),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(bb_dim // 2, bb_dim // 4))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.projector(x)
        return x


class Predictor(nn.Module):
    def __init__(self, model, dim, pred_dim):
        super(Predictor, self).__init__()
        if model == 'SimCLR' or model == 'SimPer':
            pass
        elif model == 'byol':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        elif model == 'NNCLR':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.predictor(x)
        return x

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


from functools import wraps


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projector and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, DEVICE, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.DEVICE = DEVICE

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        children = [*self.net.children()]
        print('children[self.layer]:', children[self.layer])
        return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.reshape(output.shape[0], -1)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = Projector(model='byol', bb_dim=dim, prev_dim=self.projection_hidden_size, dim=self.projection_size)
        return projector.to(hidden)

    def get_representation(self, x):

        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        if self.net.__class__.__name__ in ['AE', 'CNN_AE']:
            x_decoded, representation = self.get_representation(x)
        else:
            _, representation = self.get_representation(x)

        if len(representation.shape) == 3:
            representation = representation.reshape(representation.shape[0], -1)

        projector = self._get_projector(representation)
        projection = projector(representation)
        if self.net.__class__.__name__ in ['AE', 'CNN_AE']:
            return projection, x_decoded, representation
        else:
            return projection, representation


class NNMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation
    This class implements a nearest neighbour memory bank as described in the
    NNCLR paper[0]. During the forward pass we return the nearest neighbour
    from the memory bank.
    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548
    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.
    """

    def __init__(self, size: int = 2 ** 16):
        super(NNMemoryBankModule, self).__init__(size)

    def forward(self,
                output: torch.Tensor,
                update: bool = False):
        """Returns nearest neighbour of output tensor from memory bank
        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it
        """

        output, bank = \
            super(NNMemoryBankModule, self).forward(output, update=update)
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        similarity_matrix = \
            torch.einsum("nd,md->nm", output_normed, bank_normed)
        index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        nearest_neighbours = \
            torch.index_select(bank, dim=0, index=index_nearest_neighbours)

        return nearest_neighbours
