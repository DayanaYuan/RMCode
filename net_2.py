import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed1d.pth',
}


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class channel_Weight(nn.Module):
    def __init__(self, inchannel, ratio=16, pool_type=["avg", "max"]):
        super(channel_Weight, self).__init__()
        self.fc = nn.Sequential(Flatten(),
                                nn.Linear(inchannel, inchannel // ratio, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(inchannel // ratio, inchannel, bias=False))
        self.pool = pool_type

    def forward(self, x):
        sum = None
        for i in self.pool:
            if i == "avg":
                avg = F.avg_pool1d(x, x.size(2), stride=x.size(2))
                # C*H*W---->1*H*W
                feature = self.fc(avg)
            elif i == "max":
                max = F.max_pool1d(x, x.size(2), stride=x.size(2))
                feature = self.fc(max)
            if sum is None:
                sum = feature
            else:
                sum += feature

        weight = F.sigmoid(sum).unsqueeze(2).expand_as(x)
        return weight * x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Spatial_weight(nn.Module):
    def __init__(self):
        super(Spatial_weight, self).__init__()
        self.pool = ChannelPool()
        self.conv = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                                  nn.BatchNorm1d(1, eps=1e-5, momentum=0.01, affine=True))

    def forward(self, x):
        spatial = self.pool(x)
        weight = self.conv(spatial)
        weight = F.sigmoid(weight)
        return x * weight


class CBAM(nn.Module):
    def __init__(self, inchannel, ratio=16, pool_type=["avg", "max"]):
        super(CBAM, self).__init__()
        self.channnel_Weight = channel_Weight(inchannel, ratio=ratio, pool_type=pool_type)
        self.Spatial_weight = Spatial_weight()

    def forward(self, x):
        x = self.channnel_Weight(x)
        x = self.Spatial_weight(x)
        return x


dp_rate = 0


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=33, stride=stride,
                     padding=16, bias=False)


class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn0 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes * 2)

        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            # residual = torch.cat((residual,residual),1)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn0 = nn.BatchNorm1d(inplanes)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=33, bias=False, padding=16)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=65, stride=stride,
                               padding=32, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False, padding=0)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dp_rate)
        self.layercbam = CBAM(planes * 4)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.relu(out)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        # out = self.bn3(out)
        out = self.layercbam(out)
        if self.downsample is not None:
            residual = self.downsample(x)
            # residual = torch.cat((residual, residual), 1)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, classification, num_class=3):
        self.inplanes = 12
        self.classification = classification
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(17, self.inplanes, kernel_size=33, stride=1, padding=16,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(self.inplanes, self.inplanes, kernel_size=33, stride=2, padding=16,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(self.inplanes)
        self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(self.inplanes, self.inplanes, kernel_size=33, stride=1, padding=16,
                               bias=False)
        self.dropout = nn.Dropout(dp_rate)
        self.layer1 = self._make_layer(block, 12, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 24, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 48, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 96, layers[3], stride=2)
        # self.layer5 = self._make_layer(block, self.inplanes, layers[4], stride=2)
        # for resnet50
        self.bn_final = nn.BatchNorm1d(96 * 2 * 2)
        self.avgpool = nn.AdaptiveAvgPool1d(2 * 2)
        self.fc1 = nn.Linear(96 * 4 * 4, 384)
        # self.bn_final = nn.BatchNorm1d(96*2)
        # self.avgpool = nn.AdaptiveAvgPool1d(2)
        # self.fc1 = nn.Linear(96*4, 384)
        self.bn3 = nn.BatchNorm1d(384)
        self.fc2 = nn.Linear(384, 192)
        self.bn4 = nn.BatchNorm1d(192)
        self.fc3 = nn.Linear(192, 1)
        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out)
        residual = self.downsample(x)
        out += residual
        x = self.relu(out)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        x = self.bn_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.classification:
            x = self.fc1(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            # x = self.softmax(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.linear = nn.Linear(64, 1)
    def forward(self, x, prev_states=None):
        xr_out2 = self.linear(x)
        return xr_out2

class Combine(nn.Module):
    def __init__(self, module, input_size, device, batch_first=True, **kwargs):
        super(Combine, self).__init__(**kwargs)
        self.cnn = module
        self.device = device
        self.num_layers = 1
        self.num_direction = 1
        self.hidden_size = 64
        # self.embedding = nn.Linear(input_size,192)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=batch_first)
        # self.rnn = nn.LSTM(
        #     input_size=192,
        #     hidden_size=self.hidden_size,
        #     num_layers=self.num_layers,
        #     batch_first=batch_first)
        self.linear = nn.Linear(64, 2)
        # 将隐藏层初始状态h0，c0作为模型参数的一部分
        # state_size = (self.num_layers * self.num_direction, 1, self.hidden_size)
        # self.init_h = nn.Parameter(torch.zeros(state_size))
        # self.init_c = nn.Parameter(torch.zeros(state_size))

    def forward(self, x, prev_states=None):
        batch_size, timesteps, H, W = x.size()

        if prev_states is None:
            state_size = (self.num_layers * self.num_direction, batch_size, self.hidden_size)
            init_h = torch.zeros(state_size).to(self.device)
            # init_c = torch.zeros(state_size).to(self.device)
            # init_h = self.init_h.expand(*state_size).contiguous()
            # init_c = self.init_c.expand(*state_size).contiguous()
            prev_states = init_h

        # 将输入数据的batch_size和timestemps合并为一维，先进性cnn，然后再拆分为batch_size和timestemps维度进行lstm
        xc_in = x.view(batch_size * timesteps, H, W)
        xc_out = self.cnn(xc_in)
        xr_in = xc_out.view(batch_size, timesteps, -1)
        # xr_in = torch.as_tensor(xr_in,dtype=torch.long)
        # xr_in = self.embedding(xr_in)
        xr_out, states = self.rnn(xr_in, prev_states)
        xr_out2 = self.linear(xr_out)

        return torch.squeeze(xr_out), states


if __name__ == '__main__':
    x = torch.randint(0, 1, [8, 17, 1024])