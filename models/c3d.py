import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial




class C3D(nn.Module):

    def __init__(self, num_classes=24):

        super(C3D, self).__init__()

        self.nClasses = num_classes

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4a = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4b = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5a = nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5b = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 512)
        self.bn6 = nn.BatchNorm3d(512)

        self.fc7 = nn.Linear(512*4, 1024)
        self.bn7 = nn.BatchNorm3d(1024)
        self.fc8 = nn.Linear(1024, self.nClasses)

        self.dropout = nn.Dropout(p=0.3)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_once(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))
        x = self.pool4(x)

        x = self.relu(self.bn5a(self.conv5a(x)))
        x = self.relu(self.bn5b(self.conv5b(x)))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.bn6(self.fc6(x)))
        #x = self.dropout(x)

        # batch x 512 for one clip
        return x

    def forward_pairwise(self, x):
        x = self.relu(self.bn7(self.fc7(x)))
        #x = self.dropout(x)

        return x

    def forward(self, inputs):
        #outputs = torch.FloatTensor(inputs.size(1), 24)

        b1 = self.forward_once(inputs[:,0,:,:,:,:])
        b2 = self.forward_once(inputs[:,1,:,:,:,:])
        b3 = self.forward_once(inputs[:,2,:,:,:,:])
        b4 = self.forward_once(inputs[:,3,:,:,:,:])

        # p12 = self.forward_pairwise(torch.cat([b1, b2], 1))
        # p13 = self.forward_pairwise(torch.cat([b1, b3], 1))
        # p14 = self.forward_pairwise(torch.cat([b1, b4], 1))
        # p23 = self.forward_pairwise(torch.cat([b2, b3], 1))
        # p24 = self.forward_pairwise(torch.cat([b2, b4], 1))
        # p34 = self.forward_pairwise(torch.cat([b3, b4], 1))
        #x = torch.cat([p12, p13, p14, p23, p24, p34], 1)

        x = torch.cat([b1, b2, b3, b4], 1)

        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)

        return x
