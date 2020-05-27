import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
from torchvision import models
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init


class threeDmodel(BaseModel):
    def __init__(self,num_feature=32, num_classes=2, depth=7):
        super(threeDmodel,self).__init__()
        self.num_feature=num_feature
        
        self.conv_layer1 = self._make_conv_layer(1, self.num_feature)
        self.conv_layer2 = self._make_conv_layer(self.num_feature, self.num_feature*2)
        self.conv_layer3 = self._make_conv_layer(self.num_feature*2, self.num_feature*4)
        self.conv_layer4= nn.Sequential(
            nn.Conv3d(self.num_feature*4, self.num_feature*8, kernel_size=(1, 3, 3), padding=(0,1,1)),
            nn.BatchNorm3d(self.num_feature*8),
            nn.LeakyReLU())
        
        self.fc5 = nn.Linear(self.num_feature*8*1*4*4, self.num_feature*8)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(self.num_feature*8)
        self.drop=nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(self.num_feature*8, self.num_feature*4)
        self.relu1 = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(self.num_feature*4)
        self.drop1=nn.Dropout(p=0.5)     
        self.fc7 = nn.Linear(self.num_feature*4, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
                
        
    def forward(self,x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x=self.conv_layer4(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu1(x)
        x = self.batch1(x)
        x = self.drop1(x)
        x1=x
        x = self.fc7(x)

        return x
    
    def _make_conv_layer(self, in_c, out_c, mp_d=2):
        # note that kernals in Conv3d are (depth, width, height)
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((mp_d, 2, 2)),
        )
        return conv_layer


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
class deeperModel(BaseModel):
    def __init__(self,num_feature=32, num_classes=2):
        super(deeperModel,self).__init__()
        self.num_feature=num_feature
        
        self.layer = nn.Sequential(
            nn.Conv2d(1,self.num_feature,5,1,2),
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU(),
            nn.Conv2d(self.num_feature,self.num_feature*2,3,1,1),
            nn.BatchNorm2d(self.num_feature*2),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*2,self.num_feature*4,3,1,1),
            nn.BatchNorm2d(self.num_feature*4),
            nn.ReLU(),
            nn.Conv2d(self.num_feature*4,self.num_feature*8,3,1,1),
            nn.BatchNorm2d(self.num_feature*8),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*8,self.num_feature*16,3,1,1),
            nn.BatchNorm2d(self.num_feature*16),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(self.num_feature*16,self.num_feature*16,3,1,1),
            nn.BatchNorm2d(self.num_feature*16),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature*16*3*3,1000),
            nn.ReLU(),
            nn.Linear(1000,num_classes)
        )       
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
                
            elif isinstance(m, nn.Linear):

                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
        
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(x.size()[0],-1)
        out = self.fc_layer(out)

        return out
    
class myCustomModel(BaseModel):
    def __init__(self, num_classes = 2):
        super(myCustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        return F.log_softmax(x, dim=1)
    
class pretrainedModel(BaseModel):
    def __init__(self, num_classes = 2):
        super(pretrainedModel, self).__init__()
        resnet = models.resnet152(pretrained=True)
        resnet.train()
        for param in resnet.parameters():
            param.requires_grad = False

        # new final layer with 10 classes
        num_ftrs = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(num_ftrs, num_classes) #input is 224,224,3
            
        use_gpu = True
        if use_gpu:
            resnet = resnet.cuda()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        self.resnet = resnet
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)

    def forward(self, x):
        #print(x.size())
        x = self.resnet(x)
        return x
    
class templateModel(BaseModel):
    def __init__(self):
        super(templateModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class groundTruthModel(BaseModel):
    def __init__(self, num_classes=2):
        super(groundTruthModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                # Xavier Initialization
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)
                
            elif isinstance(m, nn.Linear):

                # Xavier Initialization
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print(x.shape)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
    
class heatmapModel(BaseModel):
    def __init__(self,num_feature=32, num_classes=2):
        super(heatmapModel,self).__init__()
        self.num_feature=num_feature
        self.layer = nn.Sequential(
            nn.Conv2d(1,self.num_feature,3,1,1),
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU(),
            nn.Conv2d(self.num_feature,self.num_feature*2,3,1,1),
            nn.BatchNorm2d(self.num_feature*2),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*2,self.num_feature*4,3,1,1),
            nn.BatchNorm2d(self.num_feature*4),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*4,self.num_feature*8,3,1,1),
            nn.BatchNorm2d(self.num_feature*8),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature*8*7*7,self.num_feature*8),
            nn.ReLU(),
            nn.Linear(self.num_feature*8, num_classes)
        )    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)  
            elif isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
                
        
    def forward(self,x):
        print(x.shape)
        out = self.layer(x)
        print(out.shape)
        out = out.view(x.size()[0],-1)
        print(out.shape)
        out = self.fc_layer(out)
        return out

class heatmapModel64(BaseModel):
    def __init__(self,num_feature=32, num_classes=2):
        super(heatmapModel64,self).__init__()
        self.num_feature=num_feature
        self.layer = nn.Sequential(
            nn.Conv2d(1,self.num_feature,3,1,1),
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU(),
            nn.Conv2d(self.num_feature,self.num_feature*2,3,1,1),
            nn.BatchNorm2d(self.num_feature*2),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*2,self.num_feature*4,3,1,1),
            nn.BatchNorm2d(self.num_feature*4),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*4,self.num_feature*8,3,1,1),
            nn.BatchNorm2d(self.num_feature*8),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            
            nn.Conv2d(self.num_feature*8,self.num_feature*16,3,1,1),
            nn.BatchNorm2d(self.num_feature*16),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature*16*8*8,self.num_feature*16),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_feature*16,num_classes)
        )    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
                
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(x.size()[0],-1)
        out = self.fc_layer(out)
        return out
    

class threeDmodel_simple(BaseModel):
    def __init__(self,num_feature=32, num_classes=2, depth=7):
        super(threeDmodel_simple,self).__init__()
        self.num_feature=num_feature
        
        self.conv_layer1 = self._make_conv_layer(1, self.num_feature)
        self.conv_layer2 = self._make_conv_layer(self.num_feature, self.num_feature*2)
        self.conv_layer4=nn.Conv3d(self.num_feature*2, self.num_feature*4, kernel_size=(1, 3, 3), padding=(0,1,1))
        
        self.fc5 = nn.Linear(self.num_feature*4*1*8*8, self.num_feature*4)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(self.num_feature*4)
        self.drop=nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(self.num_feature*4, self.num_feature*2)
        self.relu1 = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(self.num_feature*2)
        self.drop1=nn.Dropout(p=0.5)     
        self.fc7 = nn.Linear(self.num_feature*2, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
                
        
    def forward(self,x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x=self.conv_layer4(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu1(x)
        x = self.batch1(x)
        x = self.drop1(x)
        x1=x
        x = self.fc7(x)

        return x
    
    def _make_conv_layer(self, in_c, out_c, mp_d=2):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 5, 5), padding=(1,2,2)),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((mp_d, 2, 2)),
        )
        return conv_layer

class Inception3D(BaseModel):
    def __init__(self, num_feature = 16, num_classes = 7):
        super(Inception3D, self).__init__()
        num_features = int(num_feature)
        self.conv1 = self._make_conv_layer(1, num_features)
        self.conv2 = self._make_conv_layer(num_features, num_features*2, mp_d = 1)
        
        self.inception_b1_1 = BasicConv3d(num_features*2, int(num_features / 2), kernel_size = 1)
        self.inception_b1_2 = BasicConv3d(int(num_features / 2), num_features, kernel_size = (1,3,3), padding = (0,1,1))
        self.inception_b1_3 = BasicConv3d(num_features, num_features, kernel_size = (1,3,3), padding = (0,1,1))
        
        self.inception_b2_1 = BasicConv3d(num_features*2, int(num_features / 2), kernel_size = (1,1,1))
        self.inception_b2_2 = BasicConv3d(int(num_features / 2), num_features, kernel_size = (1,3,3), padding = (0,1,1))
        
        self.mp = nn.MaxPool3d((1,3,3), stride =1, padding = (0,1,1))
        self.inception_b3_1 = BasicConv3d(num_features*2, num_features, kernel_size = (1,1,1))
        
        self.inception_b4_1 = BasicConv3d(num_features*2, num_features, kernel_size = (1,1,1))
        self.conv3 = self._make_conv_layer(num_features*4, num_features*8, mp_d = 2)
        
        self.fc1 = nn.Linear(num_features*8*1*8*8, num_features*8)
        self.relu1 = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(num_features*8)
        self.drop1=nn.Dropout(p=0.5)     
        self.fc2 = nn.Linear(num_features*8, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
        
    def forward(self, x):
        #print(x.size()) #1x7x32x32
        x = self.conv1(x)
        x = self.conv2(x)
        #print(x.size()) #32x3x16x16
        b1 = self.inception_b1_1(x)
        b1 = self.inception_b1_2(b1)
        b1 = self.inception_b1_3(b1)
        #print(b1.size()) #16x3x16x16
        
        b2 = self.inception_b2_1(x)
        b2 = self.inception_b2_2(b2)
        #print(b2.size()) #16x3x16x16
        
        b3 = self.mp(x)
        b3 = self.inception_b3_1(b3)
        #print(b3.size()) #16x3x16x16
        
        b4 = self.inception_b4_1(x)
        #print(b4.size()) #16x3x16x16
        
        outs = torch.cat([b1,b2,b3,b4], 1)
        #print(outs.size())
        outs = self.conv3(outs)
        outs = outs.view(outs.size(0), -1)
        #print(outs.size()) #128x1x8x8
        
        outs = self.fc1(outs)
        outs = self.relu1(outs)
        outs = self.batch1(outs)
        outs = self.drop1(outs)
        outs = self.fc2(outs)
        
        return outs
    
    def _make_conv_layer(self, in_c, out_c, mp_d=2):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 5, 5), padding=(1,2,2)),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((mp_d, mp_d, mp_d)),
        )
        return conv_layer
        
class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs) #kernal_size = (1,3,3) padding = (1,2,2) etc.
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)
        self.relu = nn.LeakyReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
        
class model1D(BaseModel):
    def __init__(self, num_classes=7):
        super(model1D, self).__init__()
        self.fc1 = nn.Linear(7*32*32, 1000)
        self.bn1 = nn.BatchNorm1d(num_features=1000)
        self.fc2 = nn.Linear(1000, 250)
        self.bn2 = nn.BatchNorm1d(num_features=250)
        self.fc3 = nn.Linear(250, num_classes)
        self.drop1=nn.Dropout(p=0.5) 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
    def forward(self, x):
        #print(x.size())
        x = x.view(x.size()[0],-1)
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.drop1(x)
        x = self.fc3(x)
        return x

        
        
        
        
class VAE3D(BaseModel):
    def __init__(self, num_feature=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
        

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)
    

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

    
class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_features):
        super(DenseBlock, self).__init__()
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, num_features, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(num_features, num_features, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(num_features*2, num_features, kernel_size=(3, 3, 3), padding=1)
        self.conv4 = nn.Conv3d(num_features*3, num_features, kernel_size=(3, 3, 3), padding=1)
        self.conv5 = nn.Conv3d(num_features*4, num_features, kernel_size=(3, 3, 3), padding=1)
    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        temp = torch.cat([conv1,conv2],1)
        c2_dense = self.relu(temp)
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1,conv2,conv3],1))
        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4], 1))
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5],1))
        return c5_dense

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_features):
        super(TransitionBlock, self).__init__()
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm3d(out_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.pool = nn.AvgPool3d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)
        out = self.pool(x)
        return out

class Dense3D(BaseModel):
    def __init__(self,num_feature=16, num_classes=2):
        super(Dense3D,self).__init__()
        self.relu = nn.LeakyReLU()
        self.num_feature= num_feature
        self.low_conv = nn.Conv3d(in_channels = 1, out_channels = num_feature*2, kernel_size = 5, padding = 2)
        self.dense1 = self._makeDense(DenseBlock, num_feature*2, self.num_feature)
        self.t1 = self._makeTransition(TransitionBlock, in_channels = num_feature*5, out_channels = num_feature*4)
        self.dense2 = self._makeDense(DenseBlock, num_feature*4, self.num_feature)
        self.t2 = self._makeTransition(TransitionBlock, in_channels = num_feature*5, out_channels = num_feature*2)
        self.bn = nn.BatchNorm3d(num_feature*2)
        self.pre_classifier = nn.Linear(num_feature*2*8*8*1, 512)
        self.drop=nn.Dropout(p=0.5)
        self.classifier = nn.Linear(512, num_classes)
       
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                # Kaming Initialization
                init.kaiming_normal_(m.weight.data)
                #m.bias.data.fill_(0)
                init.normal_(m.bias.data)
                
        
    def forward(self,x):
        x = self.relu(self.low_conv(x))
        x = self.dense1(x)
        #print(x.size())
        x = self.t1(x)
        #print(x.size())
        x = self.dense2(x)
        x = self.t2(x)
        #print(x.size())
        x = self.bn(x)
        #print(x.size())
        x = x.view(-1, self.num_feature*2*8*8*1)
        x = self.pre_classifier(x)
        x = self.drop(x)
        x = self.classifier(x)

        return x
    
    def _makeDense(self, block, in_channels, num_features):
        layers = []
        layers.append(block(in_channels, num_features))
        return  nn.Sequential(*layers)

    def _makeTransition(self, block, in_channels, out_channels):
        layers = []
        layers.append(block(in_channels, out_channels, self.num_feature))
        return  nn.Sequential(*layers)

