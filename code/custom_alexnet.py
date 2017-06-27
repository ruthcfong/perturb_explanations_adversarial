import torch.nn as nn
import torchvision.models as models

def load_weights(m1, m2):
    s1 = m1.state_dict()
    s2 = m2.state_dict()
    for k2 in s2.keys():
        if k2 in s1 and s2[k2].size() == s1[k2].size():
            s1[k2] = s2[k2]
    m1.load_state_dict(s1) 
    return m1

class CustomAlexNet(nn.Module):

    def __init__(self, num_classes=2, num_input_channels=1, pretrained=False):
        super(CustomAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        if pretrained:
            alexnet = models.alexnet(pretrained=True)
            self.load_weights(alexnet)
            #self.features = load_weights(self.features, alexnet.features)
            #self.classifier = load_weights(self.classifier, alexnet.classifier)
            #for i in range(len(self.features)):
            #    if (hasattr(self.features[i], 'weight') 
            #            and self.features[i].in_channels == alexnet.features[i].in_channels 
            #            and self.features[i].out_channels == alexnet.features[i].out_channels):
            #        self.features[i].weight = alexnet.features[i].weight.copy()
            #        if hasattr(self.features[i], 'bias':
            #            self.features[i].bias = alexnet.features[i].bias.copy()

    def load_weights(self, pretrained_net):
        s1 = self.state_dict()
        s2 = pretrained_net.state_dict()
        for k2 in s2.keys():
            if k2 in s1 and s2[k2].size() == s1[k2].size():
                s1[k2] = s2[k2]
        self.load_state_dict(s1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
