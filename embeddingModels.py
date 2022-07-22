import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.nn import functional as F

# Embedding Model
class InceptionResnet(nn.Module):
    def __init__(self, device, dropout=0.3):
        super(InceptionResnet, self).__init__()

        #create an InceptionResNetV1 model pretrained on "VGGFace2" Dataset
        self.net = InceptionResnetV1(pretrained='vggface2', dropout_prob=dropout, device=device)

        # set the output channel number equal to the input channel number
        self.out_features = self.net.last_linear.in_features

    def forward(self, x):
        # return a 512 dimension embedding
        return self.net(x)


# Similar to google's FaceNet
class MaskNet(nn.Module):
    def __init__(self, emb_size=512, dropout=0.0, device='cuda'):
        super(MaskNet, self).__init__()
        
        # Backbone
        self.model = InceptionResnet(device, dropout=dropout)
        
        # Add a Global Average Pooling Layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # neck
        self.neck = nn.Sequential(
                nn.Linear(self.model.out_features, emb_size, bias=True),
                nn.BatchNorm1d(emb_size, eps=0.001),
            )
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # backbone
        return self.model(x)