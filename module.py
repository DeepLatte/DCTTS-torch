import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.nn.utils import weight_norm as norm

import numpy as np
from params import param

class Embed(nn.Module):
    def __init__(self, vocabSize, numUnits):
        # self.embedLayer = torch.nn.Linear(vocabSize, numUnits, bias=False)
        # nn.init.xavier_uniform(self.W.weight)
        '''
        num_embeddings : size of the dictionary of embeddings
        embedding_dim  : the size of each embedding vector
        '''
        super(Embed, self).__init__()
        self.embedLayer = nn.Embedding(num_embeddings = vocabSize,
                                       embedding_dim = numUnits,
                                       )


    def forward(self, input):
        embedOut = self.embedLayer(input)

        return embedOut

class Cv(nn.Module):
    def __init__(self, inChannel, outChannel, kernelSize,
                padding, dilation, activationF = None):
        #nn.conv1d(in_channels, out_channels, kernel size, stride, padding,
        #          dilation, ...)
        #
        super(Cv, self).__init__()
        padDic = {"same" : (kernelSize-1)*dilation // 2, 
                  "causal" : (kernelSize-1)*dilation,
                  "none" : 0}
        self.pad = padding.lower()
        self.padValue = padDic[self.pad]
        self.convOne = norm(nn.Conv1d(in_channels=inChannel, 
                                 out_channels=outChannel, 
                                 kernel_size=kernelSize,
                                 stride=1,
                                 padding=self.padValue,
                                 dilation=dilation))
        self.activationF = activationF


    def forward(self, input):
        cvOut = self.convOne(input)
        
        # In Causal mode, drop the right side of the outputs
        if self.pad == "causal" and self.padValue > 0:
            cvOut = cvOut[:, :, :-self.padValue]

        # activation Function
        if self.activationF in param.actFDic.keys():
            cvOut = param.actFDic[self.activationF](cvOut)
        elif self.activationF == None:
            pass
        else:
            raise ValueError("You should use appropriate actvation Function argument. \
                             [None, 'ReLU', 'sigmoid'].")
            
        return cvOut

class Dc(nn.Module):
    '''
    Transposed Convolution 1d
    Lout = (Lin - 1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding*1
    '''
    def __init__(self, inChannel, outChannel, kernelSize,
                padding, dilation, activationF = None):
        super(Dc, self).__init__()
        padDic = {"same" : dilation*(kernelSize-1)//2,
                  "causal" : dilation*(kernelSize-1),
                  "none" : 0}
        self.pad = padding.lower()
        self.padValue = padDic[self.pad]
        self.transposedCv = norm(nn.ConvTranspose1d(in_channels=inChannel,
                                               out_channels=outChannel,
                                               kernel_size=kernelSize,
                                               stride=2,
                                               padding=self.padValue,
                                               dilation=dilation))
        self.activationF = activationF

    def forward(self, input):
        DcOut = self.transposedCv(input)

        if self.pad == "causal":
            cvOut = cvOut[:, :, :-self.padValue]

        # activation Function
        if self.activationF in param.actFDic.keys():
            DcOut = param.actFDic[self.activationF](DcOut)
        elif self.activationF == None:
            pass
        else:
            raise ValueError("You should use appropriate actvation Function argument. \
                             [None, 'ReLU', 'sigmoid'].")
            
        return DcOut

class Hc(Cv):
    '''
    Highway Network and Convolution

    '''
    def __init__(self, inChannel, outChannel, kernelSize,
                padding, dilation):
        super(Hc, self).__init__(inChannel, outChannel*2, kernelSize,
                            padding, dilation, None)
    
    def forward(self, input):
        L = super(Hc, self).forward(input)
        H1, H2 = torch.chunk(L, 2, 1) # Divide L along axis 1 to get 2 matrices.
        self.Output = torch.sigmoid(H1) * H2 + (1-torch.sigmoid(H1)) * input

        return self.Output

