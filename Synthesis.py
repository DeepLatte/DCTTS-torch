import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from params import param
import numpy as np
from scipy.io.wavfile import write

from load_audio import textProcess, load_vocab
from data import speechDataset, collate_fn, att2img, plotAtt, plotMel
import networks_v1 as networks
import vocoder


class graph(nn.Module):
    def __init__(self, trNet):
        super(graph, self).__init__()
        self.trNet = trNet
        if self.trNet is "t2m":
            self.trainGraph = networks.t2mGraph().to(DEVICE)
            
        elif self.trNet is "SSRN":
            self.trainGraph = networks.SSRNGraph().to(DEVICE)

def dirLoad(modelNumb):
    modelPath = os.path.abspath('./model_{}'.format(modelNumb))
    logt2m = list(np.genfromtxt(os.path.join(modelPath, 't2m', 'log.csv'), delimiter=','))
    globalStep = int(logt2m[-1][0])
    t2mPATH = os.path.join(modelPath, 't2m', 'best_{}'.format(globalStep), 'bestModel_{}.pth'.format(globalStep))
    
    logSSRN = list(np.genfromtxt(os.path.join(modelPath, 'SSRN', 'log.csv'), delimiter=','))
    globalStep = int(logSSRN[0][0])
    ssrnPATH = os.path.join(modelPath, 'SSRN', 'best_{}'.format(globalStep), 'bestModel_{}.pth'.format(globalStep))

    testPATH = os.path.abspath(os.path.join(modelPath, 'synthesize'))
    wavPATH = os.path.abspath(os.path.join(testPATH, 'wav'))    
    imgPATH = os.path.abspath(os.path.join(testPATH, 'img'))
    if not os.path.exists(testPATH):
        os.mkdir(testPATH)
        os.mkdir(wavPATH)
        os.mkdir(imgPATH)

    return t2mPATH, ssrnPATH, wavPATH, imgPATH

def modelLoad(t2m, SSRN, t2mPATH, SSRNPATH):
    t2mCkpt = torch.load(t2mPATH)
    ssrnCkpt = torch.load(SSRNPATH)
    t2m.load_state_dict(t2mCkpt['model_state_dict'])
    SSRN.load_state_dict(ssrnCkpt['model_state_dict'])


def Synthesize(testLoader, idx2char, DEVICE, t2mPATH, ssrnPATH, wavPATH, imgPATH):
    # load trained model
    # sound output
    t2m = graph("t2m").trainGraph
    ssrn = graph("SSRN").trainGraph
    modelLoad(t2m, ssrn, t2mPATH, ssrnPATH) 
    
    with torch.no_grad():
        t2m.eval()
        ssrn.eval()
        globalStep = 0
        # wholeMag = torch.zeros(len(testLoader.dataset), param.max_T*param.r, param.n_mags).to(DEVICE)
        for idx, batchData in enumerate(testLoader):
            # predMel with zero values
            batchTxt, batchMel, _ = batchData
            batchTxt = batchTxt.to(DEVICE)
            # batchMel = batchMel.to(DEVICE)
            predMel = torch.zeros(param.B,  param.max_T, param.n_mels).to(DEVICE) # (B, T/r, n_mels)
            # At every time step, predict mel spectrogram
            for t in range(param.max_T-1):
                genMel, A, _  = t2m(batchTxt, predMel) #genMel : (B, n_mels, T/r)
                genMel_t = genMel[:, :, t]  # (B, n_mels)
                predMel[:, t, :] = genMel_t

            # SSRN
            predMag = ssrn(predMel) # Out : (B, T, n_mags)
            predMag = predMag.transpose(1, 2) # (B, n_mags, T*r)
            
            for idx, (text, mag) in enumerate(zip(batchTxt, predMag)):
                # alignment
                plotAtt(att2img(A[idx]), text, globalStep, imgPATH)
                plotMel(predMel[idx], globalStep, imgPATH)
                # vocoder
                wav = vocoder.spectrogram2wav(mag)
                write(data=wav,filename=os.path.join(wavPATH, 'wav_{}.wav'.format(globalStep)), rate=param.sr)
                globalStep += 1
            
        
if __name__ == "__main__":
    # input text sqeuence from user or use sample sequence.
    # text processing

    # modelNumb = int(sys.argv[1])
    modelNumb = 1
    trNet = 'SSRN'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Make Directory based on the model which has lowest loss value.
    t2mPATH, ssrnPATH, wavPATH, imgPATH = dirLoad(modelNumb)

    char2idx, idx2char = load_vocab()
    # testTxt, lenTxt = textProcess(testTxt, char2idx)
    
    # load Dataset
    testDataset = speechDataset(trNet, 2)
    testLoader = DataLoader(dataset=testDataset,
                            batch_size=param.B,
                            shuffle=False,
                            collate_fn=collate_fn,
                            drop_last=True)

    Synthesize(testLoader, idx2char, DEVICE, t2mPATH, ssrnPATH, wavPATH, imgPATH)