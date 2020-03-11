# DCTTS_TORCH

Implement DCTTS based on pytorch, the speech synthesis model based on convolutional neural network. This model yet supports only English.
You would find original paper at : [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.](https://arxiv.org/abs/1710.08969)

### Datasets

English - [The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

### Libraries

- PyTorch
- librosa
- tqdm

### How to use

1. **Get ready with LJ speech dataset.**
- Download the dataset.
- Unzip `LJSpeech-1.1.tar.gz` at `../speech_data/LJSpeech-1.1`
2. **Preprocess**
- run `load_audio.py`
3. **Train Model(Text2Mel or SSRN)**
- 1st Args* : Model you want to train. "0 for Text2Mel, 1 for SSRN".
- 2nd Args* : Put integer number `n` then it will check the folder named `Model_n` is existed or not. The folder will created if the folder is absent.
- 3rd Args* : (Optional, Default=0) `1` for training with pre-trained model.
- Train Text2Mel
   `$ python train.py 0 0`
- Train SSRN
   `$ python train.py 1 0`
4. **Synthesis**
Synthesize(Test) with test dataset.
*1st Args* : The number of folder `Model_n` which has trained models. Both models should exist in that folder.
`$ python Synthesis.py 0`

## Reference

I've refer to below articles and source codes from github. 

- [https://github.com/Kyubyong/dc_tts](https://github.com/Kyubyong/dc_tts)
- [https://github.com/chaiyujin/dctts-pytorch](https://github.com/chaiyujin/dctts-pytorch)
- [https://github.com/Yangyangii/DeepConvolutionalTTS-pytorch](https://github.com/Yangyangii/DeepConvolutionalTTS-pytorch)
- [https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
