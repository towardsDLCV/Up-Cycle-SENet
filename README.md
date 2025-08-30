<div align="center">
  
<h1>Up-Cycle-SENet: Unpaired Phase-aware Speech Enhancement Using Deep Complex Cycle Adversarial Networks</h1>

<div>
    <a href='https://scholar.google.com/citations?user=5C9TeqgAAAAJ&hl=ko&oi=sra' target='_blank'>Cheolhoon Park</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=SIfp2fUAAAAJ&hl=ko&oi=sra' target='_blank'>Hyunduck Choi</a><sup>2,*</sup>&emsp;
</div>
<div>
    <sup>1</sup>Korea University <sup>2</sup>SeoulTech
</div>


<div>
    <h4 align="center">
        • <a href="https://www.sciencedirect.com/science/article/pii/S0925231225000372?via%3Dihub" target='_blank'>Neurocomputing 2025</a> •
    </h4>
</div>

## Abstract

<div style="text-align:center">
<img src="assets/teaser.png"  width="50%" height="50%">
</div>

</div>

>Speech enhancement (SE), which reconstructs intelligible speech by removing noise or interference from noisy speech, plays an important role in many speech applications. Due to the successful introduction of deep learning in SE, a significant performance improvement was recorded compared to the traditional methods. Most spectrogram-based deep SE networks have two main issues. First, many existing methods only focus on estimating the magnitude of the spectrogram while reusing the phase information. Reusing the phase part of a noisy spectrogram results in clear performance limitations, which can become more pronounced when the distortion caused by noise is severe. Second, most deep SE models adopt supervised learning, which requires a large number of paired datasets. Constructing a large dataset that includes clean speech is highly impractical due to the significant effort and cost involved. To address this issue, we propose UP-Cycle-SENet, an end-to-end complex SE network capable of estimating both the magnitude and phase parts of the speech spectrogram under unpaired dataset conditions. The proposed network leverages complex convolutional neural networks and extended modules to efficiently extract features in the complex domain without losing information. Additionally, the introduction of a CNN-based discriminator with non-autoregressive properties makes it suitable for fast training and inference. To effectively validate the benefits of the proposed network, comparative experiments were conducted using public datasets that mix Voice Bank and DEMAND. The experimental results demonstrated that the proposed framework outperforms previous methods in both parallel and non-parallel strategies.

## Preparation for data

We use a mixture of the Voice Bank corpus and the Environments Multichannel Acoustic Noise Database ([VoiceBank-Demand](https://datashare.ed.ac.uk/handle/10283/2791)). 

- For training, 11,572 utterances with 28 speakers
- For testing, 824 utterances with 2 invisible speakers

Note that all utterances were downsampled form 48kHz to 16kHz.
To convert audio files into 16kHz, run the following command:

```bash
sox <48K.wav> -r 16000 -c 1 -b 16 <16k.wav>
```

## Enviroment setup

```bash
conda create -n UpSE python=3.8
conda activate UpSE
pip install -r requirements.txt
```

## Training & Testing
Run the following command before training the model:

```bash
mkdir saved_models
```

### A single GPU

```python
python main.py --gpu 0 \
  --batch-size 4 --epochs 70 --n-epochs-decay 30 \
  --lr 2e-4 --lr-policy 'cosine' --loss_type 'l1gan'\
  --clean-train-dir <Path> --noisy-train-dir <Path> \
  --clean-test-dir <Path> --noisy-test-dir <Path> \
```

### Single node, mutiple GPUs:

```python
python main.py --world-size 1 --rank 0 \ 
  --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' \
  --multiprocessing-distributed \ 
  --batch-size 16 --epochs 70 --n-epochs-decay 30 \
  --lr 2e-4 --lr-policy 'cosine' --loss_type 'l1gan'\
  --clean-train-dir <Path> --noisy-train-dir <Path> \
  --clean-test-dir <Path> --noisy-test-dir <Path> \
```

### Evaluation
```python
python main.py --gpu 0 \
  --batch-size 1 -e --resume <Model Path> \
  --clean-test-dir <Path> --noisy-test-dir <Path> \
```

### Logging and Tensorboard
To view results in Tensorboard, run:

```bash
tensorboard --logdir runs
```

## Results

|                               Model                               | PESQ &uparrow; | STOI &uparrow; | CSIG &uparrow; | CBAK &uparrow; | COVL &uparrow; |
|:-----------------------------------------------------------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
|                               noisy                               |      1.97      |      0.92      |      3.35      |      2.44      |      2.63      |
|  [NyTT-1 (Chen et al., 2022)](https://arxiv.org/abs/2210.15368)   |      2.37      |      0.93      |       -        |       -        |       -        |
| [MetricGAN-U (Fu et al., 2022)](https://arxiv.org/abs/2110.05866) |      2.45      |       -        |      3.47      |      2.64      |      2.91      |
|    [CCGAN (Yu et al., 2022)](https://arxiv.org/abs/2109.12591)    |      2.56      |      0.92      |      3.67      |      3.10      |      3.16      |
|    [MCGAN (Yu et al., 2021)](https://arxiv.org/abs/2107.13143)    |      2.67      |      0.93      |      3.86      |      3.20      |      3.21      |
| [CinCGAN-SE (Yu et al., 2022)](https://arxiv.org/abs/2109.12591)  |      2.84      |      0.94      |    **4.10**    |      3.36      |      3.37      |
|                     **Up-Cycle-SENet (ours)**                     |    **2.86**    |    **0.95**    |      3.97      |    **3.59**    |    **3.46**    |

- Pretrained Up-Cycle-SENet: [*Link*](https://drive.google.com/drive/folders/1jeZofrOAcQPpweZsAoq7L8mmSYy1TEIk?usp=share_link)

## Citation
If you find our work useful, please consider citing:

```tex
@article{PARK_25_Neuro,
    title = {UP-Cycle-SENet: Unpaired phase-aware speech enhancement using deep complex cycle adversarial networks},
    journal = {Neurocomputing},
    volume = {624},
    pages = {129365},
    year = {2025},
    author = {Park, Cheol-Hoon and Choi, Hyun-Duck},
}
```

## Acknowledgement

Our code is based on these wonderful repos:
* [DCCRN](https://github.com/huyanxin/DeepComplexCRN)
* [MetricGAN](https://github.com/JasonSWFu/MetricGAN)
* [SpeechBrain](https://github.com/speechbrain/speechbrain/tree/develop)
* [DB-AIAT](https://github.com/yuguochencuc/DB-AIAT)
