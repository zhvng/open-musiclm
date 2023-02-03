<img alt='diagram of MusicLM' src='musiclm.png' title="MusicLM" height='200px'>
<img alt='diagram of CLAP' src='clap.png' title="CLAP" height='200px'>

# Open MusicLM - pytorch 🗽
Pytorch implementation of MusicLM that uses CLAP instead of MuLan.

Contains a refactored DRY version of much of the code in [audiolm-pytorch](https://github.com/lucidrains/audiolm-pytorch).


# Why CLAP?
CLAP is a joint audio-text model trained on [LAION-Audio-630K](https://github.com/LAION-AI/audio-dataset) that is similar in function to MuLan.

MuLan was trained on 50 million text-music pairs. Unfortunately as an open source project we do not have the data or compute to replicate this. However, we can probably use CLAP's pretrained checkpoints + some additional fine tuning to come close.

# End Goal

The goal of this project is to replicate the results of MusicLM as quickly as possible without necessarily sticking to the architecture in the paper. For those looking for a more true-to-form implementation, check out [musiclm-pytorch](https://github.com/lucidrains/musiclm-pytorch). 


# Thank you
* [lucidrains](https://github.com/lucidrains/) for his [audiolm-pytorch](https://github.com/lucidrains/audiolm-pytorch) implementation
* LAION for [CLAP](https://github.com/LAION-AI/CLAP)

# Citations
```
@inproceedings{Agostinelli2023MusicLMGM,
    title     = {MusicLM: Generating Music From Text},
    author    = {Andrea Agostinelli and Timo I. Denk and Zal{\'a}n Borsos and Jesse Engel and Mauro Verzetti and Antoine Caillon and Qingqing Huang and Aren Jansen and Adam Roberts and Marco Tagliasacchi and Matthew Sharifi and Neil Zeghidour and C. Frank},
    year      = {2023}
}
```
```
@article{wu2022large,
  title     = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author    = {Wu, Yusong and Chen, Ke and Zhang, Tianyu and Hui, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  journal={arXiv preprint arXiv:2211:06687},
  year      = {2022},
}
```