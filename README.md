# Open MusicLM
Pytorch implementation of [MusicLM](https://arxiv.org/abs/2301.11325), a SOTA text to music model published by Google, with a few modifications. We use [CLAP](https://github.com/LAION-AI/CLAP) as a replacement for MuLan, [Encodec](https://github.com/facebookresearch/encodec) as a replacement for SoundStream, and [MERT](https://huggingface.co/m-a-p/MERT-v0) as a replacement for w2v-BERT.

<p align='center'>
<img alt='diagram of MusicLM' src='musiclm.png' title="MusicLM" height='250px'>
<img alt='diagram of CLAP' src='clap.png' title="CLAP" height='250px'>
</p>

## Why CLAP?
CLAP is a joint audio-text model trained on [LAION-Audio-630K](https://github.com/LAION-AI/audio-dataset). Similar to MuLan, it consists of an audio tower and a text tower that project their respective media onto a shared latent space (512 dimensions in CLAP vs 128 dimensions in MuLan).

MuLan was trained on 50 million text-music pairs. Unfortunately I don't have the data or compute to replicate this, so I'm counting on using CLAP's pretrained checkpoints + some additional fine tuning to come close.

## Why Encodec?
SoundStream and Encodec are both neural audio codecs that encode any waveform to a sequence of acoustic tokens, which can then be decoded into a waveform resembling the original. These intermediate tokens can then be modeled as a seq2seq task. [Encodec](https://github.com/facebookresearch/encodec) is released by Facebook and pretrained checkpoints are publicly available, whereas this is not the case with SoundStream. However, Encodec has a restrictive license, so the plan is to use Encodec to verify that our implementation works and swap it out with @lucidrain's [SoundStream implementation](https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/soundstream.py) once the community is able to train it.

## Differences from @lucidrains implementation
- Autoregressively models the CLAP/MuLan conditioning signal by passing it into the transformers as discrete tokens, as mentioned in section 3.1 of the paper. Musiclm-pytorch conditions on them with cross attention.
- TokenConditionedTransformer can support variable token sequences, which makes it easy to do further experimentation (e.g. combining multiple conditioning signals, stereo waveform generation, etc.)
- Uses existing open source models instead of training MuLan and SoundStream.
- Some modifications to increase the chance of successfully training the model.

# End Goal
The goal of this project is to replicate the results of MusicLM as quickly as possible without necessarily sticking to the architecture in the paper. For those looking for a more true-to-form implementation, check out [musiclm-pytorch](https://github.com/lucidrains/musiclm-pytorch). 

We also seek to gain a better understanding of CLAP's latent space.

# Usage
## Install
```shell
conda env create -f environment.yaml
conda activate open-musiclm
```

## Training
In order to get a working model, there are 5 parts we need to train: clap rvq, hubert kmeans, semantic stage, coarse stage & fine stage.

All parts of the model are defined in a model config (in the `./configs/model/` directory). This config contains information such as number of layers, number of quantizers, target audio lengths for each stage, etc.

When training the model, we also use a training config (in the `./configs/training/` directory), which contains the hyperparameters for our run.

### CLAP RVQ
The first step is to train the residual vector quantizer that maps continuous CLAP embeds -> a discrete token sequence.
```shell
python ./scripts/train_clap_rvq.py \
    --results_folder ./results/clap_rvq \ # where to save results and checkpoints
    --model_config ./configs/model/musiclm_small.json \
    --training_config ./configs/training/train_musiclm_fma.json
```

### Hubert Kmeans
Next we learn the kmeans layer that we use to quantize our MERT embeddings into semantic tokens.
```shell
python ./scripts/train_hubert_kmeans.py \
    --results_folder ./results/hubert_kmeans \ # where to save results and checkpoints
    --model_config ./configs/model/musiclm_small.json \
    --training_config ./configs/training/train_musiclm_fma.json
```

### Semantic Stage + Coarse Stage + Fine Stage
Once we have a working kmeans and RVQ, we can now train the semantic, coarse and fine stages. These stages can be trained concurrently.
```shell
python ./scripts/train_semantic_stage.py \
    --results_folder ./results/semantic \ # where to save results and checkpoints
    --model_config ./configs/model/musiclm_small.json \
    --training_config ./configs/training/train_musiclm_fma.json \
    --rvq_path PATH_TO_RVQ_CHECKPOINT \ # path to previously trained rvq
    --kmeans_path PATH_TO_KMEANS_CHECKPOINT # path to previously trained kmeans
```
```shell
python ./scripts/train_coarse_stage.py \
    --results_folder ./results/coarse \ # where to save results and checkpoints
    --model_config ./configs/model/musiclm_small.json \
    --training_config ./configs/training/train_musiclm_fma.json \
    --rvq_path PATH_TO_RVQ_CHECKPOINT \ # path to previously trained rvq
    --kmeans_path PATH_TO_KMEANS_CHECKPOINT # path to previously trained kmeans
```
```shell
python ./scripts/train_fine_stage.py \
    --results_folder ./results/fine \ # where to save results and checkpoints
    --model_config ./configs/model/musiclm_small.json \
    --training_config ./configs/training/train_musiclm_fma.json \
    --rvq_path PATH_TO_RVQ_CHECKPOINT \ # path to previously trained rvq
    --kmeans_path PATH_TO_KMEANS_CHECKPOINT # path to previously trained kmeans
```

# Thank you
* [@lucidrains](https://github.com/lucidrains/) for the [audiolm-pytorch](https://github.com/lucidrains/audiolm-pytorch) implementation. This repo contains a refactored version of a lot of the code in [audiolm-pytorch](https://github.com/lucidrains/audiolm-pytorch).
* [LAION](https://laion.ai/) for [CLAP](https://github.com/LAION-AI/CLAP)
* [Music Audio Pretrain team](https://huggingface.co/m-a-p) for [MERT](https://huggingface.co/m-a-p/MERT-v0)

# Citations
```bibtex
@inproceedings{Agostinelli2023MusicLMGM,
    title     = {MusicLM: Generating Music From Text},
    author    = {Andrea Agostinelli and Timo I. Denk and Zal{\'a}n Borsos and Jesse Engel and Mauro Verzetti and Antoine Caillon and Qingqing Huang and Aren Jansen and Adam Roberts and Marco Tagliasacchi and Matthew Sharifi and Neil Zeghidour and C. Frank},
    year      = {2023}
}
```
```bibtex
@article{wu2022large,
  title     = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author    = {Wu, Yusong and Chen, Ke and Zhang, Tianyu and Hui, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  journal={arXiv preprint arXiv:2211:06687},
  year      = {2022},
}
```
```bibtex
@article{defossez2022highfi,
  title     = {High Fidelity Neural Audio Compression},
  author    = {Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal   = {arXiv preprint arXiv:2210.13438},
  year      = {2022}
}
```
