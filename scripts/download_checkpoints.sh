# wav2vec k-means
if [ -e ./checkpoints/vq-wav2vec_kmeans.pt ]
then
    echo "wav2vec k-means checkpoint already exists"
else
    wget -P ./checkpoints  https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt 
fi

if [ -e ./checkpoints/clap-laion-audioset-fusion.pt ]
then
    echo "clap checkpoint already exists"
else
    gdown 1zJYnqdjBvX5iFwfudqSPvCzDyAQE3byA -O ./checkpoints/clap-laion-audioset-fusion.pt
fi

if [ -e ./checkpoints/hubert_base_ls960.pt ]
then
    echo "hubert checkpoint already exists"
else
    wget -P ./checkpoints https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt 
    wget -P ./checkpoints https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin
fi