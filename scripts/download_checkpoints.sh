# new clap checkpoint
if [ -e ./checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt ]
then
    echo "clap checkpoint already downloaded"
else
    wget -P ./checkpoints 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt'
fi
