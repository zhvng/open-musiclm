wget -P ./data https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
cd data
unzip fma_metadata.zip