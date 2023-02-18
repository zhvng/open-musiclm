

if [ -e ./data/fma_large.zip ]
then
    echo "fma_large already downloaded"
else
    echo "downloading fma_large.zip, may take a while..."
    wget -P ./data https://os.unil.cloud.switch.ch/fma/fma_large.zip
fi

if [ -e ./data/fma_large ]
then
    echo "fma_large already unzipped"
else
    echo "unzipping fma_large.zip, may take a while..."
    echo "497109f4dd721066b5ce5e5f250ec604dc78939e  data/fma_large.zip"    | sha1sum -c -
    cd data
    unzip fma_large.zip
fi