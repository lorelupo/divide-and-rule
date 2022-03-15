#!/bin/bash


model_dir=en2fr_wmt14_transformer
data_dir=wmt14.en-fr

MODEL_URL=https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2
DATA_URL=https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2

if [ -d "checkpoints/$model_dir" ]; then
  echo "Model is already available in checkpoints/$model_dir."
else
  # wget $MODEL_URL
  echo "Model successfully downloaded."
  tar -xvjf wmt14.en-fr.joined-dict.transformer.tar.bz2 --totals=USR1
  rm wmt14.en-fr.joined-dict.transformer.tar.bz2
  mv wmt14.en-fr.joined-dict.transformer checkpoints/$model_dir
fi


if [ -d "data/data-bin/$data_dir" ]; then
  echo "Data are already available in data/data-bin/$data_dir."
else
  wget $DATA_URL
  echo "Data successfully downloaded."
  tar -xvjf wmt14.en-fr.joined-dict.newstest2014.tar.bz2
  rm wmt14.en-fr.joined-dict.newstest2014.tar.bz2
  mv wmt14.en-fr.joined-dict.newstest2014 data/data-bin/$data_dir
fi

