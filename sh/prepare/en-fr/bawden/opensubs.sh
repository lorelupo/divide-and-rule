#!/bin/bash

# Move to ./data directory
cd data

# Set variables

src=en
tgt=fr
lang=$src-$tgt
prep=wmt14_en_fr
tmp=$prep/tmp
orig=bawden/Evaluating-discourse-in-NMT
prep=iwslt17.dnmt.en-fr/test_suites/bawden/large_pronoun
tmp=$prep/tmp
# mkdir bawden
# mkdir -p $tmp

N_THREADS=6
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

BPE_CODE=$prep/code
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=32000

# Setting utils ################################################################

echo "Looking for Bawden's github repository..."
if [ -d "$orig" ]; then
  echo "test suite github repository was already cloned here."
else
  echo 'Cloning test suite github repository.'
  cd bawden
  git clone https://github.com/rbawden/Evaluating-discourse-in-NMT.git
  cd ..
fi

echo 'Looking for Moses github repository (for tokenization scripts)...'
DIR="./mosesdecoder"
if [ -d "$DIR" ]; then
  echo "Moses repo was already cloned here."
else
  echo 'Cloning Moses github repository.'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi


echo 'Looking for Subword NMT repository (for BPE pre-processing)...'
DIR="./subword-nmt"
if [ -d "$DIR" ]; then
  echo "Subword NMT repo was already cloned here."
else
  echo 'Cloning Subword NMT repository.'
  git clone https://github.com/rsennrich/subword-nmt.git
fi

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

# Preparing data #############################################################
