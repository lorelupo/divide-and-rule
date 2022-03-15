#!/usr/bin/env bash

# Read script arguments and assign them to variables
# bash sh/prepare/en-fr/wmt14/large_pronoun.sh --k=3 --shuffle=True
for argument in "$@" 
do
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)   
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}" 
    fi
done

# Move to the data directory corresponding to the right language pair
src=en
tgt=fr
lang=$src-$tgt
DATA=$HOME/dev/fairseq/data/$lang
mkdir -p $DATA
cd $DATA

orig=bawden/discourse-mt-test-sets

CODE_SOURCE_DIR=wmt14/standard
TOOLS=../../tools
SCRIPTS=$TOOLS/mosesdecoder/scripts

BPEROOT=$TOOLS/subword-nmt/subword_nmt
BPE_CODE=$CODE_SOURCE_DIR/code
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
HEADS=../../scripts/retrieve_doc_heads.py

# Setting utils ###############################################################

echo 'Looking for discourse github repository...'
if [ -d "$orig" ]; then
  echo "discourse github repository was already cloned here."
else
  echo 'Cloning discourse github repository in $(pwd)/bawden.'
  cd bawden
  git clone https://github.com/rbawden/discourse-mt-test-sets.git
  cd ..
  # rename lexical-choice.json to be consistent with the other file names
  mv $orig/test-sets/lexical-choice.json $orig/test-sets/lexical_choice.json
fi


echo 'Looking for Moses github repository (for tokenization scripts)...'
DIR="$TOOLS/mosesdecoder"
if [ -d "$DIR" ]; then
  echo "Moses repo was already cloned here."
else
  echo 'Cloning Moses github repository.'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi


echo 'Looking for Subword NMT repository (for BPE pre-processing)...'
DIR="$TOOLS/subword-nmt"
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

for f in lexical_choice; do
  # Setting paths to specific test set
  prep=wmt14/test_suites/bawden/$f
  tmp=$prep/tmp
  mkdir bawden
  mkdir -p $tmp
  # Extracting sentences from json ##############################################
  if [ $f = "lexical_choice" ]; then
    python3 $orig/scripts/json2rawtext.py $orig/test-sets/lexical_choice.json lexical-choice $tmp
  else
    python3 $orig/scripts/json2rawtext.py $orig/test-sets/$f.json $f $tmp
  fi

  # Preprocessing ###############################################################
  echo "Pre-processing data..."
  for l in $src $tgt; do
    paste -d \\n $tmp/$f.prev.$l $tmp/$f.current.$l | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$f.$l
    seq 1 2 800 > $tmp/$f.$l.heads
    mv $tmp/$f.$l.heads $prep/test.$lang.$l.heads
  done

  # Print stats before BPE ######################################################
  wc -l $tmp/*

  # Applying BPE ################################################################
  for l in $src $tgt; do
    echo "Apply_bpe.py to ${f}.${l}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f.$l > $prep/$f.$l
  done

  # Build vocabularies and binarize training data ###############################
  rm -rf data-bin/$prep
  echo "Binarizing ${f}..."
  fairseq-preprocess \
      --source-lang $src \
      --target-lang $tgt \
      --testpref $prep/$f \
      --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
      --joined-dictionary \
      --destdir data-bin/$prep
  cp $prep/*.heads data-bin/$prep
done