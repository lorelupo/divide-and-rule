#!/usr/bin/env bash
# Move to ./data directory
cd data

# Setting variables
src=en
tgt=fr
lang=$src-$tgt
orig=bawden/discourse-mt-test-sets
prep=iwslt17.dnmt.en-fr/test_suites/bawden/
tmp=$prep/discourse/tmp
mkdir bawden
mkdir -p $tmp

DICTIONARY_DIR=iwslt17.dnmt.en-fr/standard

BPEROOT=subword-nmt/subword_nmt
BPE_CODE=$DICTIONARY_DIR/code
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
HEADS=../scripts/retrieve_doc_heads.py

# Setting utils ###############################################################

echo 'Looking for discourse github repository...'
if [ -d "$orig" ]; then
  echo "discourse github repository was already cloned here."
else
  echo 'Cloning discourse github repository.'
  cd bawden
  git clone https://github.com/rbawden/discourse-mt-test-sets.git
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

# Extracting sentences from json ##############################################

python3 $orig/scripts/json2rawtext.py $orig/test-sets/anaphora.json anaphora $tmp
python3 $orig/scripts/json2rawtext.py $orig/test-sets/lexical_choice.json lexical-choice $tmp

# rename lexical-choice.json to be consistent with the other file names
mv $orig/test-sets/lexical-choice.json $orig/test-sets/lexical_choice.json

# Preprocessing ###############################################################

echo "Pre-processing data..."
for f in anaphora lexical_choice; do
  for l in $src $tgt; do
    paste -d \\n $tmp/$f.prev.$l $tmp/$f.current.$l | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $tmp/$f.$l
    seq 1 2 800 > $tmp/$f.$l.heads
    mv $tmp/$f.$l.heads $prep/discourse/test.$lang.$l.heads
  done
done

# Print stats before BPE ######################################################

wc -l $tmp/*

# Applying BPE ################################################################

for l in $src $tgt; do
    for f in anaphora.$l lexical_choice.$l; do
        echo "Apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/discourse/$f
    done
done

# Build vocabularies and binarize training data ###############################

rm -rf data-bin/$prep
for f in anaphora lexical_choice; do
  echo "Binarizing ${f}..."
  fairseq-preprocess \
      --source-lang $src \
      --target-lang $tgt \
      --testpref $prep/discourse/$f \
      --srcdict data-bin/$DICTIONARY_DIR/dict.en.txt \
      --joined-dictionary \
      --destdir data-bin/$prep/$f
  cp $prep/$f*.heads data-bin/$prep/$f/
done