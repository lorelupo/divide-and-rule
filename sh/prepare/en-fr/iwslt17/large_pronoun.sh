#!/usr/bin/env bash


# Read script arguments and assign them to variables
# bash sh/prepare/en-fr/iwslt17/large_pronoun.sh --k=3 --shuffle=True
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

# n. context sentences
if [ -n "$k" ] ; then c=$k ; else c=3 ; fi
if [ -n "$shuffle" ] ; then shuffle=$shuffle ; else shuffle=False ; fi

# Setting variables
DICTIONARY_DIR=wmt17/standard
INSERT=../../scripts/insert_lines.py
TOOLS=../../tools
SCRIPTS=$TOOLS/mosesdecoder/scripts
orig=bawden/Large-contrastive-pronoun-testset-EN-FR/OpenSubs/extracted
if [ $shuffle = "True" ]; then append=.shuffled; else append= ; fi
prep=wmt17/test_suites/large_pronoun/k$c$append

tmp=$prep/tmp
mkdir bawden
mkdir -p $tmp

BPEROOT=$TOOLS/subword-nmt/subword_nmt
BPE_CODE=$DICTIONARY_DIR/code
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
HEADS=../../scripts/retrieve_doc_heads.py

# Setting utils ###############################################################

echo 'Looking for test suite github repository...'
if [ -d "$orig" ]; then
  echo "test suite github repository was already cloned here."
else
  echo 'Cloning test suite github repository.'
  cd bawden
  git clone https://github.com/rbawden/Large-contrastive-pronoun-testset-EN-FR.git
  cd 
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

# # Preprocessing ###############################################################

for f in src trg; do
  if [ $f = "src" ]; then l=$src; else l=$tgt; fi
  # some sentences have an empty context, we replace them with "..."
  rm -rf $orig/full.c$c.$f$append
  if [ $shuffle = "True" ]; then perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' < $orig/OpenSubs.c$c.context.$f > $orig/OpenSubs.c$c.context.$f$append ; fi
  python $INSERT $orig/OpenSubs.c$c.context.$f$append $orig/OpenSubs.current.$f $orig/full.c$c.$f$append $c 
  if [ -f "$LC" ]; then
    echo "lowercasing test suite..."
    cat $orig/full.c$c.$f$append | \
    awk '!NF{$0="..."}1' | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $tmp/test.$l
  else
    cat $orig/full.c$c.$f | \
    awk '!NF{$0="..."}1' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/test.$l
  fi
  # generate dummy head file heads=[1]
  python $HEADS $orig/OpenSubs.current.$f
  mv $orig/OpenSubs.current.$f.heads $prep/test.$lang.$l.heads
done

# Print stats before BPE ######################################################

wc -l $tmp/*

# Applying BPE ################################################################

for l in $src $tgt; do
    echo "Apply_bpe.py to test.$l ..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/test.$l > $prep/test.$l
done

# # Build vocabularies and binarize training data ###############################

rm -rf data-bin/$prep
echo "Binarizing test suite..."
fairseq-preprocess \
    --source-lang $src \
    --target-lang $tgt \
    --testpref $prep/test \
    --srcdict data-bin/$DICTIONARY_DIR/dict.en.txt \
    --joined-dictionary \
    --destdir data-bin/$prep
cp $prep/*.heads data-bin/$prep/