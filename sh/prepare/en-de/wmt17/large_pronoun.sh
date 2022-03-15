#!/usr/bin/env bash

# Read script arguments and assign them to variables
# bash sh/prepare/en-de/wmt17/large_pronoun.sh --k=3 --shuffle=False
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
tgt=de
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
orig=test_suites/ContraPro
if [ $shuffle = "True" ]; then append=.shuffled; else append= ; fi
prep=wmt17/test_suites/large_pronoun/k$c$append

tmp=$prep/tmp
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
  echo "cloning ContraPro and downloading data, this will take several minutes"
  git clone https://github.com/ZurichNLP/ContraPro
  cd ContraPro
  ./setup_opensubs.sh
  perl conversion_scripts/json2text_and_context.pl --source $src --target $tgt \
  --dir documents --json contrapro.json --context $c
fi

echo 'Looking for Moses github repository (for tokenization scripts)...'
if [ -d "$SCRIPTS" ]; then
  echo "Moses repo was already cloned here."
else
  echo 'Cloning Moses github repository.'
  git clone https://github.com/moses-smt/mosesdecoder.git $TOOLS/
fi

echo 'Looking for Subword NMT repository (for BPE pre-processing)...'
if [ -d "$BPEROOT" ]; then
  echo "Subword NMT repo was already cloned here."
else
  echo 'Cloning Subword NMT repository.'
  git clone https://github.com/rsennrich/subword-nmt.git $TOOLS/
fi

# Preprocessing ###############################################################
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

for l in $src $tgt; do
  rm -rf $orig/full.$c.$l$append
  if [ $shuffle = "True" ]; then shuf --random-source=<(get_seeded_random 0) < $orig/contrapro.context.$l > $orig/contrapro.context.$l$append ; fi
  python $INSERT $orig/contrapro.context.$l$append $orig/contrapro.text.$l $orig/full.c$c.$l$append $c 
  # some sentences have an empty context, we replace them with "..."
  cat $orig/full.c$c.$l$append | \
  awk '!NF{$0="..."}1' | \
  perl $TOKENIZER -threads 8 -l $l > $tmp/test.$l
  # generate heads
  sents=$(wc -l $tmp/test.$l | awk '{print $1;}')
  seq 1 $(( $k + 1 )) $sents > $prep/test.$lang.$l.heads
done

# Applying BPE ################################################################

for l in $src $tgt; do
    echo "Apply_bpe.py to test.$l ..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/test.$l > $prep/test.$l
done

# Build vocabularies and binarize training data ###############################

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