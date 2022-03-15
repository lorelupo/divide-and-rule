#!/usr/bin/env bash


# Read script arguments and assign them to variables
# bash sh/prepare/en-de/wmt17/large_pronoun_testset.sh --k=3 --shuffle=False
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
curr=$(pwd)
data=$curr/data/$lang
mkdir -p $data
cd $data

# n. context sentences
if [ -n "$k" ] ; then c=$k ; else c=3 ; fi
if [ -n "$shuffle" ] ; then shuffle=$shuffle ; else shuffle=False ; fi

# Setting variables
B=$(( $c + 1 ))
S=0
E=3

DICTIONARY_DIR=wmt17/standard
FILTER=../../scripts/filter_text_blocks.py

if [ $shuffle = "True" ]; then append=.shuffled; else append= ; fi
orig=wmt17/test_suites/large_pronoun/k$c$append
prep=wmt17/test_suites/large_pronoun_testset/k$c$append
mkdir -p $prep

# Retrieving correct blocks from contrastive examples #########################

for l in $src $tgt; do
    # retrieve full text
    python $FILTER $orig/test.$l $prep/test.$l --B=$B --S=$S --E=$E
done

# generate heads
sents=$(wc -l $prep/test.$src | awk '{print $1;}')
seq 1 $B $sents > $prep/test.$lang.$src.heads
cp $prep/test.$lang.$src.heads $prep/test.$lang.$tgt.heads

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