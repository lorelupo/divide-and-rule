#!/usr/bin/env bash
#
# OpenSubtitles2018 as prepared by Voita et al., 2019 (When a Good Translation is Wrong in Context)
#
# wget -o voita_opensubs.zip https://www.dropbox.com/s/5drjpx07541eqst/acl19_good_translation_wrong_in_context.zip?dl=1
# unzip voita_opensubs.zip
#
# bash sh/prepare/en-ru/voita_opensubs/sent_all.sh standard (--case=lowercase)

# Read script arguments and assign them to variables
for argument in "$@" 
do
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)   
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}" 
   fi
done
# Set true-case/lower-case option
if [ -n "$case" ]; then case=$case ; else case=truecase ; fi

# Move to the data directory corresponding to the right language pair
src=en
tgt=ru
lang=$src-$tgt
DATA=$HOME/dev/fairseq/data/$lang
mkdir -p $DATA
cd $DATA

# Setting variables
corpus=voita_opensubs/context_agnostic
orig=/video/getalp/mt-data/$lang/$corpus
if [ $case != "lowercase" ]
then
    corpus=$case\_$corpus
fi
BPE_TOKENS=24000
N_THREADS=8

# Standard variables
TOOLS=../../tools
BPEROOT=$TOOLS/subword-nmt/subword_nmt

HEADS=../../scripts/retrieve_doc_heads.py

###############################################################################
if [ $1 = "standard" ]
then
    # Setting variables for the current option
    prep=$corpus/standard
    tmp=$prep/tmp
    mkdir -p $tmp

    TRAIN=$tmp/train
    BPE_CODE=$prep/code
    rm -f $TRAIN*
    for l in $src $tgt; do
        if [ $case = "lowercase" ]
        then
            echo "Text is being lowercased!"
            cat $orig/$l\_train | \
            python -c "import sys; print(sys.stdin.read().lower())" >> $TRAIN.$l
            echo "Learning BPE on ${TRAIN}.${l}..."
            python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN.$l > $BPE_CODE.$l
        else
            echo "Learning BPE on ${TRAIN}.${l}..."
            python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $orig/$l\_train > $BPE_CODE.$l
        fi
    done
    rm -f $TRAIN*

    # TRAIN=$tmp/train.$lang
    # BPE_CODE=$prep/code
    # rm -f $TRAIN
    # for l in $src $tgt; do
    #     if [ $case = "lowercase" ]
    #     then
    #         echo "Text is being lowercased!"
    #         cat $orig/$l\_train | \
    #         python -c "import sys; print(sys.stdin.read().lower())" >> $TRAIN
    #     else
    #         cat $orig/$l\_train >> $TRAIN
    #     fi
    # done

    # echo "Learning BPE on ${TRAIN}..."
    # python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
    # rm -f $TRAIN


    echo "Applying BPE..."
    for l in $src $tgt; do
        if [ $case = "lowercase" ]
        then
            echo "apply_bpe.py to train.$l ..."
            python -c "import sys; print(sys.stdin.read().lower())" < $orig/$l\_train | \
                python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l  > $prep/train.$l
            echo "apply_bpe.py to valid.$l ..."
            python -c "import sys; print(sys.stdin.read().lower())" < $orig/$l\_dev | \
                python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l > $prep/valid.$l
            echo "apply_bpe.py to test.$l ..."
            python -c "import sys; print(sys.stdin.read().lower())" < $orig/$l\_test | \
                python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l > $prep/test.$l
        else
            echo "apply_bpe.py to train.$l ..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l < $orig/$l\_train > $prep/train.$l
            echo "apply_bpe.py to valid.$l ..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l < $orig/$l\_dev > $prep/valid.$l
            echo "apply_bpe.py to test.$l ..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l < $orig/$l\_test > $prep/test.$l
        fi
    done
    echo "Building vocabulary and binarizing data..." 
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --trainpref $prep/train \
        --validpref $prep/valid \
        --testpref $prep/test \
        --destdir data-bin/$prep \
        --workers $N_THREADS
else
    echo "Argument is not valid."
fi