#!/usr/bin/env bash
#
# Consistency test sets prepared by Voita et al., 2019 (When a Good Translation is Wrong in Context)
#
# bash sh/prepare/en-ru/voita_opensubs/consistency.sh standard/jumping4to4 --set=deixis_dev (--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/consistency.sh standard/jumping4to4 --set=deixis_test (--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/consistency.sh standard/jumping4to4 --set=lex_cohesion_dev(--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/consistency.sh standard/jumping4to4 --set=lex_cohesion_test (--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/consistency.sh standard/jumping4to4 --set=ellipsis_infl (--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/consistency.sh standard/jumping4to4 --set=ellipsis_vp (--case=lowercase)

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
if [ -n "$set" ]; then set=$set ; else set=deixis_dev ; fi
if [ -n "$case" ]; then case=$case ; else case=truecase ; fi

# Move to the data directory corresponding to the right language pair
src=en
tgt=ru
lang=$src-$tgt
DATA=$HOME/dev/fairseq/data/$lang
mkdir -p $DATA
cd $DATA

# Setting variables
if [ $case = "lowercase" ]
then
    corpus=voita_opensubs/testset_consistency
else
    corpus=$case\_voita_opensubs/testset_consistency
fi

orig=test_suites/good-translation-wrong-in-context/consistency_testsets/scoring_data
BPE_TOKENS=32000
N_THREADS=8

# Standard variables
TOOLS=../../tools
BPEROOT=$TOOLS/subword-nmt/subword_nmt

HEADS=../../scripts/retrieve_doc_heads.py

###############################################################################
if [ $1 = "standard" ]
then
    # Setting variables for the current option
    prep=$corpus/$set
    tmp=$prep/tmp
    mkdir -p $tmp

    echo "Pre-processing data..."
    if [ $case = "lowercase" ]
    then
        echo "Text is being lowercased!"
        # add a blank line between blocks of sentences, replace _eos with newline, lowercase
        cat $orig/$set.src | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | python -c "import sys; print(sys.stdin.read().lower())" | head -n -1 > $tmp/test.$src
        cat $orig/$set.dst | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | python -c "import sys; print(sys.stdin.read().lower())" | head -n -1 > $tmp/test.$tgt
    else
        # add a blank line between blocks of sentences, replace _eos with newline
        cat $orig/$set.src | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" > $tmp/test.$src
        cat $orig/$set.dst | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" > $tmp/test.$tgt
    fi

    # retrieve indices of headlines
    for l in $src $tgt; do
        python $HEADS $tmp/test.$l
        mv $tmp/test.$l.heads $prep/test.$lang.$l.heads
    done

    echo "Applying BPEs..."
    if [ $case = "lowercase" ]
    then
        CODE_SOURCE_DIR=voita_opensubs/context_agnostic/standard
    else
        CODE_SOURCE_DIR=$case\_voita_opensubs/context_agnostic/standard
    fi
    BPE_CODE=$CODE_SOURCE_DIR/code
    for l in $src $tgt; do
        python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l < $tmp/test.$l > $prep/test.$l
    done

    echo "Building vocabulary and binarizing data..." 
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --testpref $prep/test \
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.$src.txt \
        --tgtdict data-bin/$CODE_SOURCE_DIR/dict.$tgt.txt \
        --destdir data-bin/$prep \
        --workers $N_THREADS
    cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "jumping4to4" ]
then
    # Setting variables for the current option
    prep=$corpus/$1/$set
    tmp=$prep/tmp
    mkdir -p $tmp

    echo "Applying BPEs..."
    if [ $case = "lowercase" ]
    then
        CODE_SOURCE_DIR=voita_opensubs/context_aware/$1
    else
        CODE_SOURCE_DIR=$case\_voita_opensubs/context_aware/$1
    fi
    BPE_CODE=$CODE_SOURCE_DIR/code
    if [ $case = "lowercase" ]
    then
        echo "Text is being lowercased!"
        # source
        python -c "import sys; print(sys.stdin.read().lower())" < $orig/$set.src | head -n -1 | \
            python $BPEROOT/apply_bpe.py -c $BPE_CODE.$src > $prep/test.$src
        # target
        python -c "import sys; print(sys.stdin.read().lower())" < $orig/$set.dst | head -n -1 | \
            python $BPEROOT/apply_bpe.py -c $BPE_CODE.$tgt > $prep/test.$tgt
    else
        # source
        python $BPEROOT/apply_bpe.py -c $BPE_CODE.$src < $orig/$set.src > $prep/test.$src
        # target
        python $BPEROOT/apply_bpe.py -c $BPE_CODE.$tgt $orig/$set.dst > $prep/test.$tgt
    fi

    echo "Building vocabulary and binarizing data..." 
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --testpref $prep/test \
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.$src.txt \
        --tgtdict data-bin/$CODE_SOURCE_DIR/dict.$tgt.txt \
        --destdir data-bin/$prep \
        --workers $N_THREADS
###############################################################################
else
    echo "Argument $1 is not valid."
fi