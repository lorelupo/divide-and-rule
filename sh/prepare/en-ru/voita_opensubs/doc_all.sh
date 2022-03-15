#!/usr/bin/env bash
#
# OpenSubtitles2018 as prepared by Voita et al., 2019 (When a Good Translation is Wrong in Context)
#
# wget -o voita_opensubs.zip https://www.dropbox.com/s/5drjpx07541eqst/acl19_good_translation_wrong_in_context.zip?dl=1
# unzip voita_opensubs.zip
#
# bash sh/prepare/en-ru/voita_opensubs/doc_all.sh jumping4to4 (--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/doc_all.sh standard (--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/doc_all.sh split (--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/doc_all.sh nonredundant (--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/doc_all.sh small10 (--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/doc_all.sh small10split (--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/doc_all.sh dummy (--case=lowercase)
# bash sh/prepare/en-ru/voita_opensubs/doc_all.sh alldata (--case=lowercase)

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
if [ $case = "lowercase" ]
then
    corpus=voita_opensubs/context_aware
    CODE_SOURCE_DIR=voita_opensubs/context_agnostic/standard
else
    corpus=$case\_voita_opensubs/context_aware
    CODE_SOURCE_DIR=$case\_voita_opensubs/context_agnostic/standard
fi
    
orig=/video/getalp/mt-data/en-ru/voita_opensubs/context_aware
BPE_CODE=$CODE_SOURCE_DIR/code
BPE_TOKENS=24000
N_THREADS=8

# Standard variables
TOOLS=../../tools
BPEROOT=$TOOLS/subword-nmt/subword_nmt

HEADS=../../scripts/retrieve_doc_heads.py

###############################################################################
if [ $1 = "jumping4to4" ]
then
    # Setting variables for the current option
    prep=$corpus/jumping4to4
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
###############################################################################
elif [ $1 = "dummy-jumping4to4" ]
then
    # Setting variables for the current option
    prep=$corpus/dummy_jumping4to4
    tmp=$prep/tmp
    mkdir -p $tmp

    CODE_SOURCE_DIR=$corpus/jumping4to4
    BPE_CODE=$CODE_SOURCE_DIR/code
    N_train=100
    N_valid=20
    N_test=20
    
    echo "Applying BPE..."
    for l in $src $tgt; do
        if [ $case = "lowercase" ]
        then
            echo "apply_bpe.py to train.$l ..."
            head -n$N_train $orig/$l\_train | \
                python -c "import sys; print(sys.stdin.read().lower())" | \
                python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l | \
                head -n -1  > $prep/train.$l
            echo "apply_bpe.py to valid.$l ..."
            head -n $N_valid $orig/$l\_dev | \
                python -c "import sys; print(sys.stdin.read().lower())" | \
                python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l | \
                head -n -1 > $prep/valid.$l
            echo "apply_bpe.py to test.$l ..."
            head -n $N_test $orig/$l\_test | \
                python -c "import sys; print(sys.stdin.read().lower())" | \
                python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l | \
                head -n -1 > $prep/test.$l
        else
            echo "apply_bpe.py to train.$l ..."
            head -n$N_train $orig/$l\_train | python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l | head -n -1 > $prep/train.$l
            echo "apply_bpe.py to valid.$l ..."
            head -n $N_valid $orig/$l\_dev | python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l | head -n -1 > $prep/valid.$l
            echo "apply_bpe.py to test.$l ..."
            head -n $N_test $orig/$l\_test | python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l | head -n -1 > $prep/test.$l
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
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
        --tgtdict data-bin/$CODE_SOURCE_DIR/dict.ru.txt \
        --destdir data-bin/$prep \
        --workers $N_THREADS
###############################################################################
elif [ $1 = "standard" ]
then
    # Setting variables for the current option
    prep=$corpus/standard
    tmp=$prep/tmp
    mkdir -p $tmp

    echo "Pre-processing data"
    for l in $src $tgt; do
        if [ $case = "lowercase" ]
        then
            echo "Text is being lowercased!"
            # add a blank line between blocks of sentences, replace _eos with newline, lowercase
            cat $orig/$l\_train | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/train.$l
            #
            cat $orig/$l\_dev | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/valid.$l
            #
            cat $orig/$l\_test | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/test.$l
        else
            # add a blank line between blocks of sentences, replace _eos with newline
            cat $orig/$l\_train | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | head -n -1 > $tmp/train.$l
            cat $orig/$l\_dev | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | head -n -1 > $tmp/valid.$l
            cat $orig/$l\_test | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | head -n -1 > $tmp/test.$l
        fi
        # retrieve indices of headlines
        python $HEADS $tmp/train.$l
        python $HEADS $tmp/valid.$l
        python $HEADS $tmp/test.$l
        mv $tmp/train.$l.heads $prep/train.$lang.$l.heads
        mv $tmp/valid.$l.heads $prep/valid.$lang.$l.heads
        mv $tmp/test.$l.heads $prep/test.$lang.$l.heads
    done

    echo "Applying BPEs..."
    for l in $src $tgt; do
        for f in train.$l valid.$l test.$l; do
            echo "Apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l < $tmp/$f > $prep/$f
        done
    done
    
    echo "Building vocabulary and binarizing data..." 
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --trainpref $prep/train \
        --validpref $prep/valid \
        --testpref $prep/test \
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
        --tgtdict data-bin/$CODE_SOURCE_DIR/dict.ru.txt \
        --destdir data-bin/$prep \
        --workers $N_THREADS
    cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "split" ]
then
    # Set variables for the current option
    prep=$corpus/split
    tmp=$prep/tmp
    mkdir -p $tmp

    echo "Pre-processing data"
    for l in $src $tgt; do
        if [ $case = "lowercase" ]
        then
            echo "Text is being lowercased!"
            # add a blank line between blocks of sentences, replace _eos with newline, lowercase
            cat $orig/$l\_train | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/train.$l
            cat $orig/$l\_dev | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/valid.$l
        else
            # add a blank line between blocks of sentences, replace _eos with newline
            cat $orig/$l\_train | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | head -n -1 > $tmp/train.$l
            cat $orig/$l\_dev | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | head -n -1 > $tmp/valid.$l
        fi
    done

    echo "Splitting sentences..."
    for f in train valid ; do
        python ../../scripts/split_corpus_sentences.py \
        --src=$tmp/$f.$src --tgt=$tmp/$f.$tgt
        rm -f $tmp/$f.$src
        rm -f $tmp/$f.$tgt
        mv $tmp/$f.$src.split $tmp/$f.$src
        mv $tmp/$f.$tgt.split $tmp/$f.$tgt
    done

    echo "Retrieving documents' heads and delete empty lines..."
    for l in $src $tgt; do
        # retrieve indices of headlines
        python $HEADS $tmp/train.$l
        python $HEADS $tmp/valid.$l
        mv $tmp/train.$l.heads $prep/train.$lang.$l.heads
        mv $tmp/valid.$l.heads $prep/valid.$lang.$l.heads
    done

    echo "Applying BPEs..."
    for l in $src $tgt; do
        for f in train.$l valid.$l test.$l; do
            echo "Apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l < $tmp/$f > $prep/$f
        done
    done
    
    echo "Building vocabulary and binarizing data..." 
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --trainpref $prep/train \
        --validpref $prep/valid \
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
        --tgtdict data-bin/$CODE_SOURCE_DIR/dict.ru.txt \
        --destdir data-bin/$prep \
        --workers $N_THREADS
    cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "nonredundant" ]
then
    # Setting variables for the current option
    prep=$corpus/nonredundant
    tmp=$prep/tmp
    mkdir -p $tmp

    echo "Pre-processing data"
    for l in $src $tgt; do
        if [ $case = "lowercase" ]
        then
            echo "Text is being lowercased!"
            # replace _eos with newline, select last line of each block, lowercase
            cat $orig/$l\_train | \
            sed "s/ _eos /\n/g" | \
            awk 'NR % 4 == 0' | \
            python -c "import sys; print(sys.stdin.read().lower())" > $tmp/train.$l
            cat $orig/$l\_dev | \
            sed "s/ _eos /\n/g" | \
            awk 'NR % 4 == 0' | \
            python -c "import sys; print(sys.stdin.read().lower())" > $tmp/valid.$l
            cat $orig/$l\_test | \
            sed "s/ _eos /\n/g" | \
            awk 'NR % 4 == 0' | \
            python -c "import sys; print(sys.stdin.read().lower())" > $tmp/test.$l
        else
            # replace _eos with newline, select last line of each block
            cat $orig/$l\_train | sed "s/ _eos /\n/g" | awk 'NR % 4 == 0' > $tmp/train.$l
            cat $orig/$l\_dev | sed "s/ _eos /\n/g" | awk 'NR % 4 == 0' > $tmp/valid.$l
            cat $orig/$l\_test | sed "s/ _eos /\n/g" | awk 'NR % 4 == 0' > $tmp/test.$l
        fi
        # impute indices of headlines
        seq 1 200 1500000 > $prep/train.$lang.$l.heads
        seq 1 200 10000 > $prep/valid.$lang.$l.heads
        seq 1 200 10000 > $prep/test.$lang.$l.heads
    done

    echo "Applying BPEs..."
    for l in $src $tgt; do
        for f in train.$l valid.$l test.$l; do
            echo "Apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l < $tmp/$f > $prep/$f
        done
    done
    
    echo "Building vocabulary and binarizing data..." 
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --trainpref $prep/train \
        --validpref $prep/valid \
        --testpref $prep/test \
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
        --tgtdict data-bin/$CODE_SOURCE_DIR/dict.ru.txt \
        --destdir data-bin/$prep \
        --workers $N_THREADS
    cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "small10" ]
then
    # Setting variables for the current option
    N=150000 # =1500000/10
    prep=$corpus/small10
    tmp=$prep/tmp
    mkdir -p $tmp

    get_seeded_random()
    {
        seed="$1"
        openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
        </dev/zero 2>/dev/null
    }

    echo "Pre-processing data"
    for l in $src $tgt; do
        if [ $case = "lowercase" ]
        then
            echo "Text is being lowercased!"
            # subsample training corpus, 
            # add a blank line between blocks of sentences,
            # replace _eos with newline, lowercase
            cat $orig/$l\_train | \
            shuf --random-source=<(get_seeded_random 0) | \
            head -n $N | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/train.$l
            #
            cat $orig/$l\_dev | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/valid.$l
            #
            cat $orig/$l\_test | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/test.$l
        else
            # add a blank line between blocks of sentences, replace _eos with newline
            cat $orig/$l\_train | \
            shuf --random-source=<(get_seeded_random 0) | \
            head -n $N | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            head -n -1 > $tmp/train.$l
            #
            cat $orig/$l\_dev | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | head -n -1 > $tmp/valid.$l
            cat $orig/$l\_test | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | head -n -1 > $tmp/test.$l
        fi
        # retrieve indices of headlines
        python $HEADS $tmp/train.$l
        python $HEADS $tmp/valid.$l
        python $HEADS $tmp/test.$l
        mv $tmp/train.$l.heads $prep/train.$lang.$l.heads
        mv $tmp/valid.$l.heads $prep/valid.$lang.$l.heads
        mv $tmp/test.$l.heads $prep/test.$lang.$l.heads
    done

    echo "Applying BPEs..."
    for l in $src $tgt; do
        for f in train.$l valid.$l test.$l; do
            echo "Apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l < $tmp/$f > $prep/$f
        done
    done
    
    echo "Building vocabulary and binarizing data..." 
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --trainpref $prep/train \
        --validpref $prep/valid \
        --testpref $prep/test \
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
        --tgtdict data-bin/$CODE_SOURCE_DIR/dict.ru.txt \
        --destdir data-bin/$prep \
        --workers $N_THREADS
    cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "small10split" ]
then
    # Set variables for the current option
    N=150000 # =1500000/10
    prep=$corpus/small10split
    tmp=$prep/tmp
    mkdir -p $tmp

    get_seeded_random()
    {
        seed="$1"
        openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
        </dev/zero 2>/dev/null
    }

    echo "Pre-processing data"
    for l in $src $tgt; do
        if [ $case = "lowercase" ]
        then
            echo "Text is being lowercased!"
            # add a blank line between blocks of sentences, replace _eos with newline, lowercase
            cat $orig/$l\_train | \
            shuf --random-source=<(get_seeded_random 0) | \
            head -n $N | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/train.$l
            cat $orig/$l\_dev | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/valid.$l
        else
            # add a blank line between blocks of sentences, replace _eos with newline
            cat $orig/$l\_train | \
            shuf --random-source=<(get_seeded_random 0) | \
            head -n $N | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            head -n -1 > $tmp/train.$l
            #
            cat $orig/$l\_dev | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | head -n -1 > $tmp/valid.$l
        fi
    done

    echo "Splitting sentences..."
    for f in train valid ; do
        python ../../scripts/split_corpus_sentences.py \
        --src=$tmp/$f.$src --tgt=$tmp/$f.$tgt
        rm -f $tmp/$f.$src
        rm -f $tmp/$f.$tgt
        mv $tmp/$f.$src.split $tmp/$f.$src
        mv $tmp/$f.$tgt.split $tmp/$f.$tgt
    done

    echo "Retrieving documents' heads and delete empty lines..."
    for l in $src $tgt; do
        # retrieve indices of headlines
        python $HEADS $tmp/train.$l
        python $HEADS $tmp/valid.$l
        mv $tmp/train.$l.heads $prep/train.$lang.$l.heads
        mv $tmp/valid.$l.heads $prep/valid.$lang.$l.heads
    done

    echo "Applying BPEs..."
    for l in $src $tgt; do
        for f in train.$l valid.$l test.$l; do
            echo "Apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l < $tmp/$f > $prep/$f
        done
    done
    
    echo "Building vocabulary and binarizing data..." 
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --trainpref $prep/train \
        --validpref $prep/valid \
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
        --tgtdict data-bin/$CODE_SOURCE_DIR/dict.ru.txt \
        --destdir data-bin/$prep \
        --workers $N_THREADS
    cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "dummy" ]
then
    # Setting variables for the current option
    prep=$corpus/dummy
    tmp=$prep/tmp
    mkdir -p $tmp

    N_train=100
    N_valid=20
    N_test=20

    echo "Pre-processing data"
    for l in $src $tgt; do
        if [ $case = "lowercase" ]
        then
            echo "Text is being lowercased!"
            # subsample training corpus, 
            # add a blank line between blocks of sentences,
            # replace _eos with newline, lowercase
            cat $orig/$l\_train | \
            head -n $N_train | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/train.$l
            #
            cat $orig/$l\_dev | \
            head -n $N_valid | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/valid.$l
            #
            cat $orig/$l\_test | \
            head -n $N_test | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            python -c "import sys; print(sys.stdin.read().lower())" | \
            head -n -2 > $tmp/test.$l
        else
            # add a blank line between blocks of sentences, replace _eos with newline
            cat $orig/$l\_train | \
            head -n $N_train | \
            awk '{print $0,"\n"}' | \
            sed "s/ _eos /\n/g" | \
            head -n -1 > $tmp/train.$l
            #
            cat $orig/$l\_dev | head -n $N_valid | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | head -n -1 > $tmp/valid.$l
            cat $orig/$l\_test | head -n $N_test | awk '{print $0,"\n"}' | sed "s/ _eos /\n/g" | head -n -1 > $tmp/test.$l
        fi
        # retrieve indices of headlines
        python $HEADS $tmp/train.$l
        python $HEADS $tmp/valid.$l
        python $HEADS $tmp/test.$l
        mv $tmp/train.$l.heads $prep/train.$lang.$l.heads
        mv $tmp/valid.$l.heads $prep/valid.$lang.$l.heads
        mv $tmp/test.$l.heads $prep/test.$lang.$l.heads
    done

    # eliminate last head  which is actually end of file
    sed -i '$ d' $prep/*.heads

    echo "Applying BPEs..."
    for l in $src $tgt; do
        for f in train.$l valid.$l test.$l; do
            echo "Apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE.$l < $tmp/$f > $prep/$f
        done
    done
    
    echo "Building vocabulary and binarizing data..." 
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --trainpref $prep/train \
        --validpref $prep/valid \
        --testpref $prep/test \
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
        --tgtdict data-bin/$CODE_SOURCE_DIR/dict.ru.txt \
        --destdir data-bin/$prep \
        --workers $N_THREADS
    cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "alldata" ]
then

    # Setting variables
    if [ $case = "lowercase" ]
    then
        corpus=voita_opensubs/context_aware
        caware=voita_opensubs/context_aware/standard
        cagnostic=voita_opensubs/context_agnostic/standard
    else
        corpus=$case\_voita_opensubs/context_aware
        caware=$case\_voita_opensubs/context_aware/standard
        cagnostic=$case\_voita_opensubs/context_agnostic/standard
    fi

    # Setting variables for the current option
    prep=$corpus/alldata
    tmp=$prep/tmp
    mkdir -p $tmp

    echo "Pre-processing data..."
    for l in $src $tgt
    do  
        cat $caware/train.$l > $prep/train.$l
        cat $cagnostic/train.$l >> $prep/train.$l
    done 
    cp $caware/*heads $prep/
    cp $caware/test* $prep/
    cp $caware/valid* $prep/

    echo "Calculating heads..."
    for l in $src $tgt
    do  
        start=$(tail -1 $prep/train.$lang.$l.heads)
        lenf=$(cat $cagnostic/train.$l | wc -l)
        end=$((start + 3 + lenf))
        seq $((start + 4)) $((end)) >> $prep/train.$lang.$l.heads
    done
    
    echo "Building vocabulary and binarizing data..." 
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --trainpref $prep/train \
        --validpref $prep/valid \
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
        --tgtdict data-bin/$CODE_SOURCE_DIR/dict.ru.txt \
        --destdir data-bin/$prep \
        --workers $N_THREADS
    cp $prep/*.heads data-bin/$prep/
###############################################################################
else
    echo "Argument is not valid."
fi