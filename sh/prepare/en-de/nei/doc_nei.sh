#!/bin/bash
#
# Possible arguments: standard, split
#
# bash sh/prepare/en-de/nei/doc_nei.sh standard
# bash sh/prepare/en-de/nei/doc_nei.sh split

# Move to the data directory corresponding to the right language pair
src=en
tgt=de
lang=$src-$tgt
data=$HOME/dev/fairseq/data/$lang
mkdir -p $data
cd $data

# Setting variables
corpus=nei
iwslt_orig=iwslt17/standard
wmt_orig=/video/getalp/mt-data/$lang/wmt17
CODE_SOURCE_DIR=wmt17/standard
BPE_CODE=$CODE_SOURCE_DIR/code
BPE_TOKENS=32000
N_THREADS=8

# Standard variables
TOOLS=../../tools
SCRIPTS=$TOOLS/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=$TOOLS/subword-nmt/subword_nmt

CLEAN_IWSLT=../../scripts/clean-corpus-n-leaving-blanks.perl
CLEAN_NE=$SCRIPTS/training/clean-corpus-n.perl

HEADS=../../scripts/retrieve_doc_heads.py

if [ $1 = "standard" ]
then
    # Setting variables for the current option
    prep=$corpus/standard
    tmp=$prep/tmp
    mkdir -p $tmp

    echo "retrieving IWSLT's train, valid and test data"
    iwslt_tmp=$iwslt_orig/tmp
    for l in $src $tgt; do
    # textual datasets
    cat $iwslt_tmp/train.tags.$lang.$l          > $tmp/iwslt_train.$l
    cat $iwslt_tmp/IWSLT17.TED.tst2011.$lang.$l \
        $iwslt_tmp/IWSLT17.TED.tst2012.$lang.$l \
        $iwslt_tmp/IWSLT17.TED.tst2013.$lang.$l \
        $iwslt_tmp/IWSLT17.TED.tst2014.$lang.$l > $tmp/iwslt_valid.$l
    cat $iwslt_tmp/IWSLT17.TED.tst2015.$lang.$l > $tmp/iwslt_test.$l 
    done

    CORPORA=(
        "europarl-v7.de-en"
        "news-commentary-v12.de-en"
    )

    echo "pre-processing News and Europarl train data..."
    for l in $src $tgt; do
        for f in "${CORPORA[@]}"; do
            cat $wmt_orig/$f.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads $N_THREADS -a -l $l > $tmp/$f.$l
                echo "$f's number of lines:"
                wc -l $tmp/$f.$l
                echo ""
        done
    done

    echo "merging News and Europarl train data in a unique train file..."
    for l in $src $tgt; do
        rm $tmp/ne_train.tags.$lang.tok.$l
        for f in "${CORPORA[@]}"; do
            cat $tmp/$f.$l >> $tmp/ne_train.tags.$lang.tok.$l
        done
    done

    # Clean both trainin files from long sentences (>250tok), empty sentences
    # and sentences that highly mismatch in length (ratio)
    perl $CLEAN_NE -ratio 1.5 $tmp/ne_train.tags.$lang.tok $src $tgt $tmp/ne_train 1 250

    echo "merging all datasets..."
    # for train, we merge iwslt, news and europarl training sets
    # for valid and test we only keep iwslt's valid and testsets
    for l in $src $tgt; do
        for d in train valid test; do
            echo "merge $d.$l files"
            f=$tmp/$d.$l
            rm -rf $f
            cat $tmp/iwslt_$d.$l >> $f
            if [ $d = "train" ]; then
                echo $'' >> $f
                cat $tmp/ne_$d.$l >> $f
            fi
        done
    done

    echo "generating heads files..."
    # retrieve avg docs length in IWSLT's training set
    n_lines=$(wc -l $iwslt_orig/train.$src | cut -c 1-7)
    n_docs=$(wc -l $iwslt_orig/train.$lang.$src.heads | cut -c 1-5)
    avg_length=$((n_lines / n_docs))
    for l in $src $tgt; do
        # retrieve indices of headlines
        python $HEADS $tmp/train.$l --fill-size=$avg_length
        python $HEADS $tmp/valid.$l
        python $HEADS $tmp/test.$l
        mv $tmp/train.$l.heads $prep/train.$lang.$l.heads 
        mv $tmp/valid.$l.heads $prep/valid.$lang.$l.heads
        mv $tmp/test.$l.heads $prep/test.$lang.$l.heads
    done

    echo "Applying BPE..."
    for L in $src $tgt; do
        for f in train.$L valid.$L test.$L; do
            echo "apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
        done
    done

    echo "Binarizing data..."
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --trainpref $prep/train \
        --validpref $prep/valid \
        --testpref $prep/test \
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
        --joined-dictionary \
        --destdir data-bin/$prep \
        --workers $N_THREADS
    cp $prep/*.heads data-bin/$prep/

elif [ $1 = "split" ]
then
    # The split option will work only if the standard dataset already exists and
    # has been processed with the code above (otion "standard")
    standard_tmp=$corpus/standard/tmp
    if [ ! -d "$standard_tmp" ]; then
    echo "$standard_tmp not found."
    echo "Before using the 'split' option, launch this script with 'standard'"
    exit 1
    fi

    # Set variables for the current option
    prep=$corpus/split
    tmp=$prep/tmp
    mkdir -p $tmp

    echo "Merging all datasets..."
    # for train, we merge iwslt, news and europarl training sets
    # for valid and test we only keep iwslt's valid and testsets
    for l in $src $tgt; do
        for d in train valid; do
            echo "merge $d.$l files"
            f=$tmp/$d.$l
            rm -rf $f
            cat $standard_tmp/iwslt_$d.$l >> $f
            if [ $d = "train" ]; then
                echo $'' >> $f
                cat $standard_tmp/ne_$d.$l >> $f
            fi
        done
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

    echo "Generating heads files..."
    # retrieve avg docs length in IWSLT's training set
    n_lines=$(wc -l $iwslt_orig/train.$src | cut -c 1-7)
    n_docs=$(wc -l $iwslt_orig/train.$lang.$src.heads | cut -c 1-5)
    avg_length=$((2* n_lines / n_docs))
    for l in $src $tgt; do
        # retrieve indices of headlines
        python $HEADS $tmp/train.$l --fill-size=$avg_length
        python $HEADS $tmp/valid.$l
        mv $tmp/train.$l.heads $prep/train.$lang.$l.heads 
        mv $tmp/valid.$l.heads $prep/valid.$lang.$l.heads
    done

    echo "Applying BPE..."
    for L in $src $tgt; do
        for f in train.$L valid.$L; do
            echo "apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
        done
    done

    echo "Binarizing data..."
    rm -rf data-bin/$prep
    fairseq-preprocess \
        --source-lang $src \
        --target-lang $tgt \
        --trainpref $prep/train \
        --validpref $prep/valid \
        --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
        --joined-dictionary \
        --destdir data-bin/$prep \
        --workers $N_THREADS
    cp $prep/*.heads data-bin/$prep/

fi