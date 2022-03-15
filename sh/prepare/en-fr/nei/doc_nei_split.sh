#!/bin/bash

# Move to ./data directory
cd data

# Set variables

src=en
tgt=fr
lang=$src-$tgt
prep=nei/split
tmp=$prep/tmp

mkdir -p $tmp

N_THREADS=8
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
HEADS=../scripts/retrieve_doc_heads.py

CLEAN_IWSLT=../scripts/clean-corpus-n-leaving-blanks.perl
CLEAN_NE=$SCRIPTS/training/clean-corpus-n.perl

WMT=wmt14/standard
NEI=nei/standard
IWSLT_SPLIT=iwslt17/split

echo "split sentences..."
for f in train ; do
    python ../scripts/split_corpus_sentences.py --src=$NEI/tmp/$f.$src --tgt=$NEI/tmp/$f.$tgt
    mv $NEI/tmp/$f.$src.split $tmp/$f.$src
    mv $NEI/tmp/$f.$tgt.split $tmp/$f.$tgt
done

echo "generating heads files..."
# retrieve avg docs length in IWSLT's training set
n_lines=$(wc -l iwslt17/standard/train.en | cut -c 1-7)
n_docs=$(wc -l iwslt17/standard/train.en-fr.en.heads | cut -c 1-5)
avg_length=$((2* n_lines / n_docs))
echo "\\t Setting documents' length for NE-split to twice the average document length of IWSLT, i.e. $avg_length"

for l in $src $tgt; do
    # retrieve indices of headlines
    python $HEADS $tmp/train.$l --fill-size=$avg_length
    mv $tmp/train.$l.heads $prep/train.$lang.$l.heads 
done

# BPE and binarizaiton ########################################################
BPE_CODE=$WMT/code
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=32000

for L in $src $tgt; do
    for f in train.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done

# Build vocabularies and binarize training data ###############################
rm -rf data-bin/$prep
echo "Binarizing data..."
fairseq-preprocess \
    --source-lang $src \
    --target-lang $tgt \
    --trainpref $prep/train \
    --srcdict data-bin/$WMT/dict.en.txt \
    --joined-dictionary \
    --destdir data-bin/$prep \
    --workers $N_THREADS

cp $prep/*.heads data-bin/$prep/
cp data-bin/$IWSLT_SPLIT/valid* data-bin/$prep/
cp data-bin/$IWSLT_SPLIT/test* data-bin/$prep/