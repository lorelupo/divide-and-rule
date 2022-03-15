#!/bin/bash

# Move to the data directory corresponding to the right language pair
src=en
tgt=fr
lang=$src-$tgt
DATA=$HOME/dev/fairseq/data/$lang
mkdir -p $DATA
cd $DATA

# Set variables
prep=wmt14/standard
tmp=$prep/tmp
orig=/data2/video/getalp/mt-data/$lang/wmt14

mkdir -p $tmp $prep $orig

N_THREADS=6
TOOLS=../../tools
SCRIPTS=$TOOLS/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

BPE_CODE=$prep/code
BPEROOT=$TOOLS/subword-nmt/subword_nmt
BPE_TOKENS=32000

# Setting utils ################################################################

echo 'Looking for Moses github repository (for tokenization scripts)...'
DIR="$TOOLS/mosesdecoder"
if [ -d "$DIR" ]; then
echo "Moses repo was already cloned here."
else
echo 'Cloning Moses github repository.'
git clone https://github.com/moses-smt/mosesdecoder.git $TOOLS/
fi


echo 'Looking for Subword NMT repository (for BPE pre-processing)...'
DIR="$TOOLS/subword-nmt"
if [ -d "$DIR" ]; then
echo "Subword NMT repo was already cloned here."
else
echo 'Cloning Subword NMT repository.'
git clone https://github.com/rsennrich/subword-nmt.git $TOOLS/
fi

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

if [ -d "$orig" ]; then
echo "Retrieving dataset from $orig"
else
echo "Dataset is not available at $orig"
exit 1
fi

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt13/training-parallel-un.tgz"
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://statmt.org/wmt10/training-giga-fren.tar"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-un.tgz"
    "training-parallel-nc-v9.tgz"
    "training-giga-fren.tar"
    "test-full.tgz"
)

# Downloading data #############################################################

# cd $orig

# for ((i=0;i<${#URLS[@]};++i)); do
#     file=${FILES[i]}
#     if [ -f $file ]; then
#         echo "$file already exists, skipping download"
#     else
#         url=${URLS[i]}
#         wget "$url"
#         if [ -f $file ]; then
#             echo "$url successfully downloaded."
#         else
#             echo "$url not successfully downloaded."
#             exit -1
#         fi
#         if [ ${file: -4} == ".tgz" ]; then
#             tar zxvf $file
#         elif [ ${file: -4} == ".tar" ]; then
#             tar xvf $file
#         fi
#     fi
# done

# gunzip giga-fren.release2.fixed.*.gz
# cd ..

# Preprocessing ################################################################

# CORPORA=(
#     "training/europarl-v7.fr-en"
#     "commoncrawl.fr-en"
#     "un/undoc.2000.fr-en"
#     "training/news-commentary-v9.fr-en"
#     "giga-fren.release2.fixed"
# )

# CORPORA=(
#     "europarl-v7.fr-en"
#     "commoncrawl.fr-en"
#     "undoc.2000.fr-en"
#     "news-commentary-v9.fr-en"
#     "giga-fren.release2.fixed"
# )


# echo "pre-processing train data..."
# for l in $src $tgt; do
#     for f in "${CORPORA[@]}"; do
#         cat $orig/$f.$l | \
#             perl $NORM_PUNC $l | \
#             perl $REM_NON_PRINT_CHAR | \
#             perl $TOKENIZER -threads $N_THREADS -a -l $l > $tmp/$f.$l
#             echo "$f's number of lines:"
#             wc -l $tmp/$f.$l
#             echo ""
#     done
# done

# echo "collecting the preprocessed train data in a unique train file..."
# for l in $src $tgt; do
#     rm $tmp/train.tags.$lang.tok.$l
#     for f in "${CORPORA[@]}"; do
#         cat $tmp/$f.$l >> $tmp/train.tags.$lang.tok.$l
#     done
# done

# echo "pre-processing test data..."
# for l in $src $tgt; do
#     if [ "$l" == "$src" ]; then
#         t="src"
#     else
#         t="ref"
#     fi
#     grep '<seg id' $orig/test_full/newstest2014-fren-$t.$l.sgm | \
#         sed -e 's/<seg id="[0-9]*">\s*//g' | \
#         sed -e 's/\s*<\/seg>\s*//g' | \
#         sed -e "s/\’/\'/g" | \
#     perl $TOKENIZER -threads $N_THREADS -a -l $l > $tmp/test.$l
#     echo ""
# done

# # Splitting dataset ############################################################

# echo "splitting train and valid..."
# for l in $src $tgt; do
#     awk '{if (NR%1333 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
#     awk '{if (NR%1333 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
# done

# # Learning and applying BPE ###################################################

# TRAIN=$tmp/train.fr-en

# rm -f $TRAIN
# echo "building dataset for learning bpe..."
# for l in $src $tgt; do
#     cat $tmp/train.$l >> $TRAIN
# done

# echo "learn_bpe.py on ${TRAIN}..."
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

# for L in $src $tgt; do
#     for f in train.$L valid.$L test.$L; do
#         echo "apply_bpe.py to ${f}..."
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
#     done
# done

# # Clean both training and valid docs from long sentences (>250), empty sentences
# # and sentences that highly mismatch in length (ratio)
# perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
# perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

# for L in $src $tgt; do
#     cp $tmp/bpe.test.$L $prep/test.$L
# done

# Build vocabularies and binarize training data ###############################
rm -rf data-bin/$prep
echo "Building vocabulary and binarizing data..."
fairseq-preprocess \
    --source-lang $src \
    --target-lang $tgt \
    --trainpref $prep/train \
    --validpref $prep/valid \
    --testpref $prep/test \
    --joined-dictionary \
    --destdir data-bin/$prep \
    --workers $N_THREADS
    # --nwordssrc $BPE_TOKENS --nwordstgt $BPE_TOKENS \
