#!/bin/bash

# Move to ./data directory
cd data

# Set variables

src=en
tgt=fr
lang=$src-$tgt
prep=ne/standard
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

WMT_DIR=wmt14/standard
IWSLT_DIR=data-bin/iwslt17/standard

# Preprocess News and Europarl ################################################

orig=/data2/video/getalp/mt_data/wmt14/fr_en

# CORPORA=(
#     "europarl-v7.fr-en"
#     "news-commentary-v9.fr-en"
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

# # Clean both trainin files from long sentences (>175), empty sentences
# # and sentences that highly mismatch in length (ratio)
# perl $CLEAN_NE -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train 1 250

# Generate heads ##############################################################

# retrieve avg docs length in IWSLT's training set
n_lines=$(wc -l iwslt17/standard/train.en | cut -c 1-7)
n_docs=$(wc -l iwslt17/standard/train.en-fr.en.heads | cut -c 1-5)
avg_length=$((n_lines / n_docs))
echo "Setting documents' length for NE to $avg_length..."

echo "generating heads files..."
for l in $src $tgt; do
    # retrieve indices of headlines
    python $HEADS $tmp/train.$l --fill-size=$avg_length
    mv $tmp/train.$l.heads $prep/train.$lang.$l.heads 
done

# # BPE and binarizaiton ########################################################
# BPE_CODE=$WMT_DIR/code
# BPEROOT=subword-nmt/subword_nmt
# BPE_TOKENS=32000

# for L in $src $tgt; do
#     for f in train.$L; do
#         echo "apply_bpe.py to ${f}..."
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
#     done
# done

# # Build vocabularies and binarize training data ###############################
# rm -rf data-bin/$prep
# echo "Binarizing data..."
# fairseq-preprocess \
#     --source-lang $src \
#     --target-lang $tgt \
#     --trainpref $prep/train \
#     --srcdict data-bin/$WMT_DIR/dict.en.txt \
#     --joined-dictionary \
#     --destdir data-bin/$prep \
#     --workers $N_THREADS

# cp $prep/*.heads data-bin/$prep/

# # Copying IWSLT dev and test set to data-bin for convenience ##################

# cp IWSLT_DIR/test* data-bin/$prep/
# cp IWSLT_DIR/valid* data-bin/$prep/