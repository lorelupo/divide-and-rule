#!/bin/bash

# Move to the data directory corresponding to the right language pair
src=en
tgt=fr
lang=$src-$tgt
data=$HOME/dev/fairseq/data/$lang
mkdir -p $data
cd $data

# Set variables
prep=nei/standard
tmp=$prep/tmp
mkdir -p $tmp
CODE_SOURCE_DIR=wmt14/standard
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

# Preprocessing IWSLT w/ -a & w/o LC ########################################
orig=/data2/video/getalp/mt-data/en-fr/iwslt17

# echo "Pre-processing train data..."
# for l in $src $tgt; do
#     # Select training files
#     f=train.tags.$lang.$l
#     tok=train.tags.$lang.tok.$l
#     # Remove lines containing url, talkid and keywords (grep -v).
#     # Remove special tokens with sed -e.
#     # Then tokenize (insert spaces between words and punctuation) with moses.
#     cat $orig/$f | \
#     grep -v '<doc ' | \
#     grep -v '</doc>' | \
#     grep -v '<url>' | \
#     grep -v '</translator>' | \
#     grep -v '</reviewer>' | \
#     grep -v '</speaker>' | \
#     grep -v '</keywords>' | \
#     sed -e 's/<talkid>.*<\/talkid>//g' | \
#     sed -e 's/<title>//g' | \
#     sed -e 's/<\/title>//g' | \
#     sed -e 's/<description>//g' | \
#     sed -e 's/<\/description>//g' | \
#     perl $TOKENIZER -threads 8 -a -l $l > $tmp/$tok
#     echo ""
# done

# # Clean training files from long sentences (>250 tok),
# # [not empty sentences] and sentences that highly mismatch in length (ratio)
# perl $CLEAN_IWSLT -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang 0 250

# echo "Pre-processing valid/test data..."
# for l in $src $tgt; do
#     for o in `ls $orig/IWSLT17.TED*.$l.xml`; do
#     fname=${o##*/}
#     f=$tmp/${fname%.*}
#     echo $o $f
#     cat $o | \
#     sed '/<doc \s*/i <seg id="0">' | \
#     grep '<seg id' | \
#     sed -e 's/<seg id="[0-9]*">\s*//g' | \
#     sed -e 's/\s*<\/seg>\s*//g' | \
#     sed -e "s/\’/\'/g" | \
#     perl $TOKENIZER -threads $N_THREADS -a -l $l > $f
#     echo ""
#     done
# done

# echo "creating train, valid, test data"
# for l in $src $tgt; do
#     # textual datasets
#     cat $tmp/train.tags.$lang.$l          > $tmp/iwslt_train.$l
#     cat $tmp/IWSLT17.TED.tst2011.$lang.$l \
#         $tmp/IWSLT17.TED.tst2012.$lang.$l \
#         $tmp/IWSLT17.TED.tst2013.$lang.$l \
#         $tmp/IWSLT17.TED.tst2014.$lang.$l > $tmp/iwslt_valid.$l  
#     cat $tmp/IWSLT17.TED.tst2015.$lang.$l > $tmp/iwslt_test.$l
# done

# # Preprocess News and Europarl ################################################

# orig=/data2/video/getalp/mt-data/en-fr/wmt14/

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
#         cat $tmp/$f.$l >> $tmp/ne_train.tags.$lang.tok.$l
#     done
# done

# # Clean both trainin files from long sentences (>175), empty sentences
# # and sentences that highly mismatch in length (ratio)
# perl $CLEAN_NE -ratio 1.5 $tmp/ne_train.tags.$lang.tok $src $tgt $tmp/ne_train 1 250

# # Form train, valid and test  #################################################

# # for train, we merge iwslt, news and europarl training sets
# # for valid and test we only keep iwslt's valid and testsets

# echo "merging all datasets..."
# for l in $src $tgt; do
#     for d in train valid test; do
#         echo "merge $d.$l files"
#         f=$tmp/$d.$l
#         rm -rf $f
#         cat $tmp/iwslt_$d.$l >> $f
#         if [ $d = "train" ]; then
#             echo $'' >> $f
#             cat $tmp/ne_$d.$l >> $f
#         fi
#     done
# done

# echo "generating heads files..."
# for l in $src $tgt; do
#     # retrieve indices of headlines
#     python $HEADS $tmp/train.$l --fill-size=200
#     python $HEADS $tmp/valid.$l
#     python $HEADS $tmp/test.$l
#     mv $tmp/train.$l.heads $prep/train.$lang.$l.heads 
#     mv $tmp/valid.$l.heads $prep/valid.$lang.$l.heads
#     mv $tmp/test.$l.heads $prep/test.$lang.$l.heads
# done

# BPE and binarizaiton ########################################################

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
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
    --validpref $prep/valid \
    --testpref $prep/test \
    --srcdict data-bin/$CODE_SOURCE_DIR/dict.en.txt \
    --joined-dictionary \
    --destdir data-bin/$prep \
    --workers $N_THREADS
cp $prep/*.heads data-bin/$prep/
