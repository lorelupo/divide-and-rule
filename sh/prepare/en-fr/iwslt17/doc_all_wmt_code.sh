#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
# IWSLT17 train data, test  sets  2011-2014  as  dev  sets,  and 2015 as test set

# Move to ./data directory
cd data

# Setting variables
src=en
tgt=fr
lang=$src-$tgt
orig=iwslt17.original.$lang
prep=iwslt17/standard
tmp=$prep/tmp
mkdir -p $tmp

N_THREADS=8
GZ=$lang.tgz
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
# LC=$SCRIPTS/tokenizer/lowercase.perl

CLEAN=../scripts/clean-corpus-n-leaving-blanks.perl

URL="https://wit3.fbk.eu/archive/2017-01-trnted//texts/en/fr/en-fr.tgz"
HEADS=../scripts/retrieve_doc_heads.py

SOURCE_DIR=wmt14/standard

# Setting utils ###############################################################

echo 'Looking for Moses github repository (for tokenization scripts)...'
DIR="./mosesdecoder"
if [ -d "$DIR" ]; then
  echo "Moses repo was already cloned here."
else
  echo 'Cloning Moses github repository.'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi


echo 'Looking for Subword NMT repository (for BPE pre-processing)...'
DIR="./subword-nmt"
if [ -d "$DIR" ]; then
  echo "Subword NMT repo was already cloned here."
else
  echo 'Cloning Subword NMT repository.'
  git clone https://github.com/rsennrich/subword-nmt.git
fi

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

# Downloading data ############################################################

if [ -d "$orig" ]; then
  echo "Data are already available in local dir."
else
  mkdir -p $orig
  echo "Downloading data from ${URL}..."
  cd $orig
  wget "$URL"
  if [ -f $GZ ]; then
    echo "Data successfully downloaded."
  else
    echo "Data not successfully downloaded."
    exit
  fi
  tar zxvf $GZ
  cd ..
fi

# Preprocessing ###############################################################

echo "Pre-processing train data..."
for l in $src $tgt; do
    # Select training files
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l
    # Remove lines containing url, talkid and keywords (grep -v).
    # Remove special tokens with sed -e.
    # Then tokenize (insert spaces between words and punctuation) with moses.
    cat $orig/$lang/$f | \
    grep -v '<doc ' | \
    grep -v '</doc>' | \
    grep -v '<url>' | \
    grep -v '</translator>' | \
    grep -v '</reviewer>' | \
    grep -v '</speaker>' | \
    grep -v '</keywords>' | \
    sed -e 's/<talkid>.*<\/talkid>//g' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done

# Clean training files from long sentences (100 sentences longer then 175 tok),
# [not empty sentences] and sentences that highly mismatch in length (ratio)
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang 0 175

# echo "Lowercase everything"
# for l in $src $tgt; do
#     perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
# done

echo "Pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT17.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    cat $o | \
    sed '/<doc \s*/i <seg id="0">' | \
    grep '<seg id' | \
    sed -e 's/<seg id="[0-9]*">\s*//g' | \
    sed -e 's/\s*<\/seg>\s*//g' | \
    sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l > $f
    # perl $LC > $f
    echo ""
    done
done

# Splitting dataset ###########################################################

echo "creating train, valid, test data"
for l in $src $tgt; do
    # textual datasets
    cat $tmp/train.tags.$lang.$l          > $tmp/train.$l
    cat $tmp/IWSLT17.TED.tst2011.$lang.$l \
        $tmp/IWSLT17.TED.tst2012.$lang.$l \
        $tmp/IWSLT17.TED.tst2013.$lang.$l \
        $tmp/IWSLT17.TED.tst2014.$lang.$l > $tmp/valid.$l  
    cat $tmp/IWSLT17.TED.tst2015.$lang.$l > $tmp/test.$l
    # retrieve indices of headlines
    python $HEADS $tmp/train.$l
    python $HEADS $tmp/valid.$l
    python $HEADS $tmp/test.$l
    mv $tmp/train.$l.heads $prep/train.$lang.$l.heads
    mv $tmp/valid.$l.heads $prep/valid.$lang.$l.heads
    mv $tmp/test.$l.heads $prep/test.$lang.$l.heads
done

# Print stats before BPE ######################################################

for l in $src $tgt; do
    echo "Stats for train.$l (tokenized and cleaned):"
    words=$(wc -w $tmp/train.$l | awk '{print $1;}')
    sents=$(wc -l $tmp/train.$l | awk '{print $1;}')
    printf "%10d words \n" $words
    printf "%10d sentences \n" $sents
    printf "%10s wps \n" $(echo "scale=2 ; $words / $sents" | bc)
    echo
done

# Learning and applying BPE ###################################################

BPE_CODE=$SOURCE_DIR/code
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=32000

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "Apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done

# Build vocabularies and binarize training data ###############################

rm -rf data-bin/$prep
echo "Building vocabulary and binarizing data..." 
fairseq-preprocess \
    --source-lang $src \
    --target-lang $tgt \
    --trainpref $prep/train \
    --validpref $prep/valid \
    --testpref $prep/test \
    --srcdict data-bin/$SOURCE_DIR/dict.en.txt \
    --joined-dictionary \
    --destdir data-bin/$prep \
    --workers $N_THREADS

cp $prep/*.heads data-bin/$prep/