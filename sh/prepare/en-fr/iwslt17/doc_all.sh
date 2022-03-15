#!/usr/bin/env bash
#
# IWSLT17 train data, test  sets  2011-2014  as  dev  sets,  and 2015 as test set
#
# Possible usages:
#
# bash sh/prepare/en-fr/iwslt17/doc_all.sh standard
# bash sh/prepare/en-fr/iwslt17/doc_all.sh shuffled
# bash sh/prepare/en-fr/iwslt17/doc_all.sh split
# bash sh/prepare/en-fr/iwslt17/doc_all.sh 3split 30
# bash sh/prepare/en-fr/iwslt17/doc_all.sh zero-shot-split
# bash sh/prepare/en-fr/iwslt17/doc_all.sh aligned-split
# bash sh/prepare/en-fr/iwslt17/doc_all.sh synt-split
# bash sh/prepare/en-fr/iwslt17/doc_all.sh syntall-split

# Move to the data directory corresponding to the right language pair
src=en
tgt=fr
lang=$src-$tgt
DATA=$HOME/dev/fairseq/data/$lang
mkdir -p $DATA
cd $DATA

# Setting variables
orig=/video/getalp/mt-data/$lang/iwslt17
corpus=iwslt17
CODE_SOURCE_DIR=wmt14/standard
BPE_CODE=$CODE_SOURCE_DIR/code
BPE_TOKENS=32000
N_THREADS=8

# Standard variables
TOOLS=../../tools
SCRIPTS=$TOOLS/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
BPEROOT=$TOOLS/subword-nmt/subword_nmt

CLEAN=../../scripts/clean-corpus-n-leaving-blanks.perl
HEADS=../../scripts/retrieve_doc_heads.py


if [ $1 = "standard" ]
then
  # Setting variables #########################################################
  prep=$corpus/standard
  tmp=$prep/tmp
  mkdir -p $tmp

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

  if [ -d "$orig" ]; then
    echo "Retrieving dataset from $orig"
  else
    echo "Dataset is not available at $orig"
    exit 1
  fi

  echo "Pre-processing train data..."
  for l in $src $tgt; do
      # Select training files
      f=train.tags.$lang.$l
      tok=train.tags.$lang.tok.$l
      # Remove lines containing url, talkid and keywords (grep -v).
      # Remove special tokens with sed -e.
      # Then tokenize (insert spaces between words and punctuation) with moses.
      cat $orig/$f | \
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

  # Clean training files from long sentences (100 sents longer then 175 tok),
  # [not empty sentences] and sentences that highly mismatch in length (ratio)
  perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang 0 175

  echo "Pre-processing valid/test data..."
  for l in $src $tgt; do
      for o in `ls $orig/IWSLT17.TED*.$l.xml`; do
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
      echo ""
      done
  done

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

  echo "Printing stats train stats..."
  for l in $src $tgt; do
      echo "Stats for train.$l (tokenized and cleaned):"
      words=$(wc -w $tmp/train.$l | awk '{print $1;}')
      sents=$(wc -l $tmp/train.$l | awk '{print $1;}')
      printf "%10d words \n" $words
      printf "%10d sentences \n" $sents
      printf "%10s wps \n" $(echo "scale=2 ; $words / $sents" | bc)
      echo
  done

  echo "Applying BPEs..."
  for L in $src $tgt; do
      for f in train.$L valid.$L test.$L; do
          echo "Apply_bpe.py to ${f}..."
          python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
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
      --joined-dictionary \
      --destdir data-bin/$prep \
      --workers $N_THREADS
  cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "shuffled" ]
then
  # This option will work only if the dataset already exists and
  # has been processed with the code above
  prep=$corpus/test_shuffled
  mkdir -p $prep

  echo "Shuffling testset...."
  get_seeded_random()
  {
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
      </dev/zero 2>/dev/null
  }
  for L in $src $tgt; do
    shuf --random-source=<(get_seeded_random 0) < $corpus/standard/test.$L > $prep/test.$L
  done
  
  echo "Binarizing data..."
  rm -rf data-bin/$prep
  fairseq-preprocess \
      --source-lang $src \
      --target-lang $tgt \
      --testpref $prep/test \
      --srcdict data-bin/$corpus/standard/dict.en.txt \
      --joined-dictionary \
      --destdir data-bin/$prep \
      --workers $N_THREADS
  cp data-bin/$corpus/standard/test.*.heads data-bin/$prep/
###############################################################################
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

  echo "Retrieving train and valid data..."
  for l in $src $tgt; do
      cat $standard_tmp/train.tags.$lang.$l          > $tmp/train.$l
      cat $standard_tmp/IWSLT17.TED.tst2011.$lang.$l \
          $standard_tmp/IWSLT17.TED.tst2012.$lang.$l \
          $standard_tmp/IWSLT17.TED.tst2013.$lang.$l \
          $standard_tmp/IWSLT17.TED.tst2014.$lang.$l > $tmp/valid.$l  
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
  for L in $src $tgt; do
      for f in train.$L valid.$L; do
          echo "Apply_bpe.py to ${f}..."
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
      --srcdict data-bin/$corpus/standard/dict.en.txt \
      --joined-dictionary \
      --destdir data-bin/$prep \
      --workers $N_THREADS
  cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "3split" ]
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
  prep=$corpus/3split_7_$2
  tmp=$prep/tmp
  mkdir -p $tmp

  echo "Retrieving train and valid data..."
  for l in $src $tgt; do
      cat $standard_tmp/train.tags.$lang.$l          > $tmp/train.$l
      cat $standard_tmp/IWSLT17.TED.tst2011.$lang.$l \
          $standard_tmp/IWSLT17.TED.tst2012.$lang.$l \
          $standard_tmp/IWSLT17.TED.tst2013.$lang.$l \
          $standard_tmp/IWSLT17.TED.tst2014.$lang.$l > $tmp/valid.$l  
  done

  echo "Splitting sentences..."
  for f in train valid ; do
      python ../../scripts/split_corpus_sentences.py \
        --src=$tmp/$f.$src --tgt=$tmp/$f.$tgt --min-length 7 $2
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
  for L in $src $tgt; do
      for f in train.$L valid.$L; do
          echo "Apply_bpe.py to ${f}..."
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
      --srcdict data-bin/$corpus/standard/dict.en.txt \
      --joined-dictionary \
      --destdir data-bin/$prep \
      --workers $N_THREADS
  cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "zero-shot-split" ]
then
  # The zero-shot-split option will work only if the standard dataset already
  # exists and has been processed with the code above (otion "standard")
  standard_tmp=$corpus/standard/tmp
  if [ ! -d "$standard_tmp" ]; then
    echo "$standard_tmp not found."
    echo "Before using the 'zero-shot-split' option, launch this script with 'standard'"
    exit 1
  fi

  # Set variables for the current option
  prep=$corpus/zero_shot_split
  tmp=$prep/tmp
  mkdir -p $tmp

  echo "Retrieving train and valid data..."
  for l in $src $tgt; do
      cat $standard_tmp/train.tags.$lang.$l          > $tmp/train.$l
      cat $standard_tmp/IWSLT17.TED.tst2011.$lang.$l \
          $standard_tmp/IWSLT17.TED.tst2012.$lang.$l \
          $standard_tmp/IWSLT17.TED.tst2013.$lang.$l \
          $standard_tmp/IWSLT17.TED.tst2014.$lang.$l > $tmp/valid.$l  
  done

  echo "Splitting sentences..."
  for f in train valid ; do
      python ../../scripts/split_corpus_sentences.py \
        --src=$tmp/$f.$src --tgt=$tmp/$f.$tgt --remove-shorter
      rm -f $tmp/$f.$src
      rm -f $tmp/$f.$tgt
      # remove empty lines
      sed '/^$/d' $tmp/$f.$src.split | sed '/^ *$/d' > $tmp/$f.$src
      sed '/^$/d' $tmp/$f.$tgt.split | sed '/^ *$/d' > $tmp/$f.$tgt
      rm -f $tmp/$f.$src.split
      rm -f $tmp/$f.$tgt.split
  done

  echo "Setting every first segment of a split sentence as doc head..."
  for l in $src $tgt; do
    for f in train valid; do
      # generate heads
      sents=$(wc -l $tmp/$f.$l | awk '{print $1;}')
      seq 1 2 $sents > $prep/$f.$lang.$l.heads
    done
  done

  echo "Applying BPEs..."
  for L in $src $tgt; do
      for f in train.$L valid.$L; do
          echo "Apply_bpe.py to ${f}..."
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
      --srcdict data-bin/$corpus/standard/dict.en.txt \
      --joined-dictionary \
      --destdir data-bin/$prep \
      --workers $N_THREADS
  cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "aligned-split" ]
then
  # The aligned-split option will work only if the standard and split datasets
  # already exist and have been processed with the code above
  standard_tmp=$corpus/standard/tmp
  if [ ! -d "$standard_tmp" ]; then
    echo "$standard_tmp not found."
    echo "Before using the 'aligned-split' option, launch this script with 'standard'"
    exit 1
  fi
  split=$corpus/split
  if [ ! -d "$split" ]; then
    echo "$split not found."
    echo "Before using the 'aligned-split' option, launch this script with 'split'"
    exit 1
  fi

  # Set variables for the current option
  prep=$corpus/aligned_split
  tmp=$prep/tmp
  mkdir -p $tmp

  echo "retrieve data from iwslt/standard and doc boundaries from iwslt/split"
  # TODO(lo) all of this has to be corrected:
  # - alignments should be calculated here on files with empty rows as docs boundaries
  # - then splitted (keeping empty rows)
  # - then heads should be calculated
  for f in train valid; do
      for l in $src $tgt; do
      cp $standard_tmp/$f.$l $tmp/
      cp $split/$f.$lang.$l.heads $prep/
      done
      cp alignments/iwslt17.$f.standard.$lang.align $tmp/$f.align
  done

  echo "Splitting sentences..."
  for f in train valid; do
      python ../../scripts/split_corpus_sentences.py \
        --src=$tmp/$f.$src --tgt=$tmp/$f.$tgt --align=$tmp/$f.align \
        | tee $tmp/splitting_$f.logs
      rm -f $tmp/$f.$src
      rm -f $tmp/$f.$tgt
      mv $tmp/$f.$src.split $tmp/$f.$src
      mv $tmp/$f.$tgt.split $tmp/$f.$tgt
  done

  echo "Applying BPEs..."
  for L in $src $tgt; do
      for f in train.$L valid.$L; do
          echo "Apply_bpe.py to ${f}..."
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
      --srcdict data-bin/$corpus/standard/dict.en.txt \
      --joined-dictionary \
      --destdir data-bin/$prep \
      --workers $N_THREADS
  cp $prep/*.heads data-bin/$prep/
###############################################################################
elif [ $1 = "synt-split" ] || [ $1 = "syntall-split" ]
then
  # The synt-split option will work only if the standard and split datasets
  # already exist and have been processed with the code above
  standard_tmp=$corpus/standard/tmp
  if [ ! -d "$standard_tmp" ]; then
    echo "$standard_tmp not found."
    echo "Before using the 'synt-split' option, launch this script with 'standard'"
    exit 1
  fi
  split=$corpus/split
  if [ ! -d "$split" ]; then
    echo "$split not found."
    echo "Before using the 'synt-split' option, launch this script with 'split'"
    exit 1
  fi

  # Set variables for the current option
  if [ $1 = "synt-split" ]
  then
    prep=$corpus/synt_split
    fcoref=CorefSplitInfo_Pronoun
  else
    prep=$corpus/syntall_split
    fcoref=CorefSplitInfo_All
  fi
  tmp=$prep/tmp
  mkdir -p $tmp

  echo "retrieve data from iwslt/standard and doc boundaries from iwslt/split"
  # TODO(lo) all of this has to be corrected:
  # - alignments should be calculated here on files with empty rows as docs boundaries
  # - then splitted (keeping empty rows)
  # - then heads should be calculated
  for f in train valid; do
      for l in $src $tgt; do
      cp $standard_tmp/$f.$l $tmp/
      cp $split/$f.$lang.$l.heads $prep/
      done
  done

  echo "Splitting sentences..."
  for f in train valid; do
      python ../../scripts/split_corpus_sentences.py \
        --src=$tmp/$f.$src --tgt=$tmp/$f.$tgt --coref-info=$corpus/coref_info/$fcoref.data.$lang.$f --verbose \
        | tee $tmp/splitting_$f.logs
      rm -f $tmp/$f.$src
      rm -f $tmp/$f.$tgt
      mv $tmp/$f.$src.split $tmp/$f.$src
      mv $tmp/$f.$tgt.split $tmp/$f.$tgt
  done

  echo "Applying BPEs..."
  for L in $src $tgt; do
      for f in train.$L valid.$L; do
          echo "Apply_bpe.py to ${f}..."
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
      --srcdict data-bin/$corpus/standard/dict.en.txt \
      --joined-dictionary \
      --destdir data-bin/$prep \
      --workers $N_THREADS
  cp $prep/*.heads data-bin/$prep/
###############################################################################
else
    echo "Argument is not valid."
fi
