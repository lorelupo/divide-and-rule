#!/usr/bin/env bash
# Previous to this script, you should already have run
# sh/preprocess_iwslt16_fr_en_doc.sh

# Move to ./data directory
cd data

# Setting variables
src=en
tgt=fr
lang=$src-$tgt
prep=nei
indir=standard
outdir=test_standard_shuffled

# mkdir already to avoid awk crashing
mkdir $prep/$outdir

# shuffle data (only for testing!)
paste -d '|' $prep/$indir/test.$src $prep/$indir/test.$tgt | shuf | awk -v FS="|" -v out1=$prep/$outdir/test.$src -v out2=$prep/$outdir/test.$tgt '{ print $1 > out1 ; print $2 > out2 }'

# remove outdir for avoiding fairseq-preprocess crash!!
rm -rf data-bin/$prep/$outdir
mkdir -p data-bin/$prep/$outdir

echo "Binarizing data..."
fairseq-preprocess \
    --source-lang $src \
    --target-lang $tgt \
    --testpref $prep/$outdir/test \
    --srcdict data-bin/$prep/$indir/dict.en.txt \
    --joined-dictionary \
    --destdir data-bin/$prep/$outdir

# copy document heads in output directory
cp data-bin/$prep/$indir/test.*.heads data-bin/$prep/$outdir/