#!/bin/bash

# Read script arguments and assign them to variables
# bash sh/run/dummy/han_dummy.sh --t=train --src=en --tgt=fr --k=3 --cuda=0
for argument in "$@" 
do

    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)   
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}" 
   fi
done

# Set variables
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi
if [ -n "$data_dir" ]; then data_dir=data/data-bin/dummy.tokenized/$data_dir ; else data_dir=data/data-bin/dummy.tokenized ; fi
if [ -n "$pretrained" ]; then pretrained=$pretrained ; else pretrained=None ; fi
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/dummy/$save_dir; fi
src=$src
tgt=$tgt
num_workers=8
n_best_checkpoints=5

if [ $t = "train" ]
then
    # train
    mkdir -p $save_dir/logs
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --n-context-sents $k \
    --num-workers $num_workers \
    --pretrained-transformer-checkpoint $pretrained \
    --task translation_han \
    --arch han_transformer_test \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 1000 \
    --max-epoch 2 \
    --keep-last-epochs $n_best_checkpoints \
    --keep-best-checkpoints $n_best_checkpoints \
    --no-epoch-checkpoints \
    --log-format json \
    | tee -a $save_dir/logs/train.log
elif [ $t = "test" ]
then
    # test
    fairseq-generate $data_dir \
    --task translation_han \
    --source-lang $src \
    --target-lang $tgt \
    --path $save_dir/checkpoint.$n_best_checkpoints.best.average.pt \
    --batch-size 64 \
    --remove-bpe \
    --beam 4 \
    --lenpen 0.6 \
    --temperature 1.0 \
    --num-workers $num_workers \
    | tee $save_dir/logs/test.log
elif [ $t = "score" ]
then
    grep ^S $save_dir/logs/test.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/gen.out.src
    grep ^T $save_dir/logs/test.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- > $save_dir/logs/gen.out.ref
    grep ^H $save_dir/logs/test.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- > $save_dir/logs/gen.out.sys
    fairseq-score \
    --sentence-bleu \
    --sys $save_dir/logs/gen.out.sys \
    --ref $save_dir/logs/gen.out.ref \
    | tee $save_dir/logs/score.log
elif [ $t = "average" ]
then
    python scripts/average_checkpoints.py \
        --inputs $save_dir/checkpoint.best_bleu_* \
        --output $save_dir/checkpoint.$n_best_checkpoints.best.average.pt
else
    echo "Argument is not valid. Type 'train' or 'test'."
fi

