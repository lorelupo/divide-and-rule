#!/bin/bash

# STANDARD
# bash sh/run/en-fr/nei/han.sh --t=test-suites --cuda=0 --k=1 --save_dir=standard/k1 --pretrained=checkpoints/en-fr/iwslt17/standard/k0/checkpoint_best.pt
# bash sh/run/en-fr/nei/han.sh --t=train --cuda=0 --k=3 --save_dir=standard/k3 --pretrained=checkpoints/en-fr/iwslt17/standard/k0/checkpoint_best.pt
# bash sh/run/en-fr/nei/han.sh --t=train --cuda=1 --k=3 --save_dir=standard/k3_fromnei --pretrained=checkpoints/en-fr/nei/standard/k0/checkpoint_best.pt
# SPLIT
# bash sh/run/en-fr/nei/han.sh --t=train --cuda=1 --k=3 --save_dir=split/k3 --pretrained=checkpoints/en-fr/iwslt17/standard/k0/checkpoint_best.pt --data_dir=data/en-fr/data-bin/nei/split
# CUR
# bash sh/run/en-fr/nei/han.sh --t=finetune --cuda=1 --k=3 --save_dir=standard/cur/k3 --pretrained=checkpoints/en-fr/nei/standard/k1/checkpoint_best.pt

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

# Set variables
src=en
tgt=fr
lang=$src-$tgt
script=sh/run/$lang/nei/han.sh
task=translation_han
architecture=han_transformer_wmt_en_fr
test_suites=data/$lang/data-bin/wmt14/test_suites
bawden=data/$lang/bawden
if [ -n "$data_dir" ]; then data_dir=$data_dir ; else data_dir=data/$lang/data-bin/nei/standard ; fi
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$lang/nei/$save_dir ; fi


num_workers=8
n_best_checkpoints=5
checkpoint_path=$save_dir/checkpoint_best.pt
# checkpoint_path=$save_dir/checkpoint.avg_last$n_best_checkpoints.pt
# checkpoint_path=$save_dir/checkpoint.avg_last3.pt
# checkpoint_path=$save_dir/checkpoint_last.pt
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi
if [ -n "$pretrained" ]; then pretrained=$pretrained ; else pretrained=None ; fi
if [ -n "$testlog" ]; then testlog=$testlog ; else testlog=test ; fi
if [ -n "$mover" ]; then mover=$mover ; else mover="{}" ; fi
if [ -n "$mt" ]; then maxtok=$mt ; else maxtok=8000 ; fi
if [ -n "$uf" ]; then updatefreq=$uf ; else updatefreq=2 ; fi
if [ -n "$siu" ]; then siu=$siu ; else siu=6000 ; fi

if [ $t = "train" ]
then
    mkdir -p $save_dir/logs
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --arch $architecture \
    --pretrained-transformer-checkpoint $pretrained \
    --n-context-sents $k \
    --freeze-transfo-params \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --min-lr 1e-09 \
    --lr 5e-04 --warmup-init-lr 1e-07 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens $maxtok \
    --update-freq $updatefreq \
    --patience 5 \
    --save-interval-updates $siu \
    --keep-interval-updates 5 \
    --no-epoch-checkpoints \
    --log-format json \
    | tee -a $save_dir/logs/train.log
###############################################################################
elif [ $t = "finetune" ]
then
    mkdir -p $save_dir/logs
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --finetune-from-model $pretrained \
    --task $task \
    --arch $architecture \
    --n-context-sents $k \
    --freeze-transfo-params \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler fixed --lr 2e-04 --fa 1 --lr-shrink 0.99 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens $maxtok \
    --update-freq $updatefreq \
    --patience 5 \
    --save-interval-updates $siu \
    --keep-interval-updates 5 \
    --no-epoch-checkpoints \
    --log-format json \
    | tee -a $save_dir/logs/train.log
###############################################################################
elif [ $t = "test" ]
then
    fairseq-generate $data_dir \
    --task $task \
    --source-lang $src \
    --target-lang $tgt \
    --path $checkpoint_path \
    --batch-size 64 \
    --remove-bpe \
    --beam 4 \
    --lenpen 0.6 \
    --temperature 1 \
    --num-workers $num_workers \
    | tee $save_dir/logs/$testlog.log
    # score with sacrebleu
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | sacremoses detokenize > $save_dir/logs/$testlog.out.sys
    tools/mosesdecoder/scripts/generic/multi-bleu-detok.perl $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.score
###############################################################################
elif [ $t = "score" ]
then
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | sacremoses detokenize > $save_dir/logs/$testlog.out.sys
    tools/mosesdecoder/scripts/generic/multi-bleu-detok.perl $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.score
###############################################################################
elif [ $t = "score-ref" ]
then
    fairseq-generate $data_dir \
    --task $task \
    --source-lang $src \
    --target-lang $tgt \
    --path $checkpoint_path \
    --model-overrides $mover \
    --score-reference \
    --batch-size 64 \
    --remove-bpe \
    --num-workers $num_workers \
    | tee $save_dir/logs/$testlog.log
###############################################################################
elif [ $t = "average" ]
then
    python scripts/average_checkpoints.py \
        --inputs $save_dir/checkpoint_6_* $save_dir/checkpoint_5_* \
        --output $save_dir/checkpoint.avg_last$n_best_checkpoints.pt
###############################################################################
else
    echo "Argument t is not valid."
fi