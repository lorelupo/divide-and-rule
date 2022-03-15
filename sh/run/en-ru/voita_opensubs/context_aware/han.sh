#!/bin/bash

# bash sh/run/en-ru/voita_opensubs/context_aware/han.sh --t=test-suites --cuda=1 --lenpen=0.6 --save_dir=standard/k1

# Read script arguments and assign them to variables
for argument in "$@" 
do
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)
    # Use string manipulation to set variable names according to convention   
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        v="${v/-/_}"
        declare $v="${value}" 
   fi
done

# Set variables
src=en
tgt=ru
lang=$src-$tgt
corpus=shared_BPE_voita_opensubs/context_aware

script=sh/run/$lang/voita_opensubs/context_aware/han.sh
task=translation_han
architecture=han_transformer_wmt_en_fr
# test_suites=data/$lang/data-bin/wmt17/test_suites
# contrapro=data/en-de/test_suites/ContraPro
if [ -n "$data_dir" ]; then data_dir=$data_dir ; else data_dir=data/$lang/data-bin/$corpus/standard ; fi
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$lang/$corpus/$save_dir; fi

num_workers=8
n_best_checkpoints=5
checkpoint_path=$save_dir/checkpoint_best.pt
# checkpoint_path=$save_dir/checkpoint.avg_last$n_best_checkpoints.pt
if [ -n "$pretrained" ]; then pretrained=$pretrained ; else pretrained=None ; fi
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi
if [ -n "$lenpen" ]; then lenpen=$lenpen ; else lenpen=0.6 ; fi
if [ -n "$mover" ]; then mover=$mover ; else mover="{}" ; fi
if [ -n "$mt" ]; then maxtok=$mt ; else maxtok=16000 ; fi
if [ -n "$uf" ]; then updatefreq=$uf ; else updatefreq=2 ; fi
if [ -n "$siu" ]; then siu=$siu ; else siu=2000 ; fi
if [ -n "$gen_subset" ]; then gen_subset=$gen_subset ; else gen_subset=test ; fi
if [ -n "$testlog" ]; then testlog=$testlog ; else testlog=$gen_subset ; fi

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
    --max-source-positions 200 \
    --max-target-positions 200 \
    --patience 5 \
    --save-interval-updates $siu \
    --keep-interval-updates 10 \
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
    --max-source-positions 200 \
    --max-target-positions 200 \
    --patience 5 \
    --save-interval-updates $siu \
    --keep-interval-updates 10 \
    --no-epoch-checkpoints \
    --log-format json \
    | tee -a $save_dir/logs/train.log
###############################################################################
elif [ $t = "test" ]
then
    fairseq-generate $data_dir \
    --gen-subset $gen_subset \
    --task $task \
    --source-lang $src \
    --target-lang $tgt \
    --path $checkpoint_path \
    --batch-size 64 \
    --remove-bpe \
    --beam 4 \
    --lenpen $lenpen \
    --temperature 1.2 \
    --num-workers $num_workers \
    | tee $save_dir/logs/$testlog.log
    # score with multi-bleu
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | awk 'NR % 4 == 0' | cut -f2- | awk '{print tolower($0)}' > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | awk 'NR % 4 == 0' | cut -f2- | awk '{print tolower($0)}' > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | awk 'NR % 4 == 0' | cut -f3- | awk '{print tolower($0)}' > $save_dir/logs/$testlog.out.sys
    tools/mosesdecoder/scripts/generic/multi-bleu.perl $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.score
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
elif [ $t = "test-suites" ]
then
    # evaluate on test-set
    bash $script --t=test --save_dir=$save_dir --cuda=$cuda --lenpen=$lenpen --mover=$mover
    # evaluate on consistency testset
    data_dir=data/en-ru/data-bin/voita_opensubs/testset_consistency/ellipsis_vp
    d=ellipsis_vp
    # score reference
    bash $script --t=score-ref --src=$src --tgt=$tgt --save_dir=$save_dir --data_dir=$data_dir --testlog=$d --cuda=$cuda --mover=$mover
    # evaluate
    echo "extract scores..."
    grep ^H $save_dir/logs/$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$d.full_score 
    awk 'NR % 4 == 0' $save_dir/logs/$d.full_score | cut -c2- > $save_dir/logs/$d.score
    echo "evaluate model performance on test-suite by comparing scores..."
    repo=data/en-ru/test_suites/good-translation-wrong-in-context/
    python3 $repo/scripts/evaluate_consistency.py --repo-dir $repo --test ellipsis_vp --scores $save_dir/logs/$d.score --results-file $save_dir/logs/$d.results > $save_dir/logs/$d.result
    echo "-----------------------------------"
    bash $script --t=results --save_dir=$save_dir
###############################################################################
elif [ $t = "results" ]
then
    d=test
    echo "RESULTS FOR $save_dir/logs/$d.score"
    echo ""
    cat $save_dir/logs/$d.score
    echo "-----------------------------------"
    d=ellipsis_vp
    echo "RESULTS FOR $save_dir/logs/$d.result"
    echo ""
    cat $save_dir/logs/$d.result
    echo "-----------------------------------"
    echo ""
###############################################################################
elif [ $t = "average" ]
then
    python scripts/average_checkpoints.py \
        --inputs $save_dir/checkpoint_*0.pt \
        --output $save_dir/checkpoint.avg_last5.pt
###############################################################################
else
    echo "Argument t is not valid."
fi