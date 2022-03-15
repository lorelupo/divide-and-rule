#!/bin/bash
# bash sh/run/en-fr/wmt14/transfo_base.sh --t=finetune --cuda=1  --save_dir=checkpoints/en-fr/nei/standard/k0 --pretrained=checkpoints/en-fr/wmt14/transfo_base/checkpoint.avg10.pt --data_dir=data/en-fr/data-bin/nei/standard

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
corpus=wmt14
script=sh/run/$lang/$corpus/transfo_base.sh

task=translation
architecture=transformer_vaswani_wmt_en_fr
test_suites=data/$lang/data-bin/$corpus/test_suites
bawden=data/$lang/bawden
if [ -n "$data_dir" ]; then data_dir=$data_dir ; else data_dir=data/$lang/data-bin/$corpus/standard ; fi
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$lang/$corpus/$save_dir; fi


num_workers=8
n_best_checkpoints=5
checkpoint_path=$save_dir/checkpoint_best.pt
# checkpoint_path=$save_dir/checkpoint.avg_last$n_best_checkpoints.pt
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi
if [ -n "$pretrained" ]; then pretrained=$pretrained ; else pretrained=None ; fi
if [ -n "$testlog" ]; then testlog=$testlog ; else testlog=test ; fi
if [ -n "$siu" ]; then siu=$siu ; else siu=6000 ; fi

if [ $t = "train" ]
then
    # train
    mkdir -p $save_dir/logs
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --arch $architecture \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --min-lr 1e-09 \
    --lr 0.0005 --warmup-init-lr 1e-07 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 \
    --fp16 \
    --patience 5 \
    --log-format json \
    --save-interval-updates $siu \
    --keep-interval-updates 20 \
    --max-update 6500 \
    --no-epoch-checkpoints \
    --tensorboard-logdir $save_dir/logs \
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
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --min-lr 1e-09 \
    --lr 0.0005 --warmup-init-lr 1e-07 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 16000 \
    --patience 5 \
    --log-format json \
    --save-interval-updates $siu \
    --keep-interval-updates 10 \
    --max-update 250000 \
    --no-epoch-checkpoints \
    --tensorboard-logdir $save_dir/logs \
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
    --temperature 1.0 \
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
elif [ $t = "score-split" ]
then
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | paste -d " "  - - > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | paste -d " "  - - > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | paste -d " "  - - > $save_dir/logs/$testlog.out.sys
    fairseq-score \
    --sys $save_dir/logs/$testlog.out.sys \
    --ref $save_dir/logs/$testlog.out.ref \
    | tee $save_dir/logs/$testlog.score
###############################################################################
elif [ $t = "score-ref" ]
then
    fairseq-generate $data_dir \
    --task $task \
    --source-lang $src \
    --target-lang $tgt \
    --path $checkpoint_path \
    --score-reference \
    --batch-size 64 \
    --remove-bpe \
    --num-workers $num_workers \
    | tee $save_dir/logs/$testlog.log
###############################################################################
elif [ $t = "average" ]
then
    python scripts/average_checkpoints.py \
        --inputs $save_dir \
        --num-update-checkpoints $n_best_checkpoints \
        --output $save_dir/$avg_checkpoint
###############################################################################
elif [ $t = "test-suites" ]
then
    # evaluate on Bawden's test suites
    for d in lexical_choice; do
        data_dir=$test_suites/$d
        # score reference
        bash $script --t=score-ref --src=$src --tgt=$tgt --save_dir=$save_dir --data_dir=$data_dir --testlog=$d --cuda=$cuda
        # evaluate
        echo "extract scores..."
        grep ^H $save_dir/logs/$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$d.full_score 
        awk 'NR % 2 == 0' $save_dir/logs/$d.full_score > $save_dir/logs/$d.score
        echo "evaluate model performance on test-suite by comparing scores..."
        orig=$bawden/discourse-mt-test-sets/
        python3 $orig/scripts/evaluate.py $orig/test-sets/$d.json $d $save_dir/logs/$d.score --maximise > $save_dir/logs/$d.result
    done
    # evaluate on large pronouns test suite (original and with shuffled context)
    for s in ""; do
        data_dir=$test_suites/large_pronoun/k3$s
        d=large_pronoun$s
        # score reference
        bash $script --t=score-ref --src=$src --tgt=$tgt --save_dir=$save_dir --data_dir=$data_dir --testlog=$d --cuda=$cuda
        # evaluate
        echo "extract scores..."
        grep ^H $save_dir/logs/$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$d.full_score 
        awk 'NR % 4 == 0' $save_dir/logs/$d.full_score > $save_dir/logs/$d.score
        echo "evaluate model performance on test-suite by comparing scores..."
        orig=$bawden/Large-contrastive-pronoun-testset-EN-FR/OpenSubs
        python3 $orig/scripts/evaluate.py --reference $orig/testset-$lang.json --scores $save_dir/logs/$d.score --maximize > $save_dir/logs/$d.result
        # test with BLEU on test suite
        # bash $script --t=test --src=$src --tgt=$tgt --save_dir=$save_dir --data_dir=$data_dir --testlog=test_pronoun$s --cuda=$cuda
    done
    echo "-----------------------------------"
    echo ""
    # print results
    for d in lexical_choice; do
        echo "Results for $d"
        cat $save_dir/logs/$d.result
        echo "-----------------------------------"
        echo ""
    done
    for s in ""; do
        d=large_pronoun$s
        echo "Results for $d"
        grep total $save_dir/logs/$d.result
        tail $save_dir/logs/test_pronoun$s.score
        echo "-----------------------------------"
        echo ""
    done
###############################################################################
else
    echo "Argument t is not valid."
fi