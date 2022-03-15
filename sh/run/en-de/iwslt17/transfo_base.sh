#!/bin/bash
#
# bash sh/run/en-de/iwslt17/transfo_base.sh --t=finetune --save_dir=standard/k0 --pretrained=checkpoints/en-de/wmt17/transfo_base/checkpoint.avg_last10.pt


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
tgt=de
lang=$src-$tgt
script=sh/run/$lang/iwslt17/transfo_base.sh
task=translation
architecture=transformer_vaswani_wmt_en_fr
test_suites=data/$lang/data-bin/wmt17/test_suites
contrapro=data/en-de/test_suites/ContraPro
if [ -n "$data_dir" ]; then data_dir=$data_dir ; else data_dir=data/$lang/data-bin/iwslt17/standard ; fi
if [ -n "$lenpen" ]; then lenpen=$lenpen ; else lenpen=0.6 ; fi
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$lang/iwslt17/$save_dir; fi

num_workers=8
n_best_checkpoints=5
checkpoint_path=$save_dir/checkpoint_best.pt
# checkpoint_path=$save_dir/checkpoint.avg_last$n_best_checkpoints.pt
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi
if [ -n "$pretrained" ]; then pretrained=$pretrained ; else pretrained=None ; fi
if [ -n "$restore" ]; then restore=$restore ; else restore=checkpoint_last.pt ; fi
if [ -n "$mover" ]; then mover=$mover ; else mover="{}" ; fi
if [ -n "$mt" ]; then maxtok=$mt ; else maxtok=16000 ; fi
if [ -n "$uf" ]; then updatefreq=$uf ; else updatefreq=1 ; fi

if [ -n "$gen_subset" ]; then gen_subset=$gen_subset ; else gen_subset=test ; fi
if [ -n "$testlog" ]; then testlog=$testlog ; else testlog=$gen_subset ; fi

if [ $t = "finetune" ]
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
    --max-tokens $maxtok \
    --update-freq $updatefreq \
    --patience 5 \
    --keep-best-checkpoints $n_best_checkpoints \
    --no-epoch-checkpoints \
    --log-format json \
    | tee -a $save_dir/logs/train.log
###############################################################################
elif [ $t = "boom" ]
then
    # # BLEU on PRO (normal and shuffled)
    # for s in "" ".shuffled"; do
    #     data_dir=$test_suites/large_pronoun_testset/k3$s
    #     d=large_pronoun_testset$s
    #     # score reference
    #     bash $script --t=test --save_dir=$save_dir --testlog=$d --cuda=$cuda --data_dir=$data_dir --lenpen=$lenpen
    # done
    # # BLEU on test set
    # bash $script --t=test --save_dir=$save_dir --cuda=$cuda --lenpen=$lenpen
    # BLEU on shuffled test set
    data_dir=data/$lang/data-bin/iwslt17/test_shuffled
    bash $script --t=test --save_dir=$save_dir --testlog=test.shuffled --cuda=$cuda --data_dir=$data_dir --lenpen=$lenpen
###############################################################################
elif [ $t = "results" ]
then
    for s in "" ".shuffled"; do
        d=test$s
        echo "RESULTS FOR $save_dir/logs/$d.score"
        echo ""
        cat $save_dir/logs/$d.score
        echo "-----------------------------------"
        echo ""
    done
    for s in "" ".shuffled"; do
        d=large_pronoun_testset$s
        echo "RESULTS FOR $save_dir/logs/$d.score"
        echo ""
        cat $save_dir/logs/$d.score
        echo "-----------------------------------"
        echo ""
    done
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
        --inputs $save_dir/checkpoint.best_* \
        --output $save_dir/checkpoint.$n_best_checkpoints.best.average.pt
###############################################################################
elif [ $t = "test-suites" ]
then
    # evaluate on ContraPro (original and with shuffled context)
    for s in ".shuffled"; do
        data_dir=$test_suites/large_pronoun/k3$s
        d=large_pronoun$s
        # score reference
        bash $script --t=score-ref --src=$src --tgt=$tgt --save_dir=$save_dir --data_dir=$data_dir --testlog=$d --cuda=$cuda
        # evaluate
        echo "extract scores..."
        grep ^H $save_dir/logs/$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$d.full_score 
        awk 'NR % 4 == 0' $save_dir/logs/$d.full_score > $save_dir/logs/$d.score
        echo "evaluate model performance on test-suite by comparing scores..."
        python3 $contrapro/evaluate.py --reference $contrapro/contrapro.json --scores $save_dir/logs/$d.score --maximize > $save_dir/logs/$d.result
    done
    echo "-----------------------------------"
    echo ""
    # print results
    for s in "" ".shuffled"; do
        d=large_pronoun$s
        echo "Results for $d"
        echo "file: $save_dir/logs/$d.result"
        grep total $save_dir/logs/$d.result
        echo "-----------------------------------"
        echo ""
    done
###############################################################################
else
    echo "Argument t is not valid."
fi