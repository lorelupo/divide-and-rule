#!/bin/bash
# STANDARD
# bash sh/run/en-fr/iwslt17/han.sh --t=train --cuda=0 --k=3 --save_dir=standard/k3 --pretrained=checkpoints/en-fr/iwslt17/standard/k0/checkpoint_best.pt
# SPLIT
# bash sh/run/en-fr/iwslt17/han.sh --t=train --cuda=0 --k=3 --save_dir=split/k3 --pretrained=checkpoints/en-fr/iwslt17/standard/k0/checkpoint_best.pt --data_dir=data/en-fr/data-bin/iwslt17/split
# bash sh/run/en-fr/iwslt17/han.sh --t=finetune --cuda=1 --k=3 --save_dir=fromsplit/k3 --pretrained=checkpoints/en-fr/iwslt17/split/k3/checkpoint_best.pt
# FROMNEI
# bash sh/run/en-fr/iwslt17/han.sh --t=train --cuda=0 --k=3 --save_dir=split/fromnei/k3 --restore=checkpoints/en-fr/nei/split/k3/checkpoint_best.pt --data_dir=data/en-fr/data-bin/iwslt17/split
# bash sh/run/en-fr/iwslt17/han.sh --t=finetune --cuda=1 --k=1 --save_dir=fromsplit/fromnei/k1 --pretrained=checkpoints/en-fr/iwslt17/split/fromnei/k1/checkpoint_best.pt
# bash sh/run/en-fr/iwslt17/han.sh --t=finetune --cuda=1 --k=1 --save_dir=standard/fromnei/k1 --pretrained=checkpoints/en-fr/nei/standard/k1/checkpoint_best.pt
# CUR
# bash sh/run/en-fr/iwslt17/han.sh --t=finetune --cuda=0 --k=3 --save_dir=standard/cur/k3 --pretrained=checkpoints/en-fr/iwslt17/standard/k1/checkpoint_best.pt

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

# Global variables
if [ -n "$src" ]; then src=$src ; else src=en ; fi
if [ -n "$tgt" ]; then tgt=$tgt ; else tgt=fr ; fi
if [ -n "$corpus" ]; then corpus=$corpus ; else corpus=iwslt17 ; fi

lang=$src-$tgt
task=translation_han
this_script=sh/run/$lang/iwslt17/han.sh

# Point to right 'data' and 'save' directories if not specified as arguments
if [ -n "$data_dir" ]; then data_dir=$data_dir ; else data_dir=data/$lang/data-bin/$corpus/standard ; fi
if [ -n "$lenpen" ]; then lenpen=$lenpen ; else lenpen=0.6 ; fi
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$lang/$corpus/$save_dir; fi

# Set options
# Train options
if [ -n "$num_workers" ]; then num_workers=$num_workers ; else num_workers=8 ; fi
if [ -n "$architecture" ]; then architecture=$architecture ; else architecture=han_transformer_wmt_en_fr ; fi
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi
if [ -n "$pretrained" ]; then pretrained=$pretrained ; else pretrained=None ; fi
if [ -n "$restore" ]; then restore=$restore ; else restore=checkpoint_last.pt ; fi
if [ -n "$testlog" ]; then testlog=$testlog ; else testlog=test ; fi
if [ -n "$mover" ]; then mover=$mover ; else mover="{}" ; fi
if [ -n "$mt" ]; then maxtok=$mt ; else maxtok=8000 ; fi
if [ -n "$uf" ]; then updatefreq=$uf ; else updatefreq=2 ; fi

# Test options
n_best_checkpoints=5
if [ -n "$checkpoint_path" ]; then checkpoint_path=$checkpoint_path ; else checkpoint_path=$save_dir/checkpoint_best.pt ; fi
#if [ -n "$checkpoint_path" ]; then checkpoint_path=$checkpoint_path ; else checkpoint_path=$save_dir/checkpoint.avg_last$n_best_checkpoints.pt ; fi

# Tools
detokenizer=tools/mosesdecoder/scripts/tokenizer/tokenizer.perl
multibleu=tools/mosesdecoder/scripts/generic/multi-bleu-detok.perl

# Run
if [ $t = "train" ]
then
    mkdir -p $save_dir/logs
    python3 -u train.py $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --arch $architecture \
    --pretrained-transformer-checkpoint $pretrained \
    --restore-file $restore \
    --n-context-sents $k \
    --freeze-transfo-params \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --min-lr 1e-09 \
    --lr 1e-03 --warmup-init-lr 1e-07 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens $maxtok \
    --update-freq $updatefreq \
    --patience 5 \
    --no-epoch-checkpoints \
    --log-format json \
    | tee -a $save_dir/logs/train.log
###############################################################################
elif [ $t = "finetune" ]
then
    mkdir -p $save_dir/logs
    python3 -u train.py $data_dir \
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
    --no-epoch-checkpoints \
    --log-format json \
    | tee -a $save_dir/logs/train.log
###############################################################################
elif [ $t = "results" ]
then
    for d in lexical_choice; do
        echo "RESULTS FOR $d"
        cat $save_dir/logs/$d.result
        echo "-----------------------------------"
        echo ""
    done
    for s in "" ".shuffled"; do
        d=test$s
        echo "RESULTS FOR $save_dir/logs/$d.score"
        echo ""
        cat $save_dir/logs/$d.score
        echo "-----------------------------------"
        echo ""
    done
    for s in "" ".shuffled"; do
        d=large_pronoun$s
        echo "RESULTS FOR $save_dir/logs/$d.results"
        echo ""
        grep total $save_dir/logs/$d.result
        echo "-----------------------------------"
        echo ""
    done
###############################################################################
elif [ $t = "test" ]
then
    fairseq-generate $data_dir \
    --task $task \
    --source-lang $src \
    --target-lang $tgt \
    --path $checkpoint_path \
    --batch-size 128 \
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
    $multibleu $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.score
###############################################################################
elif [ $t = "score" ]
then
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | sacremoses detokenize > $save_dir/logs/$testlog.out.sys
    $multibleu $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.score
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
    
    test_suites=/home/getalp/lupol/dev/fairseq/data/$lang/data-bin/wmt14/test_suites


    # evaluate on test-set
    bash $this_scrip --t=test --save_dir=$save_dir --cuda=$cuda --lenpen=$lenpen
    # evaluate on shuffled test-set
    data_dir=data/$lang/data-bin/iwslt17/test_shuffled
    bash $this_scrip --t=test --save_dir=$save_dir --testlog=test.shuffled --cuda=$cuda --data_dir=$data_dir --lenpen=$lenpen
    # evaluate on Bawden's test suites
    for d in lexical_choice; do
        data_dir=$test_suites/$d
        # score reference
        bash $this_scrip --t=score-ref --src=$src --tgt=$tgt --save_dir=$save_dir --data_dir=$data_dir --testlog=$d --cuda=$cuda
        # --mover="{'n_context_sents':'1'}"
        # evaluate
        echo "extract scores..."
        grep ^H $save_dir/logs/$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$d.full_score 
        awk 'NR % 2 == 0' $save_dir/logs/$d.full_score > $save_dir/logs/$d.score
        echo "evaluate model performance on test-suite by comparing scores..."
        orig=data/$lang/bawden/discourse-mt-test-sets/
        python3 $orig/scripts/evaluate.py $orig/test-sets/$d.json $d $save_dir/logs/$d.score --maximise > $save_dir/logs/$d.result
    done
    # evaluate on large pronouns test suite (original and with shuffled context)
    for s in "" ".shuffled"; do
        data_dir=$test_suites/large_pronoun/k3$s
        d=large_pronoun$s
        # score reference
        bash $this_scrip --t=score-ref --src=$src --tgt=$tgt --save_dir=$save_dir --data_dir=$data_dir --testlog=$d --cuda=$cuda
        # evaluate
        echo "extract scores..."
        grep ^H $save_dir/logs/$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$d.full_score 
        awk 'NR % 4 == 0' $save_dir/logs/$d.full_score > $save_dir/logs/$d.score
        echo "evaluate model performance on test-suite by comparing scores..."
        orig=data/$lang/bawden/Large-contrastive-pronoun-testset-EN-FR/OpenSubs
        python3 $orig/scripts/evaluate.py --reference $orig/testset-$lang.json --scores $save_dir/logs/$d.score --maximize --results-file $save_dir/logs/$d.results > $save_dir/logs/$d.result
    done
    echo "-----------------------------------"
    echo ""
    # print results
    for d in lexical_choice; do
        echo "Results for $d"
        cat $save_dir/logs/$d.result
        echo "-----------------------------------"x
        echo ""
    done
    for s in "" ".shuffled"; do
        d=large_pronoun$s
        echo "Results for $d"
        grep total $save_dir/logs/$d.result
        echo "-----------------------------------"
        echo ""
    done
###############################################################################
else
    echo "Argument t is not valid."
fi