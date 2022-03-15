#!/bin/bash
# bash sh/run/en-fr/iwslt17/transfo_base.sh --t=finetune --save_dir=standard/power_k0 --cuda=1 --pretrained=checkpoints/en-fr/wmt14/transfo_base/checkpoint.avg10.pt
# bash sh/run/en-fr/iwslt17/transfo_base.sh --t=finetune --save_dir=split/k0 --cuda=0 --pretrained=checkpoints/en-fr/wmt14/transfo_base/checkpoint.avg10.pt --data_dir=data/data-bin/iwslt17/split
# bash sh/run/en-fr/iwslt17/transfo_base.sh --t=test --save_dir=standard/k0 --cuda=0 --data_dir=data/data-bin/wmt14/test_suites/large_pronoun/k3 --testlog=test_pronoun
# bash sh/run/en-fr/iwslt17/transfo_base.sh --t=finetune --cuda=1  --save_dir=fromsplit/k0 --pretrained=checkpoints/iwslt17/split/k0/checkpoint_best.pt

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
if [ -n "$lang" ]; then lang=$lang; else lang=$src-$tgt ; fi
if [ -n "$this_script" ]; then this_script=$this_script; else this_script=sh/run/$lang/$corpus/transfo_base.sh ; fi
if [ -n "$task" ]; then task=$task; else task=translation ; fi

# Common
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi
if [ -n "$num_workers" ]; then num_workers=$num_workers ; else num_workers=8 ; fi

# Data
if [ -n "$data_dir" ]; then data_dir=$data_dir; else data_dir=data/$lang/data-bin/$corpus/standard ; fi
if [ -n "$max_tokens" ]; then max_tokens=$max_tokens ; else max_tokens=4000; fi
if [ -n "$update_freq" ]; then update_freq=$update_freq ; else update_freq=1; fi

# Model
if [ -n "$arch" ]; then arch=$arch ; else arch=transformer_vaswani_wmt_en_fr ; fi
if [ -n "$activation_dropout" ]; then activation_dropout=$activation_dropout ; else activation_dropout=0.0 ; fi
if [ -n "$attention_dropout" ]; then attention_dropout=$attention_dropout ; else attention_dropout=0.0 ; fi

# Loss
if [ -n "$criterion" ]; then criterion=$criterion; else criterion=label_smoothed_cross_entropy; fi
if [ -n "$label_smoothing" ]; then label_smoothing=$label_smoothing; else label_smoothing=0.1; fi
if [ -n "$weight_decay" ]; then weight_decay=$weight_decay; else weight_decay=0.0; fi

# Optimization
if [ -n "$lr_scheduler" ]; then lr_scheduler=$lr_scheduler; else lr_scheduler=inverse_sqrt; fi
if [ -n "$lr" ]; then lr=$lr; else lr=1e-03; fi
if [ -n "$min_lr" ]; then min_lr=$min_lr ; else min_lr=1e-09 ; fi # stop training when the lr reaches this minimum (default -1.0)
if [ -n "$total_num_update" ]; then total_num_update=$total_num_update; else total_num_update=16000; fi # 16000 ups corresponds to 15 epoch with max_tokens=128K
if [ -n "$warmup_updates" ]; then warmup_updates=$warmup_updates ; else warmup_updates=4000 ; fi
if [ -n "$warmup_init_lr" ]; then warmup_init_lr=$warmup_init_lr ; else warmup_init_lr=1e-07 ; fi
if [ -n "$end_learning_rate" ]; then end_learning_rate=$end_learning_rate ; else end_learning_rate=1e-09 ; fi


# Checkpoints
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$lang/$corpus/$save_dir; fi
if [ -n "$ncheckpoints" ]; then ncheckpoints=$ncheckpoints ; else ncheckpoints=5 ; fi
if [ -n "$patience" ]; then patience=$patience ; else patience=10 ; fi
if [ -n "$max_update"]; then max_update=$max_update ; else max_update=0 ; fi
if [ -n "$keep_last_epochs" ]; then keep_last_epochs=$keep_last_epochs ; else keep_last_epochs=$patience ; fi
if [ -n "$keep_best_checkpoints" ]; then keep_best_checkpoints=$keep_best_checkpoints ; else keep_best_checkpoints=$ncheckpoints ; fi
if [ -n "$save_interval_updates" ]; then save_interval_updates=$save_interval_updates ; else save_interval_updatesu=6000 ; fi
if [ -n "$keep_interval_updates" ]; then keep_interval_updates=$keep_interval_updates ; else keep_interval_updates=0 ; fi
if [ -n "$scored_checkpoint" ]; then scored_checkpoint=$scored_checkpoint ; else scored_checkpoint=best ; fi
if [ $scored_checkpoint = 'best' ]
then
    checkpoint_path=$save_dir/checkpoint_best.pt
    checkpoint_prefix=best.
elif [ $scored_checkpoint = 'avg_last' ]
then
    checkpoint_path=$save_dir/checkpoint.avg_last$ncheckpoints.pt
    checkpoint_prefix=avg_last$ncheckpoints.
elif [ $scored_checkpoint = 'avg_closest' ]
then
    checkpoint_path=$save_dir/checkpoint.avg_closest$ncheckpoints.pt
    checkpoint_prefix=avg_closest$ncheckpoints.
elif [ $scored_checkpoint = 'last' ]
then
    checkpoint_path=$save_dir/checkpoint_last.pt
    checkpoint_prefix=last.
else
    echo "Argument scored_checkpoint is not valid."
fi

# Test
if [ -n "$mover" ]; then mover=$mover ; else mover="{}" ; fi
if [ -n "$lenpen" ]; then lenpen=$lenpen ; else lenpen=0.6 ; fi
if [ -n "$temperature" ]; then temperature=$temperature ; else temperature=1 ; fi
if [ -n "$batch_size" ]; then batch_size=$batch_size ; else batch_size=64 ; fi
if [ -n "$include_eos" ]; then include_eos=$include_eos ; else include_eos=0 ; fi
if [ -n "$gen_subset" ]; then gen_subset=$gen_subset ; else gen_subset=test ; fi

# Logging
if [ -n "$log_format" ]; then log_format=$log_format ; else log_format=json ; fi
if [ -n "$log_interval" ]; then log_interval=$log_interval ; else log_interval=100 ; fi
if [ -n "$trainlog" ]; then trainlog=$trainlog ; else trainlog=train ; fi
if [ -n "$ftlog" ]; then ftlog=$ftlog ; else ftlog=finetune ; fi
if [ -n "$log_prefix" ]; then log_prefix=$log_prefix ; else log_prefix= ; fi
if [ -n "$testlog" ]; then testlog=$log_prefix$checkpoint_prefix$testlog ; else testlog=$log_prefix$checkpoint_prefix$gen_subset ; fi
if [ -n "$tensorboard_logdir" ]; then tensorboard_logdir=$tensorboard_logdir ; else tensorboard_logdir=$save_dir/logs ; fi
mkdir -p $save_dir/logs

###############################################################################
if [ $t = "train" ]
then
    echo $@ >> $save_dir/logs/$trainlog.log
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --arch $arch \
    --criterion $criterion --label-smoothing $label_smoothing --weight-decay $weight_decay \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --lr-scheduler $lr_scheduler --lr $lr --warmup-updates $warmup_updates --warmup-init-lr $warmup_init_lr --min-lr $min_lr \
    --max-tokens $max_tokens \
    --update-freq $update_freq \
    --patience $patience \
    --keep-last-epochs $keep_last_epochs \
    --log-format $log_format \
    --log-interval $log_interval \
    --tensorboard-logdir $tensorboard_logdir \
    --fp16 \
    | tee -a $save_dir/logs/train.log
###############################################################################
elif [ $t = "finetune" ]
then
    echo $@ >> $save_dir/logs/$trainlog.log
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --arch $arch \
    --finetune-from-model $pretrained \
    --criterion $criterion --label-smoothing $label_smoothing --weight-decay $weight_decay \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --lr-scheduler $lr_scheduler --lr $lr --warmup-updates $warmup_updates --warmup-init-lr $warmup_init_lr --min-lr $min_lr \
    --max-tokens $max_tokens \
    --update-freq $update_freq \
    --patience $patience \
    --keep-last-epochs $keep_last_epochs \
    --log-format $log_format \
    --log-interval $log_interval \
    --tensorboard-logdir $tensorboard_logdir \
    --fp16 \
    | tee -a $save_dir/logs/train.log
###############################################################################
elif [ $t = "boom" ]
then
    for s in "" ".shuffled"; do
        data_dir=$test_suites/large_pronoun_testset/k3$s
        d=large_pronoun_testset$s
        # score reference
        bash $script --t=test --save_dir=$save_dir --testlog=$d --cuda=$cuda --data_dir=$data_dir
    done
    for s in "" ".shuffled"; do
        d=large_pronoun_testset$s
        echo "Results for $d"
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
    --temperature 1.0 \
    --num-workers $num_workers \
    | tee $save_dir/logs/$testlog.log
    # score with sacrebleu
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | sacremoses detokenize > $save_dir/logs/$testlog.out.sys
    tools/mosesdecoder/scripts/generic/multi-bleu-detok.perl $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.score
###############################################################################
elif [ $t = "boom" ]
then
    # BLEU on PRO (normal and shuffled)
    for s in "" ".shuffled"; do
        data_dir=$test_suites/large_pronoun_testset/k3$s
        d=large_pronoun_testset$s
        # score reference
        bash $script --t=test --save_dir=$save_dir --testlog=$d --cuda=$cuda --data_dir=$data_dir --lenpen=$lenpen
    done
    # BLEU on test set
    bash $script --t=test --save_dir=$save_dir --cuda=$cuda --lenpen=$lenpen
    # BLEU on shuffled test set
    data_dir=data/$lang/data-bin/iwslt17/test_shuffled
    bash $script --t=test --save_dir=$save_dir --testlog=test.shuffled --cuda=$cuda --data_dir=$data_dir --lenpen=$lenpen
###############################################################################
elif [ $t = "score" ]
then
    # grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.src
    # grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.ref
    # grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | sacremoses detokenize > $save_dir/logs/$testlog.out.sys
    # tools/mosesdecoder/scripts/generic/multi-bleu-detok.perl $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.score
    detokenizer=tools/mosesdecoder/scripts/tokenizer/detokenizer.perl
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | $detokenizer -l $tgt > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | $detokenizer -l $tgt > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | $detokenizer -l $tgt > $save_dir/logs/$testlog.out.sys
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
    # python scripts/average_checkpoints.py \
    #     --inputs $save_dir/checkpoint.best_loss* \
    #     --output $save_dir/checkpoint.avg_best5.pt \
    # | tee $save_dir/logs/average.log
    # python scripts/average_checkpoints.py \
    #     --inputs $save_dir/ \
    #     --num-epoch-checkpoints $ncheckpoints \
    #     --output $save_dir/checkpoint.avg_last$ncheckpoints.pt \
    # | tee $save_dir/logs/average.log
    python scripts/average_checkpoints.py \
        --inputs $save_dir/ \
        --num-epoch-checkpoints $(($ncheckpoints-1)) \
        --closest-to-best \
        --output $save_dir/checkpoint.avg_closest$ncheckpoints.pt \
    | tee $save_dir/logs/average.log
    # python scripts/average_checkpoints.py \
    # --inputs $save_dir/checkpoint_best.pt $save_dir/checkpoint7.pt $save_dir/checkpoint8.pt $save_dir/checkpoint9.pt $save_dir/checkpoint10.pt \
    # --output $save_dir/checkpoint.avg_closest5.pt \
    # | tee $save_dir/logs/average.log
###############################################################################
elif [ $t = "test-suites" ]
then
    # evaluate on test-set
    bash $this_script --t=test --save_dir=$save_dir --data_dir=$data_dir --cuda=$cuda --lenpen=$lenpen --mover=$mover --scored_checkpoint=$scored_checkpoint
    # evaluate on Bawden's test suites
    test_suites=~/dev/fairseq/data/$lang/data-bin/wmt14/test_suites/bawden
    for d in lexical_choice anaphora; do
        # score reference
        bash $this_script --t=score-ref --save_dir=$save_dir --data_dir=$test_suites/$d --testlog=$d --cuda=$cuda --scored_checkpoint=$scored_checkpoint --batch_size=32
        # evaluate
        echo "extract scores for $d..."
        grep ^H $save_dir/logs/$checkpoint_prefix$d.log | sed 's/^H-//g' | sort -nk 1 | awk 'NR % 2 == 0' | cut -f2 > $save_dir/logs/$checkpoint_prefix$d.score 
        echo "evaluate model performance on test-suite by comparing scores..."
        orig=data/$lang/bawden/discourse-mt-test-sets
        python3 $orig/scripts/evaluate.py $orig/test-sets/$d.json $d $save_dir/logs/$checkpoint_prefix$d.score --maximise > $save_dir/logs/$checkpoint_prefix$d.result
    done
    # evaluate on large pronouns test suite (original and with shuffled context)
    test_suites=~/dev/fairseq/data/$lang/data-bin/wmt14/test_suites
    for s in ""; do
        data_dir=$test_suites/large_pronoun/k3$s
        d=large_pronoun$s
        # score reference
        bash $this_script --t=score-ref --save_dir=$save_dir --data_dir=$test_suites/large_pronoun/k3$s --testlog=$d --cuda=$cuda --mover=$mover --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint --batch_size=32 --need_seg_label=$need_seg_label        # --mover="{'n_context_sents':'1'}"
        # evaluate
        echo "extract scores for $d..."
        grep ^H $save_dir/logs/$checkpoint_prefix$d.log | sed 's/^H-//g' | sort -nk 1 | awk 'NR % 4 == 0' |cut -f2 > $save_dir/logs/$checkpoint_prefix$d.score
        echo "evaluate model performance on test-suite by comparing scores..."
        orig=data/$lang/bawden/Large-contrastive-pronoun-testset-EN-FR/OpenSubs
        python3 scripts/evaluate_contrapro.py --reference $orig/testset-$lang.json --scores $save_dir/logs/$checkpoint_prefix$d.score --maximize --results-file $save_dir/logs/$checkpoint_prefix$d.results > $save_dir/logs/$checkpoint_prefix$d.result
    done
    # print results
    bash $this_script --t=results --save_dir=$save_dir --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint
    rm -rf $save_dir/logs/$checkpoint_prefix$d.full_score $save_dir/logs/$checkpoint_prefix$d.score $save_dir/logs/$checkpoint_prefix$d.results
###############################################################################
elif [ $t = "results" ]
then
    echo "####################################################################"
    echo "RESULTS FOR $save_dir/logs/$checkpoint_prefix"

    # best checkpoint epoch according to valid loss
    grep "valid_best_loss" $save_dir/logs/train.log | tail -n1 | grep -oP "(?<=epoch\": )[0-9]+" >> tmp.results
    # OR best checkpoint epoch according to valid bleu
    grep "valid_best_bleu" $save_dir/logs/train.log | tail -n1 | grep -oP "(?<=epoch\": )[0-9]+" >> tmp.results

    # best valid loss
    grep -oP "(?<=valid_best_loss\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1 >> tmp.results
    # OR best valid bleu
    grep -oP "(?<=valid_best_bleu\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1 >> tmp.results

    # checkpoint
    echo ${checkpoint_prefix::-1} >> tmp.results

    # bawden's test suites
    d=lexical_choice
    cat $save_dir/logs/$checkpoint_prefix$d.result | grep "Total precision" | cut -d" " -f6 | awk '{ print $1*100 }' >> tmp.results
    d=anaphora
    cat $save_dir/logs/$checkpoint_prefix$d.result | grep "Overall precision" | cut -d" " -f6 | awk '{ print $1*100 }' >> tmp.results
    # french contrapro
    d=large_pronoun
    cat $save_dir/logs/$checkpoint_prefix$d.result | grep -A5 "statistics by ante distance" | grep -v "statistics by ante distance" | cut -d" " -f5 | cut -c1-6 | awk '{print $1*100}' >> tmp.results

    # BLEU on test set
    d=test
    cat $save_dir/logs/$checkpoint_prefix$d.score | grep -oP "(?<=BLEU = )[0-9\.]+" >> tmp.results

    # transpose results
    cat tmp.results | tr "\n" ","
    rm tmp.results
    echo -e "\n"
###############################################################################
else
    echo "Argument t is not valid."
fi

