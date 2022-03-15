#!/bin/bash
# bash sh/run/en-ru/voita_opensubs/context_agnostic/transfo_base.sh --t=test --cuda=1 --save_dir=standard
# bash sh/run/en-ru/voita_opensubs/context_agnostic/transfo_base.sh --t=test-by-available-context --save_dir=checkpoints/en-ru/voita_opensubs/context_agnostic/standard/transfo_base_fairseq --cuda=$cuda --scored_checkpoint=avg_last

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
corpus=voita_opensubs/context_agnostic
this_script=sh/run/$lang/$corpus/transfo_base.sh
task=translation

# architecture=transformer_vaswani_wmt_en_fr
# architecture=transformer_base_extra_dropout
if [ -n "$architecture" ]; then architecture=$architecture ; else architecture=transformer_voita_fairseq ; fi
if [ -n "$data_dir" ]; then data_dir=$data_dir ; else data_dir=data/$lang/data-bin/$corpus/standard ; fi
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$lang/$corpus/$save_dir; fi
mkdir -p $save_dir/logs

num_workers=8
ncheckpoints=5
# checkpoint_path=$save_dir/checkpoint.avg_best$ncheckpoints.pt
# checkpoint_path=$save_dir/checkpoint.avg_last$ncheckpoints.pt
# checkpoint_path=$save_dir/checkpoint.avg_last$ncheckpoints\_old.pt
# checkpoint_path=$save_dir/checkpoint_best_bleu.pt
if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi

if [ -n "$lr" ]; then lr=$lr ; else lr=0.0005 ; fi
if [ -n "$max_tokens" ]; then max_tokens=$max_tokens ; else max_tokens=16000 ; fi
if [ -n "$update_freq" ]; then update_freq=$update_freq ; else update_freq=1 ; fi
if [ -n "$patience" ]; then patience=$patience ; else patience=15 ; fi
if [ -n "$max_update" ]; then max_update=$max_update ; else max_update=300000 ; fi
if [ -n "$keep_last_epochs" ]; then keep_last_epochs=$keep_last_epochs ; else keep_last_epochs=$patience ; fi
if [ -n "$keep_best_checkpoints" ]; then keep_best_checkpoints=$keep_best_checkpoints ; else keep_best_checkpoints=5 ; fi
if [ -n "$siu" ]; then siu=$siu ; else siu=2048 ; fi
if [ -n "$kiu" ]; then kiu=$kiu ; else kiu=10 ; fi
if [[ $scored_checkpoint = 'best' ]]
then
    checkpoint_path=$save_dir/checkpoint_best.pt
    checkpoint_prefix=best.
elif [[ $scored_checkpoint = 'avg_last' ]]
then
    checkpoint_path=$save_dir/checkpoint.avg_last$ncheckpoints.pt
    checkpoint_prefix=avg_last$ncheckpoints.
elif [[ $scored_checkpoint = 'last' ]]
then
    checkpoint_path=$save_dir/checkpoint_last.pt
    checkpoint_prefix=last.
else
    echo "Argument checkpoint_path is not valid."
fi

if [ -n "$mover" ]; then mover=$mover ; else mover="{}" ; fi
if [ -n "$lenpen" ]; then lenpen=$lenpen ; else lenpen=0.4 ; fi
if [ -n "$temperature" ]; then temperature=$temperature ; else temperature=1 ; fi
if [ -n "$gen_subset" ]; then gen_subset=$gen_subset ; else gen_subset=test ; fi

# Logging
if [ -n "$log_format" ]; then log_format=$log_format ; else log_format=json ; fi
if [ -n "$log_interval" ]; then log_interval=$log_interval ; else log_interval=100 ; fi
if [ -n "$trainlog" ]; then trainlog=$trainlog ; else trainlog=train ; fi
if [ -n "$ftlog" ]; then ftlog=$ftlog ; else ftlog=finetune ; fi
if [ -n "$log_prefix" ]; then log_prefix=$log_prefix ; else log_prefix= ; fi
if [ -n "$testlog" ]; then testlog=$log_prefix$checkpoint_prefix$testlog ; else testlog=$log_prefix$checkpoint_prefix$gen_subset ; fi
if [ -n "$tensorboard_logdir" ]; then tensorboard_logdir=$tensorboard_logdir ; else tensorboard_logdir=$save_dir/logs ; fi

if [ $t = "train" ]
then
    # train
    echo $@ >> $save_dir/logs/$trainlog.log
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --arch $architecture \
    --optimizer adam --adam-betas '(0.9, 0.998)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --min-lr 1e-09 \
    --lr $lr --warmup-init-lr 1e-07 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens $max_tokens \
    --update-freq $update_freq \
    --patience $patience \
    --keep-last-epochs $keep_last_epochs \
    --keep-best-checkpoints $keep_best_checkpoints \
    --log-format json \
    --tensorboard-logdir $tensorboard_logdir \
    --fp16 \
    | tee -a $save_dir/logs/train.log
    # --save-interval-updates $siu \
    # --keep-interval-updates $kiu \
    # --no-epoch-checkpoints \
    # --eval-bleu \
    # --eval-tokenized-bleu \
    # --eval-bleu-args '{"beam": 4, "lenpen": 0.6}' \
    # --eval-bleu-remove-bpe \
    # --best-checkpoint-metric bleu \
    # --maximize-best-checkpoint-metric \
    # --max-update $max_update \
###############################################################################
elif [ $t = "test" ]
then
    echo $@ >> $save_dir/logs/$testlog.log
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
    --temperature $temperature \
    --num-workers $num_workers \
    --seed $seed \
    | tee $save_dir/logs/$testlog.log
    # score with multi-bleu
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | paste -d' ' - - - - > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | paste -d' ' - - - - > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | paste -d' ' - - - - > $save_dir/logs/$testlog.out.sys
    tools/mosesdecoder/scripts/generic/multi-bleu.perl -lc $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.result
    echo "Result for $save_dir/logs/$testlog.result :"
    cat $save_dir/logs/$testlog.result
###############################################################################
elif [ $t = "test-by-available-context" ]
then
    # generate context-aware test set
    testlog=testaware
    # if [ ! -f "$save_dir/logs/$log_prefix$checkpoint_prefix$testlog.log" ]
    # then
    #     bash $this_script --t=test --save_dir=$save_dir --data_dir=data/en-ru/data-bin/voita_opensubs/context_aware/standard --testlog=$testlog --cuda=$cuda --lenpen=$lenpen --mover=$mover --scored_checkpoint=$scored_checkpoint
    # fi

    # extract sentences by available context
    testlog=$log_prefix$checkpoint_prefix$testlog
    rm $save_dir/logs/$testlog.bycontext.result
    available_context=0
    for n in 1 2 3 0
    do
        cat $save_dir/logs/$testlog.out.ref | awk -v n=$n 'NR % 4 == n' > $save_dir/logs/$testlog.out$available_context.ref 
        cat $save_dir/logs/$testlog.out.sys | awk -v n=$n 'NR % 4 == n' > $save_dir/logs/$testlog.out$available_context.sys
        # score
        # echo "------- available_context=$available_context ---------------------" >> $save_dir/logs/$testlog.bycontext.result
        # calculate sentence-level BLEU scores
        fairseq-score \
        --sentence-bleu \
        --sys $save_dir/logs/$testlog.out$available_context.sys \
        --ref $save_dir/logs/$testlog.out$available_context.ref \
        > $save_dir/logs/$testlog.out$available_context.sentBLEU
        # calculate average sentence-level BLEU score
        cat $save_dir/logs/$testlog.out$available_context.sentBLEU | \
            grep -o BLEU4........ | \
            cut -d' ' -f3 | \
            sed 's/,$//g' | \
            awk -v ctx=$available_context '{sum+=$1}END{print "AVG sentence-level BLEU for pos="ctx,":",sum/NR}' >> $save_dir/logs/$testlog.bycontext.result
        # calculate corpus BLEU
        # tools/mosesdecoder/scripts/generic/multi-bleu.perl -lc $save_dir/logs/$testlog.out$n.ref  < $save_dir/logs/$testlog.out$n.sys >> $save_dir/logs/$testlog.bycontext.result
        # update var
        available_context=$(($available_context+1))
    done
    # print results
    cat $save_dir/logs/$testlog.bycontext.result
    rm -rf $save_dir/logs/$testlog.out[0-4]*
    rm -rf $save_dir/logs/$testlog.all.*
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
    --remove-bpe '@@ ' \
    --num-workers $num_workers \
    | tee $save_dir/logs/$testlog.log
###############################################################################
elif [ $t = "test-suites" ]
then
    # evaluate on test-set
    bash $this_script --t=test --save_dir=$save_dir --data_dir=$data_dir --cuda=$cuda --lenpen=$lenpen --mover=$mover --scored_checkpoint=$scored_checkpoint
    # evaluate on consistency testset
    test_suite_data=data/en-ru/data-bin/lowercase_voita_opensubs/testset_consistency
    test_suite_repo=data/en-ru/test_suites/good-translation-wrong-in-context
    for d in deixis_dev
    do
        # score reference
        bash $this_script --t=score-ref --save_dir=$save_dir --data_dir=$test_suite_data/$d --testlog=$d --cuda=$cuda --mover=$mover --scored_checkpoint=$scored_checkpoint
        # evaluate
        echo "extract scores for $d..."
        grep ^H $save_dir/logs/$checkpoint_prefix$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$checkpoint_prefix$d.full_score 
        awk 'NR % 4 == 0' $save_dir/logs/$checkpoint_prefix$d.full_score | cut -c2- > $save_dir/logs/$checkpoint_prefix$d.score
        echo "evaluate model performance on $d by comparing scores..."
        python3 scripts/evaluate_consistency_voita.py --repo-dir $test_suite_repo --test $d --scores $save_dir/logs/$checkpoint_prefix$d.score --results-file $save_dir/logs/$checkpoint_prefix$d.results > $save_dir/logs/$checkpoint_prefix$d.result
        echo "-----------------------------------"
    done
    bash $this_script --t=results --save_dir=$save_dir --scored_checkpoint=$scored_checkpoint
###############################################################################
elif [ $t = "test-suites-concat" ]
then
    # evaluate on test-set
    bash $this_script --t=test-concat --save_dir=$save_dir --data_dir=$data_dir --cuda=$cuda --lenpen=$lenpen --mover=$mover --scored_checkpoint=$scored_checkpoint
    # evaluate on consistency testset
    test_suite_data=data/en-ru/data-bin/voita_opensubs/testset_consistency
    test_suite_repo=data/en-ru/test_suites/good-translation-wrong-in-context
    # deixis_dev lex_cohesion_dev deixis_test lex_cohesion_test ellipsis_infl ellipsis_vp
    for d in deixis_dev deixis_test lex_cohesion_dev lex_cohesion_test ellipsis_infl ellipsis_vp
    do
        # score reference
        bash $this_script --t=score-ref --save_dir=$save_dir --data_dir=$test_suite_data/$concat/$d --testlog=$d --cuda=$cuda --mover=$mover --scored_checkpoint=$scored_checkpoint
        # evaluate
        echo "extract scores for $d..."
        grep ^H $save_dir/logs/$checkpoint_prefix$d.log | \
            sed 's/^H-//g' | \
            sort -nk 1 | \
            cut -f2 | \
            cut -c2- > $save_dir/logs/$checkpoint_prefix$d.score 
        echo "evaluate model performance on $d by comparing scores..."
        python3 scripts/evaluate_consistency_voita.py --repo-dir $test_suite_repo --test $d --scores $save_dir/logs/$checkpoint_prefix$d.score --results-file $save_dir/logs/$checkpoint_prefix$d.results > $save_dir/logs/$checkpoint_prefix$d.result
        echo "-----------------------------------"
    done
    bash $this_script --t=results --save_dir=$save_dir --scored_checkpoint=$scored_checkpoint
###############################################################################
elif [ $t = "test-concat" ]
then
    echo $@ >> $save_dir/logs/$testlog.log
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
    --temperature $temperature \
    --num-workers $num_workers \
    --seed $seed \
    | tee $save_dir/logs/$testlog.log
    # score with multi-bleu
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | head -n -1 | sed 's/_eos //g' > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | head -n -1 | sed 's/_eos //g'  > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | head -n -1 | sed 's/_eos //g'  > $save_dir/logs/$testlog.out.sys
    # grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | head -n -1 | awk -F'_eos ' '{$1=$1}1' OFS='\n' > $save_dir/logs/$testlog.out.src
    # grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | head -n -1 | awk -F'_eos ' '{$1=$1}1' OFS='\n'  > $save_dir/logs/$testlog.out.ref
    # grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | head -n -1 | awk -F'_eos ' '{$1=$1}1' OFS='\n'  > $save_dir/logs/$testlog.out.sys
    # grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | awk -F'_eos ' '{print $NF}' > $save_dir/logs/$testlog.out.src
    # grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | awk -F'_eos ' '{print $NF}' > $save_dir/logs/$testlog.out.ref
    # grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | awk -F'_eos ' '{print $NF}' > $save_dir/logs/$testlog.out.sys
    tools/mosesdecoder/scripts/generic/multi-bleu.perl -lc $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.result
    cat $save_dir/logs/$testlog.result
###############################################################################
elif [ $t = "dev-suites" ]
then
    # evaluate on consistency testset
    test_suite_data=data/en-ru/data-bin/voita_opensubs/testset_consistency
    test_suite_repo=data/en-ru/test_suites/good-translation-wrong-in-context
    for d in deixis_dev lex_cohesion_dev
    do
        # score reference
        bash $this_script --t=score-ref --save_dir=$save_dir --data_dir=$test_suite_data/$d --testlog=$d --cuda=$cuda --mover=$mover --mode=$mode --opt=$opt --val=$val --scored_checkpoint=$scored_checkpoint
        # evaluate
        echo "extract scores for $d..."
        grep ^H $save_dir/logs/$checkpoint_prefix$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$checkpoint_prefix$d.full_score 
        awk 'NR % 4 == 0' $save_dir/logs/$checkpoint_prefix$d.full_score | cut -c2- > $save_dir/logs/$checkpoint_prefix$d.score
        echo "evaluate model performance on $d by comparing scores..."
        python3 scripts/evaluate_consistency_voita.py --repo-dir $test_suite_repo --test $d --scores $save_dir/logs/$checkpoint_prefix$d.score --results-file $save_dir/logs/$checkpoint_prefix$d.results > $save_dir/logs/$checkpoint_prefix$d.result
        # Note: except for lexical cohesion, the correct example is always the first in each contrastive block
        echo "-----------------------------------"
    done
    bash $this_script --t=results --save_dir=$save_dir --mode=$mode --opt=$opt --val=$val
############################################################################### 
elif [ $t = "search_lenpen" ]
then
    for l in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1 2
    do
    bash $this_script --t=test --save_dir=$save_dir --t=test --cuda=$cuda --lenpen=$l --gen_subset=valid --testlog=valid_bleu_lenpen$l
    # gather results
    echo "------- lenpen=$l ---------------------" >> $save_dir/logs/valid_bleu_lenpen.summary
    cat $save_dir/logs/valid_bleu_lenpen$l.score >> $save_dir/logs/valid_bleu_lenpen.summary
    rm $save_dir/logs/valid_bleu_lenpen$l*
    done
    cat $save_dir/logs/valid_bleu_lenpen.summary
############################################################################### 
elif [ $t = "search_temperature" ]
then
    for l in 0.5 0.7 0.8 0.9 1 1.1 1.2 1.3 1.5
    do
    bash $this_script --t=test --save_dir=$save_dir --t=test --cuda=$cuda --temperature=$l --gen_subset=valid --testlog=valid_bleu_temperature$l
    # gather results
    echo "------- temperature=$l ---------------------" >> $save_dir/logs/valid_bleu_temperature.summary
    cat $save_dir/logs/valid_bleu_temperature$l.score >> $save_dir/logs/valid_bleu_temperature.summary
    rm $save_dir/logs/valid_bleu_temperature$l*
    done
    cat $save_dir/logs/valid_bleu_temperature.summary
###############################################################################
elif [ $t = "average" ]
then
    # python scripts/average_checkpoints.py \
    #     --inputs $save_dir/checkpoint.best_loss* \
    #     --output $save_dir/checkpoint.avg_best5.pt \
    # | tee $save_dir/logs/average.log
    python scripts/average_checkpoints.py \
        --inputs $save_dir/ \
        --num-epoch-checkpoints 5 \
        --output $save_dir/checkpoint.avg_last5.pt \
    # | tee $save_dir/logs/average.log
    # python scripts/average_checkpoints.py \
    # --inputs $save_dir/checkpoint366.pt $save_dir/checkpoint348.pt $save_dir/checkpoint376.pt $save_dir/checkpoint.best_loss_3.566.pt $save_dir/checkpoint.best_loss_3.569.pt \
    # --output $save_dir/checkpoint.avg_best5.pt
############################################################################### 
elif [ $t = "results" ]
then
    echo "####################################################################"
    echo "RESULTS FOR $save_dir/logs/$checkpoint_prefix"

    # best checkpoint epoch according to valid loss
    grep "valid_best_loss" $save_dir/logs/train.log | tail -n1 | grep -oP "(?<=epoch\": )[0-9]+" >> tmp.results
    # OR best checkpoint epoch according to valid bleu
    grep "valid_best_bleu" $save_dir/logs/train.log | tail -n1 | grep -oP "(?<=epoch\": )[0-9]+" >> tmp.results

    # discourse dev set
    for d in deixis_dev lex_cohesion_dev
    do
        cat $save_dir/logs/$checkpoint_prefix$d.result | grep -oP "(?<=Total accuracy:  )[0-9\.]+" | awk -v x=100 '{ print $1*x }' >> tmp.results
    done

    # best valid loss
    grep -oP "(?<=valid_best_loss\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1 >> tmp.results
    # OR best valid bleu
    grep -oP "(?<=valid_best_bleu\": \")[0-9\.]+" $save_dir/logs/train.log | tail -n1 >> tmp.results

    # checkpoint
    echo ${checkpoint_prefix::-1} >> tmp.results

    # discourse test set
    for d in deixis_test lex_cohesion_test ellipsis_infl ellipsis_vp
    do
        cat $save_dir/logs/$checkpoint_prefix$d.result | grep -oP "(?<=Total accuracy:  )[0-9\.]+" | awk -v x=100 '{ print $1*x }' >> tmp.results
    done

    # BLEU on test set
    d=test
    cat $save_dir/logs/$checkpoint_prefix$d.result | grep -oP "(?<=BLEU = )[0-9\.]+" >> tmp.results

    # transpose results
    cat tmp.results | tr "\n" ","
    rm tmp.results
    echo -e "\n"
else
    echo "Argument t is not valid."
fi
