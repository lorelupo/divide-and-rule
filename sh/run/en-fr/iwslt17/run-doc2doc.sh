# This script provides command-line statements for training and testing a
# Transformer model for the translation task, on a given corpus and
# with possible source and target extended context
# via the "mode" and "val" arguments

# Read arguments
for argument in "$@" 
do
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)
    # Use string manipulation to set variable names according to convention   
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        v="${v/-/_}"
        v="${v/-/_}"
        declare $v="${value}" 
   fi
done

# Activate virtualenv
#workon nmt

# Move to working directory (must)
cd ~

# Global variables
if [ -n "$src" ]; then src=$src ; else src=en ; fi
if [ -n "$tgt" ]; then tgt=$tgt ; else tgt=fr ; fi
if [ -n "$corpus" ]; then corpus=$corpus ; else corpus=iwslt17 ; fi

lang=$src-$tgt
task=doc2doc_translation
this_script=/home/getalp/alonzocl/dev/scripts/en-fr/run-doc2doc.sh

# Point to right 'data' and 'save' directories if not specified as arguments
if [ -n "$data_dir" ]; then data_dir=$data_dir; else data_dir=data/$lang/data-bin/$corpus/ ; fi
if [[ $save_dir != "checkpoints/"* ]]; then save_dir=checkpoints/$lang/$corpus/$save_dir; fi
mkdir -p $save_dir/logs

# Set options
# Task specific
if [ -z "$mode" ]
then
    echo "You must specify --mode and --val (optional) for the doc2doc task."
    echo "Options are:"
    echo "1) --mode: none, block, slide_block "
    echo "   --val=<desired-num-of-tokens> (default=1000)"
    echo "2) --mode: n2n_block, slide_n2one, slide_n2n"
    echo "   --val=<desired-num-of-sentences> (default=5)"
    exit 0 
fi
if [ $mode = 'none' ] || [ $mode = 'block' ]  || [ $mode = 'slide_block' ]
then
    opt=num-tok
    if [ -n "$val" ]; then val=$val ; else val=1000 ; fi
elif [ $mode = 'n2n_block' ] || [ $mode = 'slide_n2one' ] || [ $mode = 'slide_n2n' ]
then
    opt=num-sent
    if [ -n "$val" ]; then val=$val; else val=5; fi
fi
if [ -n "$max_src_pos" ]; then max_src_pos=$max_src_pos; else max_src_pos=1024; fi
if [ -n "$max_tgt_pos" ]; then max_tgt_pos=$max_tgt_pos; else max_tgt_pos=1024; fi

# Train
if [ -n "$num_workers" ]; then num_workers=$num_workers ; else num_workers=4 ; fi
if [ -n "$arch" ]; then arch=$arch ; else arch=transformer_vaswani_wmt_en_fr ; fi
if [ -n "$dropout" ]; then dropout=$dropout; else dropout=0.1; fi
if [ -n "$lr" ]; then lr=$lr; else lr=1e-03; fi
if [ -n "$min_lr" ]; then min_lr=$min_lr ; else min_lr=1e-09 ; fi # stop training when the lr reaches this minimum (default -1.0)
if [ -n "$lr_sched" ]; then lr_sched=$lr_sched; else lr_sched=inverse_sqrt; fi
if [ -n "$warmup_upd" ]; then warmup_upd=$warmup_upd ; else warmup_upd=4000 ; fi
if [ -n "$warmup_init_lr" ]; then warmup_init_lr=$warmup_init_lr ; else warmup_init_lr=1e-07 ; fi

if [ -n "$cuda" ] ; then export CUDA_VISIBLE_DEVICES=$cuda ; fi
if [ -n "$ddp_backend" ] ; then ddp_backend=$ddp_backend ; else ddp_backend=c10d ; fi
if [ -n "$seed" ]; then seed=$seed ; else seed=0 ; fi
if [ -n "$max_tokens" ]; then max_tokens=$max_tokens ; else max_tokens=4096; fi
if [ -n "$siu" ]; then siu=$siu ; else siu=6000 ; fi
if [ -n "$kiu" ]; then kiu=$kiu ; else kiu=20 ; fi
if [ -n "$pretrained" ]; then pretrained=$pretrained ; else pretrained=None ; fi
if [ -n "$update_freq" ]; then update_freq=$update_freq ; else update_freq=1; fi

if [ -n "$trainlog" ]; then trainlog=$trainlog ; else trainlog=train ; fi
if [ -n "$ftlog" ]; then ftlog=$ftlog ; else ftlog=finetune ; fi
if [ -n "$finaltlog" ]; then finaltlog=$finaltlog ; else finaltlog=finaltun ; fi

# Test
n_best_checkpoints=5
if [ -n "$checkpoint_path" ]; then checkpoint_path=$checkpoint_path ; else checkpoint_path=$save_dir/checkpoint_best.pt ; fi
#if [ -n "$checkpoint_path" ]; then checkpoint_path=$checkpoint_path ; else checkpoint_path=$save_dir/checkpoint_last.pt ; fi

# max_len_a is not necessary if max_tgt_positions is set up properly
if [ -n "$max_len_a" ]; then max_len_a=$max_len_a ; else max_len_a=0 ; fi
if [ -n "$lenpen" ]; then lenpen=$lenpen ; else lenpen=0.6 ; fi
if [ -n "$unkpen" ]; then unkpen=$unkpen ; else unkpen=0 ; fi
if [ -n "$testlog" ]; then testlog=$testlog ; else testlog=test ; fi

# Tools
detokenizer=tools/mosesdecoder/scripts/tokenizer/detokenizer.perl
multibleu=tools/mosesdecoder/scripts/generic/multi-bleu-detok.perl

# Run
if [ $t = "train" ]
then
    # other params: --decoder-normalize-before --sentence-avg  --weight-decay 0.0001
    echo $@ >> $save_dir/logs/$trainlog.log
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --max-source-positions $max_src_pos \
    --max-target-positions $max_tgt_pos \
    --mode $mode \
    --$opt $val \
    --arch $arch \
    --finetune-from-model $pretrained \
    --dropout $dropout \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-9 \
    --clip-norm 0.0 \
    --lr-scheduler $lr_sched --warmup-updates $warmup_upd --min-lr $min_lr \
    --lr $lr --warmup-init-lr $warmup_init_lr \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --ddp-backend $ddp_backend \
    --max-tokens $max_tokens \
    --update-freq $update_freq \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates $siu \
    --keep-interval-updates $kiu \
    --patience 5 \
    --no-epoch-checkpoints \
    --keep-best-checkpoints $n_best_checkpoints \
    --log-format json \
    | tee -a $save_dir/logs/$trainlog.log
elif [ $t = "finetune" ]
then
    # not used so far
    echo $@ >> $save_dir/logs/$ftlog.log
    fairseq-train $data_dir \
    --save-dir $save_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --max-source-positions $max_src_pos \
    --max-target-positions $max_tgt_pos \
    --mode $mode \
    --$opt $val \
    --finetune-from-model $pretrained \
    --arch $arch \
    --dropout $dropout \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler $lr_sched --warmup-updates $warmup_upd --min-lr $min_lr \
    --lr $lr --warmup-init-lr $warmup_init_lr \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --ddp-backend $ddp_backend \
    --max-tokens $max_tokens \
    --update-freq $update_freq \
    --save-interval-updates $siu \
    --keep-interval-updates $kiu \
    --patience 5 \
    --no-epoch-checkpoints \
    --keep-best-checkpoints $n_best_checkpoints \
    --log-format json \
    | tee -a $save_dir/logs/$ftlog.log
elif [ $t = "test" ]
then
    # --max-len-a 3 --lenpen 2 --unkpen 1  \
    echo $@ >> $save_dir/logs/$testlog.log
    fairseq-generate $data_dir \
    --seed $seed \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --max-source-positions $max_src_pos \
    --max-target-positions $max_tgt_pos \
    --mode $mode \
    --$opt $val \
    --path $checkpoint_path \
    --batch-size 64 \
    --remove-bpe '@@ ' \
    --beam 4 \
    --max-len-a $max_len_a --lenpen $lenpen --unkpen $unkpen \
    --temperature 1.0 \
    | tee $save_dir/logs/$testlog.log
    # extract src, tgt (ref) and system hypothesis (sys) from output
    grep ^S $save_dir/logs/$testlog.log | sed 's/^S-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.src
    grep ^T $save_dir/logs/$testlog.log | sed 's/^T-//g' | sort -nk 1 | cut -f2- | sacremoses detokenize > $save_dir/logs/$testlog.out.ref
    grep ^H $save_dir/logs/$testlog.log | sed 's/^H-//g' | sort -nk 1 | cut -f3- | sacremoses detokenize > $save_dir/logs/$testlog.out.sys
    # score
    $multibleu $save_dir/logs/$testlog.out.ref < $save_dir/logs/$testlog.out.sys | tee $save_dir/logs/$testlog.score
elif [ $t = "score-ref" ]
then
    fairseq-generate $data_dir \
    --source-lang $src \
    --target-lang $tgt \
    --num-workers $num_workers \
    --task $task \
    --max-source-positions $max_src_pos \
    --max-target-positions $max_tgt_pos \
    --mode $mode \
    --$opt $val \
    --path $checkpoint_path \
    --score-reference \
    --batch-size 64 \
    --remove-bpe '@@ ' \
    --beam 4 \
    --max-len-a $max_len_a --lenpen $lenpen --unkpen $unkpen \
    --temperature 1.0 \
    | tee $save_dir/logs/$testlog.log
elif [ $t = "test-suites" ]
then
    test_suites=/home/getalp/lupol/dev/fairseq/data/$lang/data-bin/wmt14/test_suites
    orig=/home/getalp/lupol/dev/fairseq/data/$lang/bawden/discourse-mt-test-sets

    # lexical_choice (only for en-fr)
    d=lexical_choice
    bash $this_script --t=score-ref --src=$src --tgt=$tgt \
    --data-dir=$test_suites/$d --save-dir=$save_dir \
    --mode=$mode --opt=$opt --val=$val --testlog=$d

    echo "Extracting scores..."
    grep ^H $save_dir/logs/$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$d.full_score 
    awk 'NR % 2 == 0' $save_dir/logs/$d.full_score > $save_dir/logs/$d.score
    
    echo "Evaluate model performance on test-suite by comparing scores..."
    python $orig/scripts/evaluate.py $orig/test-sets/$d.json $d $save_dir/logs/$d.score --maximise > $save_dir/logs/$d.result

    # anaphora (only for en-fr)
    d=anaphora
    bash $this_script --t=score-ref --src=$src --tgt=$tgt \
    --data-dir=$test_suites/$d --save-dir=$save_dir \
    --mode=$mode --opt=$opt --val=$val --testlog=$d

    echo "Extracting scores..."
    grep ^H $save_dir/logs/$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$d.full_score 
    awk 'NR % 2 == 0' $save_dir/logs/$d.full_score > $save_dir/logs/$d.score
    
    echo "Evaluate model performance on test-suite by comparing scores..."
    python $orig/scripts/evaluate.py $orig/test-sets/$d.json $d $save_dir/logs/$d.score --maximise > $save_dir/logs/$d.result

    ##contrapro
    orig=/home/getalp/lupol/dev/fairseq/data/$lang/bawden/Large-contrastive-pronoun-testset-EN-FR/OpenSubs
    s=k3
    data_dir=$test_suites/large_pronoun/$s
    d=last_large_pronoun$s
    
    bash $this_script --t=score-ref --src=$src --tgt=$tgt \
    --save-dir=$save_dir --data-dir=$data_dir \
    --mode=$mode --opt=$opt --val=$val --testlog=$d

    echo "Extracting scores..."
    grep ^H $save_dir/logs/$d.log | sed 's/^H-//g' | sort -nk 1 | cut -f2 > $save_dir/logs/$d.full_score 
    awk 'NR % 4 == 0' $save_dir/logs/$d.full_score > $save_dir/logs/$d.score
    
    echo "Evaluate model performance on test-suite by comparing scores..."
    python $orig/scripts/evaluate.py --reference $orig/testset-$lang.json \
    --scores $save_dir/logs/$d.score --maximize \
    --results-file $save_dir/logs/$d.results > $save_dir/logs/$d.result
    
elif [ $t = "results" ]
then
    d=test
    echo "Results for $save_dir/logs/$d.score"
    cat $save_dir/logs/$d.score
    echo "-----------------------------------"
    d=lexical_choice
    echo "Results for $d"
    echo "file: $save_dir/logs/$d.result"
    cat $save_dir/logs/$d.result
    echo "-----------------------------------"
    d=anaphora
    echo "Results for $d"
    echo "file: $save_dir/logs/$d.result"
    cat $save_dir/logs/$d.result
    echo "-----------------------------------"
    d=large_pronoun
    echo "Results for $d"
    echo "file: $save_dir/logs/$d.result"
    grep total $save_dir/logs/$d.result
    echo "-----------------------------------"
else
    echo "Argument t is not valid."
fi
