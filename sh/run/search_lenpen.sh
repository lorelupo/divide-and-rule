#!/bin/bash

# bash sh/run/search_lenpen.sh

cuda=1
k=1


# DE k0 ##################
for l in 0.01, 0.1, 0.3, 0.5, 0.8, 1,2, 4, 8
do
 bash sh/run/en-ru/voita_opensubs/context_agnostic/transfo_base.sh --save_dir=standard/transfo_base_lr0009 --t=test --cuda=$cuda --lenpen=$l --gen_subset=valid --testlog=valid_lenpen$l
 # gather results
 echo "------- lenpen=$l ---------------------" >>
 cat checkpoints/en-de/iwslt17/standard/k0/logs/valid_lenpen$l.score >>
done

for l in 0.01, 0.1, 0.3, 0.5, 0.8, 1,2, 4, 8
do
 echo "------- lenpen=$l ---------------------"
 cat checkpoints/en-de/iwslt17/standard/k0/logs/valid_lenpen$l.score
done


# # RU k0 ##################
# # 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5
# for l in 0.1 0.2 0.3
# do
#  bash sh/run/en-ru/voita_opensubs/context_agnostic/transfo_base.sh --t=test --cuda=$cuda --lenpen=$l --save_dir=standard/transfo_base_old --gen_subset=valid --testlog=valid_lenpen$l
# done

# for l in 0.1 0.2 0.3
# do
#  echo "------- lenpen=$l ---------------------"
#  cat checkpoints/en-ru/voita_opensubs/context_agnostic/standard/transfo_base_old/logs/valid_lenpen$l.score
# done

# # RU k1 ##################
# for l in 0.1 0.2 0.3
# do
#  bash sh/run/en-ru/voita_opensubs/context_aware/han.sh --t=test --cuda=$cuda --lenpen=$l --save_dir=standard/k$k --gen_subset=valid --testlog=valid_lenpen$l
# done

# for l in 0.1 0.2 0.3
# do
#  echo "------- lenpen=$l ---------------------"
#  cat checkpoints/en-ru/voita_opensubs/context_aware/standard/k$k/logs/valid_lenpen$l.score
# done

# # RU k1-d&r ##############
# for l in 0.1 0.2 0.3
# do
#  bash sh/run/en-ru/voita_opensubs/context_aware/han.sh --t=test --cuda=$cuda --lenpen=$l --save_dir=fromsplit/k$k --gen_subset=valid --testlog=valid_lenpen$l
# done

# for l in 0.1 0.2 0.3
# do
#  echo "------- lenpen=$l ---------------------"
#  cat checkpoints/en-ru/voita_opensubs/context_aware/fromsplit/k$k/logs/valid_lenpen$l.score
# done