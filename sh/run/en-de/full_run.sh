#!/bin/bash

# STANDARD

# bash sh/run/en-de/iwslt17/han.sh --t=train --cuda=0 --k=1 --save_dir=standard/k1 --pretrained=checkpoints/en-de/iwslt17/standard/k0/checkpoint_best.pt
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --cuda=0 --k=1 --save_dir=standard/k1
# bash sh/run/en-de/iwslt17/han.sh --t=train --cuda=0 --k=3 --save_dir=standard/k3 --pretrained=checkpoints/en-de/iwslt17/standard/k0/checkpoint_best.pt
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --cuda=0 --k=1 --save_dir=standard/k3

# NEI

# bash sh/run/en-de/nei/han.sh --t=train --cuda=1 --k=1 --save_dir=standard/k1 --pretrained=checkpoints/en-de/iwslt17/standard/k0/checkpoint_best.pt
# bash sh/run/en-de/iwslt17/han.sh --t=finetune --cuda=1 --k=1 --save_dir=standard/fromnei/k1 --pretrained=checkpoints/en-de/nei/standard/k1/checkpoint_best.pt
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --cuda=1 --k=1 --save_dir=standard/fromnei/k1

# bash sh/run/en-de/nei/han.sh --t=train --cuda=0 --k=3 --save_dir=standard/k3 --pretrained=checkpoints/en-de/iwslt17/standard/k0/checkpoint_best.pt 
# bash sh/run/en-de/iwslt17/han.sh --t=finetune --cuda=0 --k=3 --save_dir=standard/fromnei/k3 --pretrained=checkpoints/en-de/nei/standard/k3/checkpoint_last.pt
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --cuda=0 --k=3 --save_dir=standard/fromnei/k3

# SPLIT

# bash sh/run/en-de/iwslt17/han.sh --t=train --cuda=0 --k=1 --save_dir=split/k1 --pretrained=checkpoints/en-de/iwslt17/standard/k0/checkpoint_best.pt --data_dir=data/en-de/data-bin/iwslt17/split
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --cuda=0 --k=1 --save_dir=split/k1
# bash sh/run/en-de/iwslt17/han.sh --t=finetune --cuda=0 --k=1 --save_dir=fromsplit/k1 --pretrained=checkpoints/en-de/iwslt17/split/k1/checkpoint_best.pt
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --cuda=0 --k=1 --save_dir=fromsplit/k1

# bash sh/run/en-de/iwslt17/han.sh --t=train --cuda=1 --k=3 --save_dir=split/k3 --pretrained=checkpoints/en-de/iwslt17/standard/k0/checkpoint_best.pt --data_dir=data/en-de/data-bin/iwslt17/split
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --cuda=1 --k=3 --save_dir=split/k3
# bash sh/run/en-de/iwslt17/han.sh --t=finetune --cuda=1 --k=3 --save_dir=fromsplit/k3 --pretrained=checkpoints/en-de/iwslt17/split/k3/checkpoint_best.pt
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --cuda=1 --k=3 --save_dir=fromsplit/k3

# NEI-SPLIT

# bash sh/run/en-de/nei/han.sh --t=train --k=1 --cuda=0 --save_dir=split/k1 --pretrained=checkpoints/en-de/iwslt17/standard/k0/checkpoint_best.pt --data_dir=data/en-de/data-bin/nei/split
# bash sh/run/en-de/iwslt17/han.sh --t=finetune --k=1 --cuda=0 --save_dir=split/fromnei/k1 --pretrained=checkpoints/en-de/nei/split/k1/checkpoint_best.pt --data_dir=data/en-de/data-bin/iwslt17/split
# bash sh/run/en-de/iwslt17/han.sh --t=finetune --k=1 --cuda=0 --save_dir=fromsplit/fromnei/k1 --pretrained=checkpoints/en-de/iwslt17/split/fromnei/k1/checkpoint_best.pt
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --k=1 --cuda=0 --save_dir=split/fromnei/k1
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --k=1 --cuda=0 --save_dir=fromsplit/fromnei/k1

# bash sh/run/en-de/nei/han.sh --t=train --k=3 --cuda=1 --save_dir=split/k3 --pretrained=checkpoints/en-de/iwslt17/standard/k0/checkpoint_best.pt --data_dir=data/en-de/data-bin/nei/split
# bash sh/run/en-de/iwslt17/han.sh --t=finetune --k=3 --cuda=1 --save_dir=split/fromnei/k3 --pretrained=checkpoints/en-de/nei/split/k3/checkpoint_best.pt --data_dir=data/en-de/data-bin/iwslt17/split
# bash sh/run/en-de/iwslt17/han.sh --t=finetune --k=3 --cuda=1 --save_dir=fromsplit/fromnei/k3 --pretrained=checkpoints/en-de/iwslt17/split/fromnei/k3/checkpoint_best.pt
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --k=3 --cuda=1 --save_dir=split/fromnei/k3
# bash sh/run/en-de/iwslt17/han.sh --t=test-suites --k=3 --cuda=1 --save_dir=fromsplit/fromnei/k3
