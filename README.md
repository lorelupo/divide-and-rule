# Divide and Rule: Effective Pre-Training for Context-Aware Multi-Encoder Translation Models

This repository contains the official implementation of the experiments discussed in the ACL22 paper [Divide and Rule: Effective Pre-Training for Context-Aware Multi-Encoder Translation Models](https://arxiv.org/abs/2103.17151). The implementation is based on the [fairseq package v0.9.0](https://github.com/pytorch/fairseq/tree/v0.9.0).

# Requirements and Installation

[fairseq](https://github.com/pytorch/fairseq) requires the following:

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

To install and develop locally:
```bash
cd divide-and-rule
pip install --editable ./
```

# Splitting methods

The implementation of the four splitting methods proposed in our paper can be found in [scripts/split_corpus_sentences.py](scripts/split_corpus_sentences), whose basic usage (*middle-split*) is the following:

```
python scripts/split_corpus_sentences.py --src=/your/source/corpus --tgt=/your/target/corpus
```

# Replicate results

The steps to reproduce our _K1-d&r_ results on IWSLT En-De are enumerated below:

1. Prepare WMT17 dataset `bash sh/prepare/en-de/wmt17/sent_all.sh`

2. Train K0 on WMT17 `bash sh/run/en-de/wmt17/transfo_base.sh --t=train --save_dir=transfo_base` and average 10 best checkpoints `bash sh/run/en-de/wmt17/transfo_base.sh --t=average --save_dir=transfo_base`

3. Download IWSLT7 [from WIT3](https://wit3.fbk.eu/2017-01) and unzip it in `./data/en-de/orig/iwslt17`

4. Prepare IWSLT17 `bash sh/prepare/en-de/iwslt17/doc_all.sh standard`

5. Finetune K0 on IWSLT17 `bash sh/run/en-de/iwslt17/transfo_base.sh --t=finetune --save_dir=standard/k0 --pretrained=checkpoints/en-de/wmt17/transfo_base/checkpoint.avg_last10.pt`

6. Prepare IWSLT17split `bash sh/prepare/en-de/iwslt17/doc_all.sh split`

7. Train K1 on IWSLT17split `bash sh/run/en-de/iwslt17/han.sh --t=train --cuda=0 --k=1 --save_dir=split/k1 --pretrained=checkpoints/en-de/iwslt17/standard/k0/checkpoint_best.pt --data_dir=data/en-de/data-bin/iwslt17/split`

8. Finetune K1 on the original IWSLT17 `bash sh/run/en-de/iwslt17/han.sh --t=finetune --cuda=0 --k=1 --save_dir=fromsplit/k1 --pretrained=checkpoints/en-de/iwslt17/split/k1/checkpoint_best.pt`

9. Prepare ContraPro `bash sh/prepare/en-de/wmt17/large_pronoun.sh --k=3 --shuffle=False`

10. Evaluate K1 `bash sh/run/en-de/iwslt17/han.sh --t=test-suites --cuda=0 --k=1 --save_dir=fromsplit/k1`

11. Display results `bash sh/run/en-de/iwslt17/han.sh --t=results --k=1 --save_dir=fromsplit/k1`

The other results presented in the paper can be reproduced with analogous scripts, that can be found in `./sh` .
For the English-Russian language pair, we use a corpus extracted from OpenSubtitles2018 by [Voita et al. (2019)](https://www.aclweb.org/anthology/P19-1116/), available [here](https://github.com/lena-voita/good-translation-wrong-in-context#training-data).

# Citation

The [Divide and Rule](https://arxiv.org/abs/2103.17151) paper:
```bibtex
@article{lupo2021divide,
  title={Divide and Rule: Training Context-Aware Multi-Encoder Translation Models with Little Resources},
  author={Lupo, Lorenzo and Dinarelli, Marco and Besacier, Laurent},
  journal={arXiv preprint arXiv:2103.17151},
  year={2021}
}
```

For Fairseq, please cite:

```bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
