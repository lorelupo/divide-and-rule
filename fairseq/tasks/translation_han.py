import os
import sys

import torch
import numpy as np

from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq import utils

from fairseq.data import (
    AppendTokenDataset, ConcatDataset, data_utils, iterators, encoders,
    indexed_dataset, DocLanguagePairDataset, PrependTokenDataset,
    StripTokenDataset, TruncateDataset, FairseqDataset, Dictionary
)
import itertools
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# TODO(lo) check if possible to binarize heads as well.
# TODO(lo) Check dataset_impl for heads (now='raw').
# TODO(lo) Log heads (num documents) found.
# TODO(lo) check whether heads Tensor goes on GPU during training.


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    shuffle_sents=False,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False
):
    def split_exists(split, src, tgt, lang, data_path):
        datafile = os.path.join(
            data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang)
        )
        headsfile = os.path.join(
            data_path, '{}.{}-{}.{}.{}'.format(split, src, tgt, lang, 'heads')
        )
        exist = indexed_dataset.dataset_exists(datafile, impl=dataset_impl) and \
            indexed_dataset.dataset_exists(headsfile, impl='raw')
        return exist

    def load_doc_heads(doc_heads_path):
        heads = []
        with open(doc_heads_path) as infile:
            for line in infile:
                heads.append(int(line.strip()))
        if heads[0] == 0:
            return torch.LongTensor(heads)
        elif heads[0] == 1:
            return torch.LongTensor(heads) - 1
        else:
            raise ValueError(
                'The first document head should be the first line, with id=0 or 1'
            )

    src_datasets = []
    tgt_datasets = []
    doc_heads = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(
                data_path, '{}.{}-{}.'.format(split_k, src, tgt)
            )
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(
                data_path, '{}.{}-{}.'.format(split_k, tgt, src)
            )
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    'Dataset or document heads not found: {} ({})'.format(
                        split, data_path
                    )
                )

        # load indexed datasets
        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            '{} {} {}-{} {} examples'.format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        # load heads of documents
        src_split_heads = load_doc_heads(prefix + src + '.heads')
        tgt_split_heads = load_doc_heads(prefix + tgt + '.heads')
        assert torch.all(torch.eq(src_split_heads, tgt_split_heads)), \
            'Heads of source and target documents are not the same for {}.'.format(
                prefix)

        # add the line after the document's last line as fictional head
        doc_heads_split = torch.cat(
            (src_split_heads, src_split_heads.new_tensor([len(src_dataset)]))
        )
        doc_heads.append(doc_heads_split)

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0]
        doc_heads = doc_heads[0]
    else:  # TODO(lo) check whether this is useful
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index('[{}]'.format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index('[{}]'.format(tgt))
            )
        eos = tgt_dict.index('[{}]'.format(tgt))

    tgt_dataset_sizes = tgt_dataset.sizes

    return DocLanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        doc_heads,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        shuffle_sents=shuffle_sents,
        eos=eos
    )


@register_task('translation_han')
class HANTranslationTask(TranslationTask):
    """
    Document-level translation task for the
    Hierarchical Attention Network <https://www.aclweb.org/anthology/D18-1325/>`_.
    """
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            "--shuffle-sents",
            action="store_true",
            help="shuffle dataset sentences before batching"
            )
        # fmt: on

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            shuffle_sents=self.args.shuffle_sents,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
        )

    # TODO(lo) only useful for the interactive mode
    # def build_dataset_for_inference(self, src_tokens, src_lengths):
    #     return DocLanguagePairDataset(src_tokens, src_lengths, self.source_dictionary, doc_heads)

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochSplitAndBatchIterator: a batched iterator over different
                dasaset splits for every epoch.
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        # For default fairseq task, return same iterator across epochs
        # as datasets are not dynamic, can be overridden in task specific
        # setting.
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # don't get indices ordered by example size as for standard translation

        # filter examples that are too large
        throaway_indices = dataset.ordered_indices()
        if max_positions is not None:
            old_indices = throaway_indices
            throaway_indices = self.filter_indices_by_size(
                indices=throaway_indices,
                dataset=dataset,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
            )
            dataset.exclude_indices = np.array(
                list(set(old_indices) - set(throaway_indices))
            )

        # return a reusable, sharded iterator
        # EpochSplitAndBatchIterator instead of EpochBatchIterator
        epoch_iter = iterators.EpochSplitAndBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter
