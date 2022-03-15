import logging
import sys
import ast
from typing import Optional
from distutils.util import strtobool
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils, checkpoint_utils
from fairseq.models import (
    FairseqEncoder, FairseqEncoderDecoderModel, BaseFairseqModel,
    register_model, register_model_architecture
)
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    transformer_iwslt_fr_en,
    transformer_vaswani_wmt_en_fr,
    transformer_vaswani_wmt_en_de_big
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import MultiheadAttention, LayerNorm, SinusoidalPositionalEmbedding

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@register_model("han_transformer")
class HanTransformerModel(FairseqEncoderDecoderModel):
    """
    See `"Document-Level Neural Machine Translation with
    Hierarchical Attention Networks" (Miculicich, et al, 2018)
    <https://www.aclweb.org/anthology/D18-1325/>`_.
    """
    def __init__(self, args, encoder, decoder, cache):
        super().__init__(encoder, decoder)
        self.freeze_transfo_params=args.freeze_transfo_params
        self.was_training = False

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--n-context-sents',
            type=int,
            metavar='N',
            default=3,
            help='Number of past sentences to use as context'
        )
        parser.add_argument(
            '--max-context-sents',
            type=int,
            metavar='N',
            default=3,
            help='Maximum number of past sentences allowed by the model'
        )
        parser.add_argument(
            '--han-heads',
            type=int,
            metavar='N',
            help='Num of word-level attention heads'
        )
        parser.add_argument(
            "--pretrained-transformer-checkpoint",
            type=str,
            metavar="STR",
            default=None,
            help="Transformer encoder-decoder model to use for initializing \
                sentence-level parameters",
        )
        parser.add_argument(
            '--freeze-transfo-params',
            action='store_true',
            default=False,
            help=
            'Freeze pretrained weights and disable dropout during training'
        )
        parser.add_argument(
            '--use-segment-embs',
            action='store_true',
            default=False,
            help='Enable distance embeddings for context and current segments.'
        )
        parser.add_argument(
            '--lrn-segment-embs',
            type=lambda x:bool(strtobool(x)),
            default=False,
            help='Use learned embeddings for context and current segments, \
                instead of sinusoidal embeddings.'
        )

        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # transform str args that could be "None" or "False" to None or False
        # this is a turnaround for passing params through bash scripts
        pc = getattr(args, "pretrained_transformer_checkpoint", None)
        pc = None if pc=='None' else pc
        args.pretrained_transformer_checkpoint = pc

        # set any default arguments
        han_base_architecture(args)

        # build sentence-level transformer model
        transformer_model = cls.build_transformer_model(args, task)

        # build cache
        cache = HiddenStatesCache(cache_size=int(args.n_context_sents))

        # build hierarchical encoder
        encoder = cls.build_han_encoder(args, transformer_model.encoder, cache)

        return cls(args, encoder, transformer_model.decoder, cache)

    @classmethod
    def build_transformer_model(cls, args, task):

        model = TransformerModel.build_model(args, task)
        state_dict = model.state_dict()

        # load pre-trained transformer if available
        if  getattr(args, "pretrained_transformer_checkpoint", None):
            # load pre-trained model on cpu
            pretrain_state = checkpoint_utils.load_checkpoint_to_cpu(
                args.pretrained_transformer_checkpoint
                )
            pretrain_state_dict = pretrain_state["model"]
            # extract relevant pre-trained params
            for key in pretrain_state_dict.keys():
                for search_key in [
                    "embed_tokens", "embed_positions", "layers",
                    "output_projection"
                ]:
                    if search_key in key:
                        # for every key match, copy params
                        state_dict[key] = pretrain_state_dict[key]
            # load new state_dict to model
            model.load_state_dict(state_dict, strict=True)
            logger.info(
                'loaded pre-trained Transformer model from {}'.format(
                    args.pretrained_transformer_checkpoint
                )
            )
        if getattr(args, "freeze_transfo_params", False):
            for param in model.parameters():
                param.requires_grad = False
        
        return model

    @classmethod
    def build_han_encoder(cls, args, transformer_encoder, cache):
        return HanEncoder(args, transformer_encoder, cache)

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, doc_heads, id,
        sort_order
    ):  
        # when performing validation before the end of an epoch,
        # we might have a batch that is the continuation of a document but
        # don't have its history in cache because there was a dev step before.
        # If this is the case, we add a fake head to pretend that there is no
        # history because we concluded a document in the previous training step
        if not self.was_training and self.training:
            fake_head = utils.sort_back(id, sort_order)[0]
            if fake_head not in doc_heads:
                logger.info('adding fake head!')
                doc_heads = torch.cat((doc_heads, fake_head.unsqueeze(0)), dim=0)
            
        self.was_training = self.training

        if self.freeze_transfo_params:
            self.encoder.transformer_encoder.eval()
            self.decoder.eval()

        # -> T x B x C
        encoder_out = self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            doc_heads=doc_heads,
            id=id,
            sort_order=sort_order
        )

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths
        )

        if self.was_training and self.freeze_transfo_params:
            self.encoder.transformer_encoder.train()
            self.decoder.train()

        return decoder_out


class HanEncoder(FairseqEncoder):
    def __init__(self, args, transformer_encoder, cache):
        super().__init__(transformer_encoder.dictionary)
        self.cache = cache
        self.transformer_encoder = transformer_encoder
        self.K = int(args.n_context_sents)
        self.max_K = int(args.max_context_sents)
        self.embed_dim = args.encoder_embed_dim
        self.dropout = args.dropout
        self.normalize_before = args.encoder_normalize_before
        self.han_heads = args.han_heads
        self.attention_dropout = args.attention_dropout
        logger.info(
            "past context sentences modeled: {} ".format(self.K)
        )

        # segment embeddings
        if args.use_segment_embs: 
            if args.lrn_segment_embs:
                logger.info("Learning segment embeddings.")
                self.segment_embs = nn.Embedding(
                    self.max_K+1, self.embed_dim, padding_idx=None
                )
                self.register_buffer('segment_ids', torch.arange(self.max_K+1))
            else:
                logger.info("Using sinusoidal segment embeddings.")
                self.register_buffer(
                    'segment_embs',
                    SinusoidalPositionalEmbedding.get_embedding(
                        num_embeddings=self.max_K+1,
                        embedding_dim=self.embed_dim,
                        padding_idx=None,
                    )
                )

        # layer norms
        self.layer_norm_word_level = LayerNorm(self.embed_dim)
        self.layer_norm_sentence_level = LayerNorm(self.embed_dim)
        self.layer_norm_fc = LayerNorm(self.embed_dim)
        self.layer_norm_final = LayerNorm(self.embed_dim)

        # hierarchical attention
        self.word_attn = self.build_word_attention(self.embed_dim)
        self.sent_attn = self.build_sent_attention(self.embed_dim)

        # final position-wise FFNN
        self.fc1 = self.build_fc1(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = self.build_fc2(args.encoder_ffn_embed_dim, self.embed_dim)
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)

        # gate functions
        self.linear = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.sigmoid = nn.Sigmoid()

    def build_word_attention(self, embed_dim):
        return MultiheadAttention(
            embed_dim,
            self.han_heads,
            dropout=self.attention_dropout,
            self_attention=False
        )

    def build_sent_attention(self, embed_dim):
        return MultiheadAttention(
            embed_dim,
            self.han_heads,
            dropout=self.attention_dropout,
            self_attention=False
        )

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def forward(self, src_tokens, src_lengths, doc_heads, id, sort_order):

        ### Encode batch, build context and update cache ######################

        encoder_out = self.transformer_encoder(
            src_tokens, src_lengths=src_lengths
        ) # T x B x C

        # use cache only if not empty
        dummy_only = True if self.cache.id is None else False

        # expand input with cached values
        expanded_id = self.expand_id(id, dummy_only) # B -> 1 + K + B
        expanded_encoder_out = self.expand_h(
            encoder_out.encoder_out, dummy_only
        ) # T x B x C -> T' x (1 + K + B) x C
        expanded_encoder_padding_mask = self.expand_h_padding_mask(
            encoder_out.encoder_padding_mask, dummy_only
        ) # B x T -> (1 + K + B) x T'

        # for every encoded sentence in the batch,
        # get the position of context of padding in
        # expanded_encoder_out and expanded_encoder_padding_mask
        where_context = self.get_context_position(
            id=id, expanded_id=expanded_id, doc_heads=doc_heads
        ) # K x B

        # update cache
        self.cache.update_cache(
            id=id,
            h=encoder_out.encoder_out,
            h_padding_mask=encoder_out.encoder_padding_mask,
            sort_order=sort_order,
            doc_heads=doc_heads
        )
    
        ### Hierarchical encoding #############################################       

        # word-level attention
        query_word = encoder_out.encoder_out # T x B x C
        w = encoder_out.encoder_out.new_empty(
            encoder_out.encoder_out.shape + (self.K, )
        ) # T x K x B x C
        for k in range(self.K):
            # kth sentence in the context
            context_k = expanded_encoder_out[:, where_context[k], :] # T' x B x C
            # padding mask for the kth context sentence
            padding_mask_k = expanded_encoder_padding_mask[where_context[k], :] # B x T'
            # word attention over kth context sentence
            w_attn_k, _ = self.word_attn(
                query=query_word,
                key=context_k,
                value=context_k,
                key_padding_mask=padding_mask_k
            ) # T x B x C
            # Note: some w_attn_k are word representations
            # contextualized over a dummy context.
            # Hence, they have to be masked in the sentence-level attention.

            # layer norm
            w[..., k] = self.layer_norm_word_level(w_attn_k) # T x K x B x C
            if hasattr(self, 'segment_embs'):
                if hasattr(self, 'segment_ids'):
                    # add learned segment embeddings
                    w[..., k] = w[..., k] + self.segment_embs(self.segment_ids[k+1]) # T x K x B x C
                else:
                    # add sinusoidal segment embeddings
                    w[..., k] = w[..., k] + self.segment_embs[k+1] # T x K x B x C
                   

        w = w.permute(0, 3, 1, 2) # T x B x C x K -> T x K x B x C

        # if self.K > 1:
        # sentence-level attention
        query_sent = encoder_out.encoder_out # T x B x C
        if hasattr(self, 'segment_embs'):
            if hasattr(self, 'segment_ids'):
                # add learned segment embeddings
                query_sent = query_sent + self.segment_embs(self.segment_ids[0]) # T x B x C
            else:
                # add sinusoidal segment embeddings
                query_sent = query_sent + self.segment_embs[0] # T x B x C
                
        s = w.new_empty(query_sent.shape)
        # Mask dummy keys with 1e-8 so that if all the K keys are dummies,
        # the result of the softmax is not NaN but 1/K. -> K x B
        mask = where_context.float()
        mask[where_context != 0] = 0
        mask[where_context == 0] = -1e8
        mask = mask.T # K x B -> B x K
        attn_mask = mask.repeat_interleave(
            repeats=self.han_heads, dim=0).unsqueeze(1) # (H x B) x 1 x K
        # Looping over word position: some query words in the batch are pads,
        # some sentence-level word representations (keys) are dummies.
        for t, word in enumerate(query_sent):
            s[t, ...], _ = self.sent_attn(
                query=word.unsqueeze(0),  # 1 x B x C
                key=w[t, ...],  # K x B x C
                value=w[t, ...], # K x B x C
                attn_mask=attn_mask # (H x B) x 1 x K
            )
        s = self.layer_norm_sentence_level(s) # T x B x C

        # position-wise feed forward
        residual = s
        s = self.activation_fn(self.fc1(s)) # T x B x C
        s = F.dropout(
            s, p=float(self.activation_dropout), training=self.training
        ) # T x B x C
        s = self.fc2(s) # T x B x C
        s = F.dropout(s, p=self.dropout, training=self.training) # T x B x C
        s = s + residual # T x B x C
        s = self.layer_norm_fc(s) # T x B x C

        ### Gate encodings ###################################################

        weight = self.linear(torch.cat([encoder_out.encoder_out, s], dim=2)) # T x B x C
        # assign 0 weight to sentences that did not have any context.
        # tomask = ~torch.any(mask == 0, dim=1)
        # weight[:, tomask, :] = float("-inf")
        weight = self.sigmoid(weight) # T x B x C
        out = (1 - weight) * encoder_out.encoder_out + weight * s # T x B x C
        # last layernorm
        out = self.layer_norm_final(out) # T x B x C

        return EncoderOut(
            encoder_out=out,  # T x B x C
            encoder_padding_mask=encoder_out.encoder_padding_mask,  # B x T
            encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
            encoder_states=encoder_out.encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    def expand_id(self, id, dummy_only):
        if not dummy_only:
            # expand by prepending cache
            expanded_id = torch.cat((self.cache.id, id), dim=0)
        else:
            expanded_id = id
        # expand by prepending dummy index
        dummy_id = id.new_tensor([-1])
        expanded_id = torch.cat((dummy_id, expanded_id), dim=0)
        return expanded_id

    def expand_h(self, h, dummy_only):

        if not dummy_only:
            # expand by prepending cache
            self.cache.h, h, _ = utils.pad_smaller(
                self.cache.h, h, dim=0, value=0
            )
            expanded_h = torch.cat((self.cache.h, h), dim=1)
        else:
            expanded_h = h
        # expand by prepending dummy context
        dummy_context = expanded_h.new_zeros(
            (expanded_h.shape[0], 1, self.embed_dim)
        )
        expanded_h = torch.cat((dummy_context, expanded_h), dim=1)
        return expanded_h

    def expand_h_padding_mask(self, h_padding_mask, dummy_only):
        if not dummy_only:
            # expand by prepending cache
            self.cache.h_padding_mask, h_padding_mask, _ = utils.pad_smaller(
                self.cache.h_padding_mask, h_padding_mask, dim=1, value=1
            )
            expanded_h_padding_mask = torch.cat(
                (self.cache.h_padding_mask, h_padding_mask), dim=0
            )
        else:
            expanded_h_padding_mask = h_padding_mask
        # expand by prepending the padding mask for dummy context (no padding)
        dummy_padding = expanded_h_padding_mask.new_zeros(
            (1, expanded_h_padding_mask.shape[-1])
        )
        expanded_h_padding_mask = torch.cat(
            (dummy_padding, expanded_h_padding_mask), dim=0
        )

        return expanded_h_padding_mask

    def get_context_position(self, id, expanded_id, doc_heads):

        # retrieve context positions
        where_no_context = id.new_tensor([])
        where_context = []
        for k in range(1, self.K + 1):
            # select ids of sentences that do not have kth context
            no_context_curr = torch.where(id[:, None] == doc_heads + (k - 1))[0]
            where_no_context = torch.cat((where_no_context, no_context_curr))
            # id of kth context sentences
            idk = (id - k)
            idk[where_no_context] = -1
            # retrieve position of the kth context sentence
            wc = torch.where(idk[:, None] == expanded_id)[1]
            where_context.append(wc)

        return torch.stack(where_context)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        return self.transformer_encoder.reorder_encoder_out(
            encoder_out, new_order
        )

class HiddenStatesCache(nn.Module):
    def __init__(self, cache_size):
        super().__init__()
        self.register_buffer('id', None)
        self.register_buffer('h', None)
        self.register_buffer('h_padding_mask', None)
        self.K = cache_size

    def reset(self):
        self.id = None
        self.h = None
        self.h_padding_mask = None

    def update_cache(self, id, h, h_padding_mask, sort_order, doc_heads):

        # sort id in the original order; retrieve K last ones
        cid = utils.sort_back(id, sort_order)[-self.K:]
        # check whether some ids correspond to document heads
        where_heads = torch.where(cid[:, None] == doc_heads)[0]

        if where_heads.shape[0] > 0:
            # only save to cache from the last head on
            cid = cid[where_heads[-1]:]

        if cid[:, None] in (doc_heads - 1):
            # empty cache if the first sentence of the next batch is an head
            # note: this will not work for the last batch #TODO(lo)
            cid = None

        if cid is not None:
            # update cached id
            self.id = cid.detach()
            # update cached hidden states
            self.h = h[:, torch.where(self.id[:, None] == id)[1], :].detach()
            self.h_padding_mask = h_padding_mask[
                torch.where(self.id[:, None] == id)[1], :].detach()
        else:
            self.reset()



@register_model_architecture("han_transformer", "han_transformer_test")
def han_transformer_test(args):
    # han args
    args.han_heads = getattr(args, "han_heads", 2)
    # transformer args
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 100)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 100)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(
        args, "decoder_embed_dim", args.encoder_embed_dim
    )
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    
    base_architecture(args)

@register_model_architecture("han_transformer", "han_transformer")
def han_base_architecture(args):
    # han args
    args.han_heads = getattr(args, "han_heads", 8)
    # transformer args
    base_architecture(args)

@register_model_architecture("han_transformer", "han_transformer_iwslt_fr_en")
def han_transformer_iwslt_fr_en(args):
    # transformer args
    transformer_iwslt_fr_en(args)
    # han args
    args.han_heads = getattr(args, "han_heads", args.encoder_attention_heads)

@register_model_architecture("han_transformer", "han_transformer_iwslt_wmt_en_fr")
def han_transformer_iwslt_wmt_en_fr(args):
    # han args
    args.han_heads = getattr(args, "han_heads", 8)
    # transformer args
    transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("han_transformer", "han_transformer_wmt_en_fr")
def han_transformer_iwslt_wmt_en_fr(args):
    # han args
    args.han_heads = getattr(args, "han_heads", 8)
    # transformer args
    transformer_vaswani_wmt_en_fr(args)