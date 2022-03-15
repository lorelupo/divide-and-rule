# import math
# from typing import Any, Dict, List, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from fairseq import options, utils
# from fairseq.models import (
#     TransformerModel,
#     TransformerEncoder,
#     TransformerDecoder,
#     FairseqEncoder,
#     FairseqEncoderDecoderModel,
#     FairseqIncrementalDecoder,
#     register_model,
#     register_model_architecture,
#     base_architecture
# )
# from fairseq.models.fairseq_encoder import EncoderOut
# from fairseq.modules import (
#     AdaptiveSoftmax,
#     LayerDropModuleList,
#     LayerNorm,
#     MultiheadAttention,
#     PositionalEmbedding,
#     SinusoidalPositionalEmbedding,
#     TransformerDecoderLayer,
#     TransformerEncoderLayer,
# )
# from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
# from torch import Tensor

# @register_model("zhang_transformer")
# class ZhangTransformerModel(TransformerModel):
#     """
#     See `"Improving the Transformer Translation Model
#     with Document-Level Context" (Zhang, et al, 2018)
#     <https://www.aclweb.org/anthology/D18-1049/>`_.
#     """

#     def __init__(self, encoder, encoder_c, decoder, args):
#         super().__init__(encoder, decoder, args)

#     @staticmethod
#     def add_args(parser):
#         """Add model-specific arguments to the parser."""
#         # fmt: off
#         super().add_args(parser)
#         parser.add_argument('--use-context', type=bool, metavar='BOOL',
#                              help='Whether to use the context encoder')
#         # fmt: on
    
#     @classmethod
#     def build_model(cls, args, task):
#         """Build a new model instance."""

#         # set any default arguments
#         zhang_base_architecture(args)

#         # build transformer model
#         transformer_model = TransformerModel.build_model(args, task)

#         return cls(args, transformer_model.encoder, encoder_c, transformer_model.decoder)

# class ZhangEncoder(TransformerEncoder):
#     def __init__(self, args, dictionary, embed_tokens):
#         super().__init__(args, dictionary, embed_tokens)

#     def build_encoder_layer(self, args):
#         return ZhangEncoderLayer(args)

# class ZhangDecoder(TransformerDecoder):
#     def __init__(self, args, dictionary, embed_tokens):
#         super().__init__(args, dictionary, embed_tokens)

#     def build_decoder_layer(self, args):
#         return ZhangDecoderLayer(args)

# class ZhangEncoderLayer(TransformerEncoderLayer):
#     def __init__(self, args):
#         super().__init__(args)

#         # context attention
#         self.context_attn = self.build_context_attention(self.embed_dim, args)
#         self.context_attn_layer_norm = LayerNorm(self.embed_dim)

#     def build_context_attention(self, embed_dim, args):
#         return MultiheadAttention(
#             embed_dim,
#             args.encoder_attention_heads,
#             dropout=args.attention_dropout,
#             self_attention=False
#         )

#     # TODO
#     def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
#         return NotImplementedError


# def zhang_base_architecture(args):
#     args.use_context = getattr(args, "use_context", True)
#     base_architecture(args)

