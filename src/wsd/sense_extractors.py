from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, DebertaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F

from transformers.models.deberta.modeling_deberta import (
    # DebertaEmbeddings,
    DebertaEncoder,
    DebertaLayerNorm,
    StableDropout,
)


class SenseExtractorOutput(NamedTuple):
    non_masked_logits: torch.Tensor
    prediction_logits: torch.Tensor
    prediction_probs: torch.Tensor
    prediction_markers: torch.Tensor
    gold_labels: torch.Tensor
    attentions: torch.Tensor


class SenseExtractor(ABC, nn.Module):
    evaluation_mode = False

    @staticmethod
    def compute_markers(input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        output_markers = torch.zeros_like(input_ids)
        markers_positions = torch.argmax(logits, dim=-1, keepdim=True)
        output_markers = output_markers.scatter(1, markers_positions, 1.0)
        return output_markers

    @staticmethod
    def mask_logits(logits: torch.Tensor, logits_mask: torch.Tensor) -> torch.Tensor:
        logits_dtype = logits.dtype
        if logits_dtype == torch.float16:
            logits = logits * (1 - logits_mask) - 65500 * logits_mask
        else:
            logits = logits * (1 - logits_mask) - 1e30 * logits_mask
        return logits

    @abstractmethod
    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
        definitions_mask: Optional[torch.Tensor] = None,
        gold_markers: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> SenseExtractorOutput:
        raise NotImplementedError

    @abstractmethod
    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        raise NotImplementedError


class DebertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.output_to_half = False
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


# copy paste from huggingface to add the "relative_positions" in input to the parameters in the "DebertaEncoder"
class ConanDebertaModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config)
        self.z_steps = 0
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        padding_attention_mask=None,
        token_type_ids=None,
        relative_pos=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=padding_attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
            relative_pos=relative_pos,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    return_att=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )


class DebertaPositionalExtractor(SenseExtractor):
    def __init__(self, transformer_model: str, dropout: float, use_definitions_mask: bool, output_attentions: bool):
        super().__init__()
        self.model: ConanDebertaModel = ConanDebertaModel.from_pretrained(transformer_model)
        self.classification_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.model.config.hidden_size, 1, bias=False),
        )
        self.use_definitions_mask = use_definitions_mask
        self.output_attentions = output_attentions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        padding_attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        relative_pos: Optional[torch.Tensor],
    ) -> BaseModelOutput:
        forward_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "padding_attention_mask": padding_attention_mask,
        }

        if token_type_ids is not None:
            forward_params["token_type_ids"] = token_type_ids

        if relative_pos is not None:
            forward_params["relative_pos"] = relative_pos

        model_out = self.model(**forward_params, return_dict=True, output_attentions=self.output_attentions)

        return model_out

    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
        definitions_mask: Optional[torch.Tensor] = None,
        gold_markers: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> SenseExtractorOutput:
        model_out: BaseModelOutput = self.forward(
            input_ids,
            attention_mask,
            padding_attention_mask,
            token_type_ids,
            relative_pos,
        )

        last_hidden_state = model_out.last_hidden_state
        classification_logits = self.classification_head(last_hidden_state).squeeze(-1)

        if self.output_attentions:
            attentions = torch.stack(model_out.attentions, dim=0).transpose(0, 1)  # (bsz, layer, head, seq, seq)
        else:
            attentions = None

        non_masked_logits = classification_logits
        if self.use_definitions_mask:
            classification_logits = self.mask_logits(classification_logits, definitions_mask)

        labels = None
        if gold_markers is not None and not self.evaluation_mode:
            labels = torch.argmax(gold_markers, dim=-1)

        prediction_probs = torch.softmax(classification_logits, dim=-1)
        prediction_markers = self.compute_markers(input_ids, classification_logits)

        return SenseExtractorOutput(
            non_masked_logits=non_masked_logits,
            prediction_logits=classification_logits,
            prediction_probs=prediction_probs,
            prediction_markers=prediction_markers,
            gold_labels=labels,
            attentions=attentions,
        )

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        self.model.resize_token_embeddings(new_num_tokens)
