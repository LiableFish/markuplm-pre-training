from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple
import torch
from transformers import MarkupLMTokenizerFast, BatchEncoding
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, DataCollatorMixin

from markuplm_pretraining.node_utils import DirectedNodeRelation, get_directed_node_relation, get_first_xpath_token_mask

_TITLE = "title"
_BODY = "body"


# TODO: add whole-word masking collator
@dataclass
class DataCollatorForMarkupLMLanguageModelling(DataCollatorMixin):
    """
    Data collator for MarkupLM pre-training. 
    Returns batches suitable for 
    - Masked Markup Language Modeling (MMLM)
    - Node Relationship Prediction (NRP)   
    - Title-Page Matching (TPM)
    """
    tokenizer: MarkupLMTokenizerFast
    mlm_probability: float = 0.15
    title_replace_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    max_node_pairs: int = 1000
    non_others_node_pairs_ratio: float = 0.8

    def __post_init__(self):
        if not isinstance(self.tokenizer, MarkupLMTokenizerFast):
            raise ValueError("Only MarkupLMTokenizerFast is supported")
    
    def tf_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support TensorFlow")
    
    def numpy_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support Numpy")

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(examples[0], Mapping):
            raise ValueError("Providing list of lists is not supported")
        
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
        )

        special_tokens_mask = self._torch_ensure_special_tokens_mask(batch)
        
        batch["input_ids"], batch["mmlm_labels"] = self._torch_mask_tokens(
            batch["input_ids"], batch["xpath_tags_seq"], special_tokens_mask=special_tokens_mask
        )

        batch["nrp_ids"], batch["nrp_labels"] = self._torch_get_directed_node_pairs(
             batch["xpath_tags_seq"], batch["xpath_subs_seq"], special_tokens_mask=special_tokens_mask,
        )

        # Probably need to create it before padding as titles may have different length?
        batch["input_ids"], batch["tpm_labels"] = self._torch_replace_title(
            batch["input_ids"], batch["xpath_tags_seq"], special_tokens_mask=special_tokens_mask
        )

        return batch
    
    def _torch_ensure_special_tokens_mask(self, batch: BatchEncoding) -> torch.LongTensor:
        input_ids = batch["input_ids"]
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=input_ids.device)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        return special_tokens_mask

    def _torch_mask_tokens(
        self, 
        inputs_ids: torch.LongTensor,  # [batch, seq_len]
        xpath_tags_seq: torch.LongTensor,  # [batch, seq_len, max_width]
        *,
        special_tokens_mask: torch.LongTensor, # [batch, seq_len]
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:  # ([batch, seq_len], [batch, seq_len])
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        **Do not** mask tokens in <title> element.
        """
        import torch

        labels = inputs_ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Do not mask special tokens
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Do not mask tokens from <title> element
        title_element_mask = (xpath_tags_seq == self.tokenizer.tags_dict.get(_TITLE)).any(dim=-1)
        probability_matrix.masked_fill_(title_element_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs_ids, labels
    
    def _torch_get_directed_node_pairs(
        self,
        xpath_tags_seq: torch.LongTensor,  # [batch, seq_len, max_width]
        xpath_subs_seq: torch.LongTensor, # [batch, seq_len, max_width]
        *,
        special_tokens_mask: torch.LongTensor, # [batch, seq_len]
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:  #([batch, seq_len, seq_len], [batch, seq_len, seq_len])
        """
        1. Calculate node pairs using code from node_utils
        2. Calculate mask for first xpath token index for each node + false for all special tokens
        3. Sample pairs according to ratio, max_pairs and node labels
        4. Return pair indices and corresponding labels
        """
        batch_size, _, _ = xpath_tags_seq.shape

        # TODO: this can be done on dataset creation level along with tokenization, not on dynamic collator level
        node_relations = get_directed_node_relation(
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            tags_pad_id=self.tokenizer.pad_tag_id,
            subs_pad_id=self.tokenizer.pad_width,
        )  # [batch_size, seq_len, seq_len]

        # TODO: this can be done on dataset creation level along with tokenization, not on dynamic collator level
        first_xpath_token_mask = get_first_xpath_token_mask(
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            tags_pad_id=self.tokenizer.pad_tag_id,
            subs_pad_id=self.tokenizer.pad_width,
        )  # [batch_size, seq_len]

        mask = first_xpath_token_mask & ~special_tokens_mask # [batch_size, seq_len]
        node_relations_mask = mask.unsqueeze(2) & mask.unsqueeze(1) # [batch_size, seq_len, seq_len]

        nrp_ids = []
        # We have to iterate over batch size because the number of "true nodes" is not the same for each batch
        # TODO: try using `torch.bernouli(torch.full(tensor.shape, self.max_node_pairs/tensor.shape[-1])` instead for approx sampling 
        for i in range(batch_size):
            true_node_pair_indices = torch.nonzero(node_relations_mask[i], as_tuple=False)
            true_node_relations = node_relations[i][tuple(true_node_pair_indices.T)]
            n_non_others_pairs_to_sample = int(self.max_node_pairs * self.non_others_node_pairs_ratio)

            sample_non_others_pair_indices = self._sample_indices(
                true_node_pair_indices[true_node_relations != DirectedNodeRelation.OTHERS],
                num_samples=n_non_others_pairs_to_sample,
            )
            sample_others_pair_indices = self._sample_indices(
                true_node_pair_indices[true_node_relations == DirectedNodeRelation.OTHERS],
                num_samples=self.max_node_pairs - n_non_others_pairs_to_sample,
            )
            sample_pair_indices = torch.cat((sample_non_others_pair_indices, sample_others_pair_indices), dim=0)
            sample_pair_indices = torch.cat(
                (
                    torch.full((sample_pair_indices.shape[0], 1), fill_value=i),  # Adding batch id
                    sample_pair_indices
                ),
                dim=1,
            )
            nrp_ids.append(sample_pair_indices)

        nrp_ids = torch.cat(nrp_ids)
        nrp_labels = node_relations[tuple(nrp_ids.T)]
        return nrp_ids, nrp_labels

    @staticmethod    
    def _sample_indices(
        indices: torch.LongTensor,  # [N_examples, N_dim]
        *,
        num_samples: int,
    ) -> torch.LongTensor:  # [N_samples, N_dim]
        n_examples, _ = indices.shape
        return indices[torch.randperm(n_examples)[:min(n_examples, num_samples)]]

    def _torch_replace_title(
        self, 
        inputs_ids: torch.LongTensor,  # [batch, seq_len]
        xpath_tags_seq: torch.LongTensor,  # [batch, seq_len, max_width]
        *,
        special_tokens_mask: torch.LongTensor, # [batch, seq_len]
    ):
        return NotImplemented
    
    