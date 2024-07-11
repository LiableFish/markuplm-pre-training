from enum import Enum

import torch


class DirectedNodeRelation(int, Enum):
    SELF = 0
    PARENT = 1
    CHILD = 2
    SIBLING = 3
    ANCESTOR = 4
    DESCENDANT = 5
    OTHERS = 6


def get_first_xpath_token_mask(
    *,
    xpath_tags_seq: torch.LongTensor, # [batch, seq_len, max_width],
    xpath_subs_seq: torch.LongTensor, # [batch, seq_len, max_width],
    tags_pad_id: int,
    subs_pad_id: int,
) -> torch.BoolTensor: # [batch_size, seq_len]
    batch_size, _, max_width = xpath_tags_seq.shape
    shifted_xpath_tags_seq = torch.cat(
        [
            torch.full((batch_size, 1, max_width), fill_value=tags_pad_id, dtype=xpath_tags_seq.dtype, device=xpath_tags_seq.device),
            xpath_tags_seq[:, :-1],
        ],
        dim=1,
    )
    shifted_xpath_subs_seq = torch.cat(
        [
            torch.full((batch_size, 1, max_width), fill_value=subs_pad_id, dtype=xpath_subs_seq.dtype, device=xpath_subs_seq.device),
            xpath_subs_seq[:, :-1],
        ],
        dim=1,
    )

    mask = (xpath_tags_seq != shifted_xpath_tags_seq).any(dim=-1) | (xpath_subs_seq != shifted_xpath_subs_seq).any(dim=-1)
    mask[:, 0] = True
    return mask

def get_directed_node_relation(
    *,
    xpath_tags_seq: torch.LongTensor, # [batch, seq_len, max_width]
    xpath_subs_seq: torch.LongTensor, # [batch, seq_len, max_width]
    tags_pad_id: int,
    subs_pad_id: int,
) -> torch.LongTensor:  # [batch_size, seq_len, seq_len]
    batch_size, seq_len, _ = xpath_tags_seq.shape

    labels = torch.full(
        (batch_size, seq_len, seq_len), 
        fill_value=DirectedNodeRelation.OTHERS, 
        dtype=torch.int64,
        device=xpath_tags_seq.device,
    )

    first_tags_true_length, second_tags_true_length, first_equals_second_tags_true_lengths = _get_true_node_pair_lengths_and_longest_prefix(
        xpath_tags_seq, pad_id=tags_pad_id,
    )

    first_subs_true_length, second_subs_true_length, first_equals_second_subs_true_lengths = _get_true_node_pair_lengths_and_longest_prefix(
        xpath_subs_seq, pad_id=subs_pad_id,
    )

    self_mask = (
        _get_self_node_mask(first_tags_true_length, second_tags_true_length, first_equals_second_tags_true_lengths)
        & _get_self_node_mask(first_subs_true_length, second_subs_true_length, first_equals_second_subs_true_lengths)
    )
    labels[self_mask] = DirectedNodeRelation.SELF

    sibling_mask = (
        (
            _get_self_node_mask(first_tags_true_length, second_tags_true_length, first_equals_second_tags_true_lengths)
            | _get_sibling_node_mask(first_tags_true_length, second_tags_true_length, first_equals_second_tags_true_lengths)
        )
        & _get_sibling_node_mask(first_subs_true_length, second_subs_true_length, first_equals_second_subs_true_lengths)
    )
    labels[sibling_mask] = DirectedNodeRelation.SIBLING

    ancestor_mask = (
        _get_ancestor_node_mask(first_tags_true_length, second_tags_true_length, first_equals_second_tags_true_lengths)
        & _get_ancestor_node_mask(first_subs_true_length, second_subs_true_length, first_equals_second_subs_true_lengths)
    )
    labels[ancestor_mask] = DirectedNodeRelation.ANCESTOR

    parent_mask = (
        _get_parent_node_mask(first_tags_true_length, second_tags_true_length, first_equals_second_tags_true_lengths)
        & _get_parent_node_mask(first_subs_true_length, second_subs_true_length, first_equals_second_subs_true_lengths)
    )
    labels[parent_mask] = DirectedNodeRelation.PARENT

    descendant_mask = (
        _get_descendant_node_mask(first_tags_true_length, second_tags_true_length, first_equals_second_tags_true_lengths)
        & _get_descendant_node_mask(first_subs_true_length, second_subs_true_length, first_equals_second_subs_true_lengths)
    )
    labels[descendant_mask] = DirectedNodeRelation.DESCENDANT

    child_mask  = (
        _get_child_node_mask(first_tags_true_length, second_tags_true_length, first_equals_second_tags_true_lengths)
        & _get_child_node_mask(first_subs_true_length, second_subs_true_length, first_equals_second_subs_true_lengths)
    )
    labels[child_mask] = DirectedNodeRelation.CHILD

    return labels


def _get_true_node_pair_lengths_and_longest_prefix(
    seq: torch.LongTensor,
    pad_id: int,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    batch_size, seq_len, width = seq.shape

    first_seq = seq.unsqueeze(2).expand(batch_size, seq_len, seq_len, width)
    second_seq = seq.unsqueeze(1).expand(batch_size, seq_len, seq_len, width)

    first_true_lengths = (first_seq != pad_id).sum(dim=-1)
    second_true_lengths = (second_seq != pad_id).sum(dim=-1)

    diff = (first_seq != second_seq).to(dtype=torch.long)
    
    longest_true_prefix = torch.argmax(diff, dim=-1)

    # If there's no difference, torch.argmax will return 0, which might be incorrect.
    # In this case we need to set `longest_true_prefix` as either `first_true_lengths` or `second_true_lengths`.
    no_diff = (diff == 0).all(dim=-1)
    assert (first_true_lengths[no_diff] == second_true_lengths[no_diff]).all()
    longest_true_prefix = torch.where(no_diff, first_true_lengths, longest_true_prefix)
    
    return first_true_lengths, second_true_lengths, longest_true_prefix


def _get_self_node_mask(
    first_true_lengths: torch.LongTensor,
    second_true_lengths: torch.LongTensor,
    longest_true_prefix: torch.LongTensor,
) -> torch.BoolTensor:
    return (first_true_lengths == second_true_lengths) & (longest_true_prefix == first_true_lengths)


def _get_sibling_node_mask(
    first_true_lengths: torch.LongTensor,
    second_true_lengths: torch.LongTensor,
    longest_true_prefix: torch.LongTensor,
) -> torch.BoolTensor:
    return (first_true_lengths == second_true_lengths) & (longest_true_prefix == first_true_lengths - 1)


def _get_ancestor_node_mask(
    first_true_lengths: torch.LongTensor,
    second_true_lengths: torch.LongTensor,
    longest_true_prefix: torch.LongTensor,
) -> torch.BoolTensor:
    return (first_true_lengths < second_true_lengths) & (longest_true_prefix == first_true_lengths)


def _get_parent_node_mask(
    first_true_lengths: torch.LongTensor,
    second_true_lengths: torch.LongTensor,
    longest_true_prefix: torch.LongTensor,
) -> torch.BoolTensor:
    ancestor_mask = _get_ancestor_node_mask(first_true_lengths, second_true_lengths, longest_true_prefix)
    return ancestor_mask & (first_true_lengths + 1 == second_true_lengths)


def _get_descendant_node_mask(
    first_true_lengths: torch.LongTensor,
    second_true_lengths: torch.LongTensor,
    longest_true_prefix: torch.LongTensor,
) -> torch.BoolTensor:
    return (first_true_lengths > second_true_lengths) & (longest_true_prefix == second_true_lengths)


def _get_child_node_mask(
    first_true_lengths: torch.LongTensor,
    second_true_lengths: torch.LongTensor,
    longest_true_prefix: torch.LongTensor,
) -> torch.BoolTensor:
    descendant_mask = _get_descendant_node_mask(first_true_lengths, second_true_lengths, longest_true_prefix)
    return descendant_mask & (first_true_lengths == second_true_lengths + 1)
