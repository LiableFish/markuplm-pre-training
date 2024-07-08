import pytest
import torch
from markuplm_pretraining.node_utils import DirectedNodeRelation, get_directed_node_relation, get_first_xpath_token_mask
from markuplm_pretraining.html_utils import get_directed_node_relation_from_xpaths


def _get_true_node_relations(xpaths: list[str]) -> torch.LongTensor:
    true_relations = torch.full((len(xpaths), len(xpaths)), DirectedNodeRelation.OTHERS)

    for i, first in enumerate(xpaths):
        for j, second in enumerate(xpaths):
            true_relations[i][j] = get_directed_node_relation_from_xpaths(first, second)

    true_relations = true_relations.unsqueeze(0)
    return true_relations

def test_get_directed_node_relation(markuplm_processor, test_html):
    features = markuplm_processor.feature_extractor(test_html)
    xpaths = features["xpaths"][0]
    nodes = ["test"] * len(xpaths) # Ensuring that each node has only one input id

    markuplm_processor.parse_html = False
    encoding = markuplm_processor(nodes=nodes, xpaths=xpaths, return_tensors="pt")
   
    relations = get_directed_node_relation(
        xpath_tags_seq=encoding.xpath_tags_seq,
        xpath_subs_seq=encoding.xpath_subs_seq,
        tags_pad_id=markuplm_processor.tokenizer.pad_tag_id,
        subs_pad_id=markuplm_processor.tokenizer.pad_width,
    )
    relations = relations[:, 1:-1, 1:-1]  # remove special tokens

    assert (relations == _get_true_node_relations(xpaths)).all()

def test_get_directed_node_relation_with_mask(markuplm_processor, test_html):
    features = markuplm_processor.feature_extractor(test_html)
    xpaths = features["xpaths"][0]
    nodes = features["nodes"][0]

    markuplm_processor.parse_html = False
    encoding = markuplm_processor(nodes=nodes, xpaths=xpaths, return_tensors="pt")
   
    relations = get_directed_node_relation(
        xpath_tags_seq=encoding.xpath_tags_seq,
        xpath_subs_seq=encoding.xpath_subs_seq,
        tags_pad_id=markuplm_processor.tokenizer.pad_tag_id,
        subs_pad_id=markuplm_processor.tokenizer.pad_width,
    )
    mask = get_first_xpath_token_mask(
        xpath_tags_seq=encoding.xpath_tags_seq,
        xpath_subs_seq=encoding.xpath_subs_seq,
        tags_pad_id=markuplm_processor.tokenizer.pad_tag_id,
        subs_pad_id=markuplm_processor.tokenizer.pad_width,
    )
    mask = mask.unsqueeze(2) & mask.unsqueeze(1)

    relations = relations[torch.nonzero(mask, as_tuple=True)].reshape(1, len(xpaths) + 2, len(xpaths) + 2)
    relations = relations[:, 1:-1, 1:-1]  # remove special tokens

    assert (relations == _get_true_node_relations(xpaths)).all()
