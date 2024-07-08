import pytest

from markuplm_pretraining.html_utils import get_directed_node_relation_from_xpaths
from markuplm_pretraining.node_utils import DirectedNodeRelation

@pytest.mark.parametrize(
    "first_xpath, second_xpath, relation",
    [
        ('/html/head/title', '/html/head/title', DirectedNodeRelation.SELF),
        ('/html/body/div/li[1]/div/span[1]', '/html/body/div/li[1]/div/span[1]', DirectedNodeRelation.SELF),
        ('/html/body/div/li[2]', '/html/body/div/li[1]/div/span[1]', DirectedNodeRelation.OTHERS),
        ('/html/body/div/li[1]/div/span[1]', '/html/body/div/li[2]/div/span[1]', DirectedNodeRelation.OTHERS),
        ('/html/body/div/li[1]/div/span[1]', '/html/body/div/li[1]/div/span[2]', DirectedNodeRelation.SIBLING),
        ('/html/body/div/li[1]/div/span[1]', '/html/body/div/li[1]/div/ul', DirectedNodeRelation.SIBLING),
        ('/html/body/div/ul/li/span', '/html/body/div/ul/li', DirectedNodeRelation.CHILD),
        ('/html/body/div/ul/li', '/html/body/div/ul/li/span', DirectedNodeRelation.PARENT),
        ('/html/body/div/li[1]/div', '/html/body/div', DirectedNodeRelation.DESCENDANT),
        ('/html/body/div', '/html/body/div/li[1]/div', DirectedNodeRelation.ANCESTOR),
    ],
)
def test_get_directed_node_relation(first_xpath, second_xpath, relation):
    assert get_directed_node_relation_from_xpaths(first_xpath, second_xpath) == relation
