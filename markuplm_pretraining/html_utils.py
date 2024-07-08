from markuplm_pretraining.node_utils import DirectedNodeRelation


def get_directed_node_relation_from_xpaths(first_xpath: str, second_xpath: str) -> DirectedNodeRelation:
    if first_xpath == second_xpath:
        return DirectedNodeRelation.SELF
    
    first_xpath_parts = first_xpath.split("/")
    second_xpath_parts = second_xpath.split("/")

    if (
        len(first_xpath_parts) == len(second_xpath_parts)
        and first_xpath_parts[:-1] == second_xpath_parts[:-1]
    ):
        return DirectedNodeRelation.SIBLING
    
    if (
        len(first_xpath_parts) < len(second_xpath_parts)
        and first_xpath_parts == second_xpath_parts[:len(first_xpath_parts)]
    ):
        if len(first_xpath_parts) + 1 == len(second_xpath_parts):
            return DirectedNodeRelation.PARENT
        
        return DirectedNodeRelation.ANCESTOR
    
    if (
        len(first_xpath_parts) > len(second_xpath_parts)
        and first_xpath_parts[:len(second_xpath_parts)] == second_xpath_parts
    ):
        if len(first_xpath_parts) == len(second_xpath_parts) + 1:
            return DirectedNodeRelation.CHILD
        
        return DirectedNodeRelation.DESCENDANT
    
    return DirectedNodeRelation.OTHERS
