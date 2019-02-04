

# traverse fparser2 nodes
# to stop further traversal, return not None from func 
# The return code will be forwarded to initial caller
# func will collect anything in bag during processing
def traverse(node, func, bag, subnode='items', prerun=True, depth=0):

    ret = None

    if prerun and func is not None:
        ret = func(node, bag, depth)
        if ret is not None: return ret

    if node and hasattr(node, subnode):
        for child in getattr(node, subnode, []):
            ret = traverse(child, func, bag, subnode=subnode, prerun=prerun, depth=depth+1)

    if not prerun and func is not None:
        ret = func(node, bag, depth)
        if ret is not None: return ret

    return ret
