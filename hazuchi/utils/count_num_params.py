import jax
import chex


def count_num_params(params: chex.PyTreeDef) -> int:
    return sum([v.size for v in jax.tree_leaves(params)])
