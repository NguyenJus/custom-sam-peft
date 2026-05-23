"""Per-patch modules — one file per patch, alphabetical order.

Populated by Task 5.7. Each module exposes ``apply(model, runtime) -> None``.
``Sam3Patches.apply`` iterates ``_ALL_PATCHES`` in this deterministic order.
"""

from custom_sam_peft.models._patches import (
    addmm_act_grad_safe,
    encode_prompt_dtype,
    forward_grounding_skip_matching,
    mha_input_dtype,
    module_input_dtype,
    pos_enc_dtype,
    roi_align_dtype,
    text_pool_dtype,
)

_ALL_PATCHES = [
    addmm_act_grad_safe.apply,
    encode_prompt_dtype.apply,
    forward_grounding_skip_matching.apply,
    mha_input_dtype.apply,
    module_input_dtype.apply,
    pos_enc_dtype.apply,
    roi_align_dtype.apply,
    text_pool_dtype.apply,
]
