"""
baselines/fixed_chain_ctx.py
=============================
Fixed Chain + Full Context variant (Table 3 / Ablation ladder).

Variant: Fixed Chain ✓ (Ctx=✓, C-Gate=✗, BT=✗, Nag=5) — Table 3, row 2.

This is identical to fixed_chain.py but explicitly ensures the full
history context buffer is always passed (vs. a truncated or no-ctx variant).
This isolates the contribution of the context buffer alone (§9):

    "If Fixed Chain + Full Ctx closes most of the gap, the primary
     contribution is the context buffer design."

In practice, our Fixed Chain already passes the full context — this variant
is kept separate for exact ablation bookkeeping and Table 3 reproducibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from baselines.fixed_chain import FixedChainBaseline, FixedChainTrace
from data.loader import PlantRecord
from utils.vllm_client import VLLMClient


class FixedChainCtxBaseline(FixedChainBaseline):
    """
    Fixed Sequential Chain with full-history context buffer explicitly enabled.
    Table 3: Fixed Chain + Full Ctx (Ctx=✓, C-Gate=✗, BT=✗, Nag=5).
    """

    VARIANT_NAME = "Fixed Chain + Full Ctx"

    def __init__(
        self,
        client: VLLMClient,
        label_space: Dict[str, List[str]],
        confidence_weights: Optional[Dict[str, int]] = None,
    ):
        super().__init__(client, label_space, confidence_weights)

    def run(self, image_id: str, image_b64: str) -> FixedChainTrace:
        """
        Identical to FixedChainBaseline.run() — full context always passed.
        The parent class already implements this; override only for
        clear variant labelling in traces.
        """
        trace = super().run(image_id, image_b64)
        # Tag trace with variant name for analysis
        trace.__dict__["variant"] = self.VARIANT_NAME
        return trace

    def run_batch(self, records: List[PlantRecord]) -> List[FixedChainTrace]:
        return [self.run(r.image_id, r.image_b64) for r in records]
