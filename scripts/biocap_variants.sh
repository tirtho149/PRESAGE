# scripts/biocap_variants.sh
# ============================================================================
# Single source of truth for the BioCAP-on-Bugwood training matrix.
#
# Each variant declares the caption STRATEGY, the projector mode, the
# epoch count, and any data-subset filter. The strategy maps directly
# onto plantswarm.captioning.STRATEGIES. The variant TAG is used as a
# subdir name under data/wds_shards/ and a job name in SLURM.
#
# Used by:
#   scripts/submit_biocap_train.sh     -- single variant submission
#   scripts/submit_biocap_matrix.sh    -- sbatches all variants
#   scripts/aggregate_biocap_tables.py -- collates results by tag
#
# Each variant is a bash array via `eval` indirection to keep the file
# portable to older bashes. Conceptual fields:
#   STRATEGY        plantswarm.captioning strategy
#   PROJ            "dual" | "single"
#   EPOCHS          training epochs
#   SUBSET          "all" | "covered" | "non_covered"
#   PAPER_TABLES    comma-separated paper-table refs (for reporting)
# ============================================================================

# Variant order matters for sbatch array indexing (#SBATCH --array=0-NN).
# Comment-only diff between variants makes the matrix audit-friendly.

BIOCAP_VARIANTS=(
    # Caption-strategy ablation (paper Table 3)
    "T01:label_only:dual:50:all:T3"             # Table 3 row "None"
    "T02:summary_only:dual:50:all:T3"           # KB summary only
    "T03:canonical_full:dual:50:all:T3"         # canonical no-deltas
    "T04:canonical_deltas_3:dual:50:all:T1,T3,T17,T18,T19,T20"   # MAIN METHOD
    # Number-of-deltas ablation (paper Table 6)
    "T05:canonical_deltas_1:dual:50:all:T6"
    "T06:canonical_deltas_5:dual:50:all:T6"
    "T07:canonical_deltas_7:dual:50:all:T6"
    # Training-recipe ablation (paper Figure 3)
    "T08:canonical_deltas_3:single:50:all:Fig3"
    "T09:canonical_deltas_3:dual:100:all:Fig3"
    # KB-coverage ablation (paper Table 4 analog)
    "T10:canonical_deltas_3:dual:50:covered:T4"
    "T11:canonical_deltas_3:dual:50:non_covered:T4"
)

# Helper for shell consumers.
biocap_parse_variant() {
    # Usage: biocap_parse_variant "T04:..."
    # Sets globals: VARIANT_TAG STRATEGY PROJ EPOCHS SUBSET PAPER_TABLES
    local IFS=':'
    read -r VARIANT_TAG STRATEGY PROJ EPOCHS SUBSET PAPER_TABLES <<<"$1"
}
