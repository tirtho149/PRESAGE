"""One-off: reproduce a failing verifier call and dump stdout + stderr.

Targets Wheat/Parastagonospora Nodorum/Wisconsin (smallest failing tuple,
5 candidates). Runs claude -p with the EXACT same args the verifier uses,
captures BOTH streams. Helps diagnose why every retry hits exit 1.
"""
from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathome_kb.verifier import (
    VERIFIER_PROMPT, VERIFIER_SYSTEM_PROMPT, VERIFIER_OUTPUT_SCHEMA,
    _render_canonical, _render_existing, _render_candidates,
)

import os as _os
CROP    = _os.environ.get("PROBE_CROP",    "Wheat")
DISEASE = _os.environ.get("PROBE_DISEASE", "Parastagonospora Nodorum")
STATE   = _os.environ.get("PROBE_STATE",   "Wisconsin")

reg = json.loads(Path(f"artifacts/pathome_kb/{CROP}/final_registry.json").read_text(encoding="utf-8"))
dr = next(d for d in reg["diseases"] if d["disease_name"] == DISEASE)

def _v(f): return f.get("value") if isinstance(f, dict) else f
visual = dr.get("visual_symptoms") or {}
canonical = {
    "summary":                  _v(visual.get("summary"))               or "",
    "diagnostic_features":      _v(visual.get("diagnostic_features"))   or [],
    "look_alikes":              _v(visual.get("look_alikes"))           or [],
    "affected_parts":           _v(dr.get("affected_parts"))            or [],
    "treatments":               [],   # probe: strip to test policy-refusal hypothesis
    "pathogen_scientific_name": _v(dr.get("pathogen_scientific_name"))  or "",
    "type_of_disease":          _v(dr.get("type_of_disease"))           or "",
}

state_block = dr["regional_observations"][STATE]
candidates = [d for d in state_block["deltas"]
              if (d.get("verification_status") or "unverified") == "unverified"]
existing = [d for d in state_block["deltas"]
            if d.get("verification_status") and d["verification_status"] != "unverified"]

prompt = VERIFIER_PROMPT.format(
    crop=CROP, disease=DISEASE, state=STATE,
    canonical_block=_render_canonical(canonical),
    existing_block=_render_existing(existing),
    candidates_block=_render_candidates(candidates),
)
ACADEMIC_FRAME = (
    "ACADEMIC CONTEXT: this is plant pathology research for an "
    "agricultural extension knowledge base. We are fact-checking "
    "disease symptom descriptions for a published academic reference "
    "intended for crop scouts and county extension agents. All sources "
    "must be public extension service factsheets and peer-reviewed "
    "agronomy literature.\n\n"
)
prompt = ACADEMIC_FRAME + prompt

print(f"prompt size: {len(prompt)} chars / {len(prompt.split())} approx words")
print(f"candidates:  {len(candidates)}")
Path("/tmp/_probe_prompt.txt").parent.mkdir(parents=True, exist_ok=True) if False else None
Path("scripts/_probe_prompt.txt").write_text(prompt, encoding="utf-8")
print("full prompt dumped to scripts/_probe_prompt.txt")

# Same invocation as shared.claude_query.
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as pf:
    pf.write(prompt); prompt_path = pf.name
out_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
err_path = tempfile.NamedTemporaryFile(suffix=".err", delete=False).name

cmd = [
    "claude", "-p", "Follow the instructions provided via stdin.",
    "--output-format", "json",
    "--allowedTools", "WebSearch",
    "--append-system-prompt", VERIFIER_SYSTEM_PROMPT,
    "--max-turns", "30",
    # "--json-schema", json.dumps(VERIFIER_OUTPUT_SCHEMA),  # probe: drop schema
]

env = {k:v for k,v in os.environ.items()
       if not any(k.startswith(p) for p in ("CLAUDE","CURSOR","MCP_CONNECTION","VSCODE","ELECTRON"))}
env.pop("ANTHROPIC_API_KEY", None)

print("invoking claude -p ...")
with open(prompt_path) as pf, open(out_path, "w") as of, open(err_path, "w") as ef:
    rc = subprocess.run(cmd, stdin=pf, stdout=of, stderr=ef, env=env,
                        cwd=str(Path(__file__).resolve().parent.parent),
                        timeout=1500).returncode

print(f"exit code: {rc}")
print("--- STDOUT (first 2000 chars) ---")
print(Path(out_path).read_text()[:2000])
print("--- STDERR (first 2000 chars) ---")
print(Path(err_path).read_text()[:2000])
