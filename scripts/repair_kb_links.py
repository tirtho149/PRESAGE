#!/usr/bin/env python3
"""Remove non-working links from the PRESAGE final registry.

Uses artifacts/link_check.json (from verify_kb_links.py). Any URL classified
`dead` (404/410/DNS) or `unknown` (SSL/timeout, unreachable after retries) is
stripped from the KB so every remaining link is verified reachable
(alive 2xx/3xx, or blocked 403/429 = live-but-publisher-bot-blocked).

- Delta web_support: drop bad url entries. If a delta loses ALL web support,
  downgrade verification_status (verified/weakly_supported -> field_observation
  if swarm-supported, else provisional), saving _prior_status + _web_pruned.
- Canonical source urls: clear a dead url (keep the value/quote), mark _url_pruned.

Backs up the whole KB first. Idempotent.
"""
import json, glob, os, tarfile

KB = "artifacts/pathome_kb"
LC = "artifacts/link_check.json"

lc = json.load(open(LC))
BAD = {u for u, v in lc.items() if v.get("status") in ("dead", "unknown")}
print(f"{len(BAD)} non-working URLs to strip (dead+unknown)")

# backup
bak = "artifacts/pathome_kb_backup_pre_linkprune.tgz"
if not os.path.exists(bak):
    with tarfile.open(bak, "w:gz") as t:
        t.add(KB, arcname="pathome_kb")
    print("backup ->", bak)

CANON_FIELDS = ["pathogen_scientific_name", "type_of_disease",
                "affected_parts", "visual_symptoms", "treatments"]

removed_delta = 0
downgraded = 0
removed_canon = 0
files = sorted(glob.glob(f"{KB}/*/final_registry.json"))
for f in files:
    d = json.load(open(f))
    changed = False
    for dz in d.get("diseases", []):
        # canonical urls
        for k in CANON_FIELDS:
            v = dz.get(k)
            if isinstance(v, dict) and v.get("url") in BAD:
                v["_url_pruned"] = v.pop("url")
                v["_url_status"] = lc[v["_url_pruned"]]["status"]
                removed_canon += 1
                changed = True
        # delta web_support
        for st in (dz.get("regional_observations") or {}).values():
            for dl in st.get("deltas", []):
                ws = dl.get("web_support") or []
                if not ws:
                    continue
                keep = [w for w in ws if w.get("url") not in BAD]
                if len(keep) != len(ws):
                    dropped = [w.get("url") for w in ws if w.get("url") in BAD]
                    dl["web_support"] = keep
                    dl.setdefault("_pruned_urls", []).extend(dropped)
                    removed_delta += (len(ws) - len(keep))
                    changed = True
                    if not keep:  # lost all web support -> downgrade
                        prior = dl.get("verification_status")
                        if prior in ("verified", "weakly_supported"):
                            dl["_prior_status"] = prior
                            dl["verification_status"] = (
                                "field_observation" if dl.get("swarm_support") else "provisional")
                            dl["_web_pruned"] = True
                            downgraded += 1
    if changed:
        json.dump(d, open(f, "w"), indent=1, ensure_ascii=False)

print(f"delta links removed: {removed_delta}")
print(f"deltas downgraded (lost all web support): {downgraded}")
print(f"canonical urls cleared: {removed_canon}")
