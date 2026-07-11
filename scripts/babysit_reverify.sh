#!/usr/bin/env bash
# Poll the reverify array; log web-cited + field_observation progress until done.
JID=$(cat /work/mech-ai-scratch/tirtho/PRESAGE/reverify_array_jobid.txt)
cd /work/mech-ai-scratch/tirtho/PRESAGE
counts(){ /work/mech-ai-scratch/tirtho/.venv/bin/python - <<'PY'
import json,glob,collections
c=t=0; s=collections.Counter()
for f in glob.glob('artifacts/pathome_kb/*/final_registry.json'):
    d=json.load(open(f))
    for dd in d.get('diseases',[]):
        for st,obs in (dd.get('regional_observations') or {}).items():
            for x in (obs.get('deltas') if isinstance(obs,dict) else obs) or []:
                if isinstance(x,dict):
                    t+=1; s[(x.get('verification_status') or '?').lower()]+=1
                    if x.get('web_support'): c+=1
print(f"{c}/{t} web-cited | field_obs={s['field_observation']} "
      f"prov={s['provisional']} novel={s['novel_plausible']} unver={s['unverified']}")
PY
}
while squeue -j "$JID" -h -t all 2>/dev/null | grep -qE "R|PD|CG"; do
  run=$(squeue -j "$JID" -h -t R 2>/dev/null | wc -l)
  pd=$(squeue -j "$JID" -h -t PD 2>/dev/null | wc -l)
  done=$(sacct -j "$JID" -n -X --format=State 2>/dev/null | grep -c COMPLETED)
  echo "[$(date +%H:%M)] running=$run pending=$pd completed_tasks=$done | $(counts)"
  sleep 180
done
echo "=== REVERIFY ARRAY DONE $(date) ==="
sacct -j "$JID" -n -X --format=State 2>/dev/null | sort | uniq -c
counts
