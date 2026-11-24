#!/usr/bin/env bash
# Run bench_modes.py in a container on two GPUs of a node, with this branch applied to
# the image's sglang (see README.md). Refuses GPUs another container holds, and stops
# its container as soon as another one takes them (shared nodes).
#
#   IMAGE=<sglang ROCm image> WEIGHTS=<DeepSeek-V4.1-Flash dir> PROMPT=<text file> \
#   HARNESS=<dir with shadow.py and record/task.py> TREE=<FlyDSL tree for "tuned"> \
#   OUT=<empty output dir> GPUS="renderD176 renderD184" STAGE=smoke|bench \
#   benchmark/kernel_profiles/run_container.sh
#
# Optional: PROFILES (file in this directory, default profiles_resident.json),
# RECIPE (default recipe.dsv41-flash-tp2.json), KP_MAX_BS (2 for the smoke),
# NAME (container name, default kp-bench), APPLY=0 (the image already has the branch).
set -euo pipefail
: "${IMAGE:?}" "${WEIGHTS:?}" "${PROMPT:?}" "${HARNESS:?}" "${TREE:?}" "${OUT:?}" "${GPUS:?}"
STAGE=${STAGE:-bench}
PROFILES=${PROFILES:-profiles_resident.json}
RECIPE=${RECIPE:-recipe.dsv41-flash-tp2.json}
NAME=${NAME:-kp-bench}
APPLY=${APPLY:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BASE=47361c5  # the upstream commit the branch is built on

holders() {
  for c in $(docker ps -q); do
    docker inspect --format '{{.Name}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$c"
  done | grep -v "^/$NAME " | grep -E "($(echo $GPUS | sed 's/ /|/g'))( |$)" || true
}
busy=$(holders)
[ -n "$busy" ] && { echo "GPUs in use: $busy"; exit 5; }

mkdir -p "$OUT/kp" "$OUT/wf"
cp -r "$HERE/kpkg" "$HERE/loader" "$HERE/bench_modes.py" "$OUT/kp/"
cp "$HERE/$PROFILES" "$OUT/kp/profiles.json"
cp "$HARNESS/shadow.py" "$OUT/kp/"
cp "$HERE/$RECIPE" "$OUT/wf/recipe.json"
cp "$PROMPT" "$OUT/wf/prompt.txt"
# field_order.py is excluded: older images order the fields differently, so its two
# names are inserted in place below.
git -C "$REPO" diff "$BASE" HEAD --output="$OUT/kp.patch" -- python/ \
  ':!python/sglang/srt/arg_groups/field_order.py'

devices=$(for g in $GPUS; do printf -- '--device /dev/dri/%s ' "$g"; done)
docker run -d --name "$NAME" --label kp-bench=1 --device /dev/kfd $devices \
  --group-add video --group-add render --shm-size 64g --security-opt seccomp=unconfined \
  -v "$WEIGHTS":/work/models/dsv41:ro -v "$OUT/wf":/wf:ro -v "$HARNESS":/lean:ro \
  -v "$OUT/kp":/kp:ro -v "$TREE":/trees/ab-w4:ro -v "$OUT":/out \
  -e PYTHONPATH=/kp/loader -e KP_OUT=/out -e PYTHONDONTWRITEBYTECODE=1 \
  -e KP_STAGE="$STAGE" -e KP_APPLY="$APPLY" ${KP_MAX_BS:+-e KP_MAX_BS=$KP_MAX_BS} \
  --entrypoint bash "$IMAGE" -c '
cd /sgl-workspace/sglang
if [ "$KP_APPLY" = 1 ]; then
  git apply --whitespace=nowarn /out/kp.patch > /out/apply.log 2>&1 || { echo APPLY_FAILED >> /out/apply.log; exit 3; }
  /opt/venv/bin/python - >> /out/apply.log 2>&1 <<"EOF"
from pathlib import Path
p = Path("python/sglang/srt/arg_groups/field_order.py")
text = p.read_text()
start = text.index("POSITIONAL_FIELD_ORDER = (")
end = text.index("\n)", start)
add = "".join(f"\n    \"{n}\"," for n in ("enable_forward_hot_reload", "kernel_profiles") if f"\"{n}\"" not in text[start:end])
p.write_text(text[:end] + add + text[end:])
print("field_order: added", add.split())
if not Path("python/sglang/srt/managers/scheduler_components/pause_state.py").exists():
    # Older schedulers keep the pause in _engine_paused rather than a PauseState.
    p = Path("python/sglang/srt/managers/scheduler_components/kernel_profiles.py")
    text = p.read_text()
    old = """    pause_state = getattr(scheduler, "pause_state", None)
    return bool(pause_state is not None and pause_state.paused)"""
    assert text.count(old) == 1, "pause check not found"
    p.write_text(text.replace(old, "    return bool(scheduler._engine_paused)"))
    print("pause check: _engine_paused")
EOF
fi
/opt/venv/bin/python -c "import sglang.srt.server_args" >> /out/apply.log 2>&1 || { echo IMPORT_FAILED >> /out/apply.log; exit 4; }
exec /opt/venv/bin/python /kp/bench_modes.py > /out/driver.log 2>&1'

# The guard outlives this shell: it removes the container if another one takes its GPUs.
cat > "$OUT/guard.sh" <<GUARD
while docker ps -q --filter name=^$NAME\$ | grep -q .; do
  other=\$(for c in \$(docker ps -q); do docker inspect --format '{{.Name}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' \$c; done | grep -v '^/$NAME ' | grep -E '($(echo $GPUS | sed 's/ /|/g'))( |\$)')
  if [ -n "\$other" ]; then
    echo "\$(date -u +%FT%TZ) stopping $NAME: \$other" >> "$OUT/guard.log"
    docker rm -f $NAME >> "$OUT/guard.log" 2>&1
    break
  fi
  sleep 5
done
echo "\$(date -u +%FT%TZ) guard exits" >> "$OUT/guard.log"
GUARD
setsid nohup bash "$OUT/guard.sh" > /dev/null 2>&1 < /dev/null &
sleep 20
cat "$OUT/apply.log" 2>/dev/null || true
docker ps -a --filter name="^$NAME\$" --format '{{.Names}} {{.Status}}'
echo "follow: python3 $HERE/summarize.py $OUT/events.jsonl"
