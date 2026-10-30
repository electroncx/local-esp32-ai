#!/usr/bin/env bash
# Checks everything a change can break, on the shipped model (firmware/data/model.bin).
#
#   ./test.sh              engine build, routing, every trained phrasing and every
#                          held-out question on the real C engine (a few minutes)
#   ./test.sh --firmware   also builds every firmware target (needs PlatformIO)
#   ./test.sh --qemu PATH  also runs the ESP32 firmware in Espressif's emulator
#                          (PATH = qemu-system-xtensa; implies --firmware)
#   ./test.sh --arduino DIR  also runs the Arduino IDE sketch's build (exported to
#                          DIR, e.g. by arduino-cli --output-dir) in the emulator
#
# Passes only if: the engine builds without warnings; every trained phrasing is
# answered exactly; no held-out question gets a wrong answer (declining with
# "I don't know." is allowed); no answer breaks the directness rules; and the
# firmware builds and answers in the emulator the same way the PC build does.
set -euo pipefail
cd "$(dirname "$0")"
firmware=0 qemu="" arduino=""
while [ $# -gt 0 ]; do
  case "$1" in
    --firmware) firmware=1 ;;
    --qemu) qemu="$2"; firmware=1; shift ;;
    --arduino) arduino="$2"; shift ;;
    *) echo "usage: $0 [--firmware] [--qemu path/to/qemu-system-xtensa] [--arduino build-dir]" >&2; exit 2 ;;
  esac
  shift
done

step() { printf '\n== %s\n' "$*"; }

step "PC engine builds without warnings"
make -C host -B CFLAGS="-O2 -std=c99 -Wall -Wextra -pedantic -Werror"

step "routing: every phrasing reaches its fact, no test question reaches a wrong one"
python3 train/check_routing.py --show 20

step "answers on the C engine: trained phrasings, held-out questions, directness"
python3 train/evaluate.py

step "the Arduino IDE sketch matches the firmware and the model"
python3 firmware/make_arduino.py --check

step "questions run through the model in blocks: same answers as one character at a time"
python3 - <<'EOF'
import random, sys
sys.path.insert(0, "train")
import engine
from common import DEFAULT_TIER, canonical, dropped_of, load_facts
dropped = dropped_of("firmware/data/model.bin")
keys = random.Random(3).sample([canonical(qs) for qs, _ in load_facts(DEFAULT_TIER)
                                if canonical(qs) not in dropped], 300)
one = engine.ask("host/tinyai", "firmware/data/model.bin", keys, "--raw", "--block", "1")
for b in ("5", "65"):
    got = engine.ask("host/tinyai", "firmware/data/model.bin", keys, "--raw", "--block", b)
    same = sum(x == y for x, y in zip(one, got))
    print(f"block {b}: {same}/{len(keys)} identical to block 1")
    if same < len(keys):
        sys.exit(1)
EOF

if [ "$firmware" = 1 ]; then
  step "firmware builds"
  (cd firmware && pio run -e esp32dev -e esp32dev-dio -e esp32-s2 -e esp32-s3 -e esp32-c3)
fi

emulator_check() {  # $1 = qemu, $2 = build folder
  python3 - "$1" "$2" <<'EOF'
import random, subprocess, sys
sys.path.insert(0, "train")
from common import DEFAULT_TIER, canonical, dropped_of, load_eval, load_facts
qemu = sys.argv[1]
held = load_eval(DEFAULT_TIER)
# Trained questions the model itself answers (not refused for lack of room):
# each runs the full float pipeline, which must match the PC bit for bit.
dropped = dropped_of("firmware/data/model.bin")
trained = random.Random(7).sample([(canonical(qs), a) for qs, a in load_facts(DEFAULT_TIER)
                                   if canonical(qs) not in dropped], 200)
rows = held[:150] + trained
pc = subprocess.run(["host/tinyai", "firmware/data/model.bin"], input="\n".join(q for q, _ in rows) + "\n",
                    capture_output=True, text=True, check=True).stdout.split("\n")[:len(rows)]
out = subprocess.run([sys.executable, "firmware/qemu_test.py", "--qemu", qemu, "--build", sys.argv[2]],
                     input="\n".join(q for q, _ in rows) + "\n", capture_output=True, text=True)
print(out.stderr.strip())
if out.returncode:
    sys.exit(f"the emulator run failed (exit {out.returncode}): see above")
got = [l.split("\t", 1)[1] if "\t" in l else "" for l in out.stdout.split("\n")[:len(rows)]]
diff = [(q, p, g) for (q, _), p, g in zip(rows, pc, got) if p != g]
print(f"emulator matches the PC build on {len(rows) - len(diff)}/{len(rows)} questions")
for q, p, g in diff[:10]:
    print(f"    {q!r}: PC {p!r}, ESP32 {g!r}")
sys.exit(1 if diff or len(got) < len(rows) else 0)
EOF
}

if [ -n "$qemu" ]; then
  step "the ESP32 firmware answers like the PC build, in the emulator"
  emulator_check "$qemu" firmware/.pio/build/esp32dev-dio
fi

if [ -n "$arduino" ]; then
  step "the Arduino IDE build answers like the PC build, in the emulator"
  [ -n "$qemu" ] || { echo "--arduino needs --qemu" >&2; exit 2; }
  emulator_check "$qemu" "$arduino"
fi

step "all checks passed"
