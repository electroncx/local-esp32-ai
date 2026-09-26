"""Score the exported model by running the real C engine (host build).

    make -C host && python train/evaluate.py

The fact set is the model's tier (written next to it by export.py). Checks:
  1. every phrasing of every fact in the tier -> its exact answer
  2. the tier's eval files: held-out paraphrases, refusals and arithmetic.
     Eval questions that are also trained phrasings are scored separately.
  3. directness: no answer may exceed the hard cap or open with filler
  4. the gate index (built in Python) parses every phrasing exactly like the
     C gate does

Fails on any trained miss, any wrong eval answer (declining is allowed), any
directness violation or any C/Python parser drift.
"""

import argparse
import re
import sys

import engine
import gate_index
from common import MAX_A, canonical, dropped_of, load_eval, load_facts, normalize, tier_of

# Preamble a direct answer never starts with. "Sure, ..." and "OK, ..." are
# filler; a bare "OK." (Oklahoma's abbreviation) is an answer.
FILLER = re.compile(r"^((sure|well|so|okay|ok)[,!]|(the answer|i think|great question|it is|as an)\b)", re.I)


run = engine.ask  # the C engine, on all CPU cores


REFUSAL = "I don't know."


def report(name, rows):
    """Prints the score; returns (right, wrong answers, refusals of an answerable question)."""
    ok = sum(got == want for _, want, got in rows)
    wrong = sum(got != want and got != REFUSAL for _, want, got in rows)
    print(f"{name}: {ok}/{len(rows)} ({wrong} wrong, {len(rows) - ok - wrong} refused)")
    for q, want, got in rows:
        if got != want:
            print(f"    {q!r}: want {want!r}, got {got!r}")
    return ok, wrong, len(rows) - ok - wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=None, choices=["small", "max4mb", "large"],
                    help="fact set (default: the one export.py wrote next to --model)")
    ap.add_argument("--bin", default="host/tinyai")
    ap.add_argument("--model", default="firmware/data/model.bin")
    ap.add_argument("--eval", nargs="*", default=None, help="default: the tier's eval files")
    ap.add_argument("--min-heldout", type=float, default=0.0,
                    help="exit 1 if held-out accuracy falls below this")
    args = ap.parse_args()
    args.tier = args.tier or tier_of(args.model)

    phrasings = sorted({normalize(q) for qs, _ in load_facts(args.tier) for q in qs})
    c_side = run(args.bin, args.model, phrasings, "--words")
    py_side = [" ".join([str(gate_index.QTYPES.index(gate_index.qtype(p)))] +
                        gate_index.content_words(p)) for p in phrasings]
    drift = [(p, c, py) for p, c, py in zip(phrasings, c_side, py_side) if c != py]
    print(f"gate index matches C parser: {len(phrasings) - len(drift)}/{len(phrasings)}")
    for p, c, py in drift[:10]:
        print(f"    {p!r}: C {c!r}, Python {py!r}")

    # A fact export.py had to drop (model wrong, no room for its answer) must be refused.
    dropped = dropped_of(args.model)
    pairs = [(q, "I don't know." if canonical(qs) in dropped else a) for qs, a in load_facts(args.tier) for q in qs]
    if dropped:
        print(f"facts refused by design (the model gets them wrong, no room to store them): {len(dropped)}")
    # Every phrasing of a fact reaches the same key, so the model runs once per
    # key: the C gate routes each phrasing (--gate), then the C engine answers
    # each distinct key. Same result as asking every phrasing, 4x faster.
    routes = run(args.bin, args.model, [q for q, _ in pairs], "--gate")
    keys = sorted({r for r in routes if r != "-" and not r.startswith("=")})
    answer = dict(zip(keys, run(args.bin, args.model, keys)))
    got = [r[1:] if r.startswith("=") else "I don't know." if r == "-" else answer[r] for r in routes]
    t_ok, _, _ = report("trained questions", [(q, a, g) for (q, a), g in zip(pairs, got)])

    # Eval questions that are also trained phrasings (after normalizing) only
    # check consistency; the held-out score counts the rest, never trained on.
    trained = {normalize(q) for q, _ in pairs}
    held_all = load_eval(args.tier, args.eval)
    got_all = run(args.bin, args.model, [q for q, _ in held_all])
    rows = [(q, a, g) for (q, a), g in zip(held_all, got_all)]
    _, seen_wrong, _ = report("eval questions that are trained phrasings",
                              [r for r in rows if normalize(r[0]) in trained])
    held = [r for r in rows if normalize(r[0]) not in trained]
    ok, wrong, refused = report("held-out (never trained on)", held)

    bad = [g for g in got + got_all if len(g) > MAX_A or FILLER.match(g)]
    print(f"directness violations: {len(bad)}")
    for g in bad:
        print(f"    {g!r}")
    # Pass: every trained phrasing exact, no wrong held-out answer (declining is
    # allowed: "I don't know." is the safe failure), direct answers, C == Python.
    if bad or drift or t_ok < len(pairs) or wrong or seen_wrong or ok / len(held) < args.min_heldout:
        sys.exit(1)


if __name__ == "__main__":
    main()
