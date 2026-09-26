"""Fast routing check without running the model: for every trained phrasing
and every eval question, does the gate (or the calculator) pick the right
fact? Catches ambiguous or colliding phrasings before hours of training.

    make -C host && python train/check_routing.py --model firmware/data/model.bin
"""

import argparse
import sys

import engine
from common import canonical, dropped_of, load_eval, load_facts, tier_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=None, choices=["small", "max4mb", "large"],
                    help="fact set (default: the one export.py wrote next to --model)")
    ap.add_argument("--bin", default="host/tinyai")
    ap.add_argument("--model", default="firmware/data/model.bin")
    ap.add_argument("--eval", nargs="*", default=None, help="default: the tier's eval files")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()
    args.tier = args.tier or tier_of(args.model)
    facts = load_facts(args.tier)
    key2ans = {canonical(qs): a for qs, a in facts}
    dropped = dropped_of(args.model)  # refused by design: their phrasings must route to "-"

    def route(qs):
        out = engine.ask(args.bin, args.model, qs, "--gate")
        return [o[1:] if o.startswith("=") else ("I don't know." if o == "-" else key2ans.get(o, "?"))
                for o in out]

    failed = False
    for name, rows in [("eval", load_eval(args.tier, args.eval)),
                       ("trained", [(q, "I don't know." if canonical(qs) in dropped else a)
                                    for qs, a in facts for q in qs])]:
        got = route([q for q, _ in rows])
        bad = [(q, w, g) for (q, w), g in zip(rows, got) if g != w]
        refused = sum(g == "I don't know." for _, _, g in bad)
        print(f"{name}: {len(rows) - len(bad)}/{len(rows)} routed right "
              f"({len(bad) - refused} to a wrong answer, {refused} refused)")
        for q, w, g in bad[:args.show]:
            print(f"    {q!r}: want {w!r}, got {g!r}")
        # Fail on anything routed to a wrong answer, or a trained phrasing that
        # isn't routed to its own fact; refusing a test question is allowed.
        failed |= (len(bad) - refused) > 0 or (name == "trained" and bool(bad))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
