"""Turn the raw sources into the four splits the rest of the project reads."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caesura import data  # noqa: E402

OUT = data.DATA


def main() -> int:
    made = []

    for split, name in [
        ("train.clean.100", "train"),
        ("dev.clean", "dev"),
        ("test.clean", "test_libritts"),
    ]:
        rows = data.load_libritts_split(split)
        passages = data.build_passages(rows, "libritts_r")
        path = os.path.join(OUT, f"{name}.jsonl")
        n = data.write_jsonl(passages, path)
        counts = data.label_counts(passages)
        total = sum(counts.values())
        made.append((name, n, total, counts))

    swb = data.load_switchboard(limit=500)
    path = os.path.join(OUT, "test_switchboard.jsonl")
    n = data.write_jsonl(swb, path)
    made.append(("test_switchboard", n, sum(data.label_counts(swb).values()), data.label_counts(swb)))

    adv = data.load_adversarial()
    made.append(("adversarial", len(adv), sum(data.label_counts(adv).values()), data.label_counts(adv)))

    print(f"{'split':20s} {'passages':>9s} {'tokens':>9s}  {'none':>7s} {'b2':>7s} {'b3':>7s}")
    for name, n, total, counts in made:
        print(
            f"{name:20s} {n:9d} {total:9d}  "
            f"{counts.get('none',0):7d} {counts.get('b2',0):7d} {counts.get('b3',0):7d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
