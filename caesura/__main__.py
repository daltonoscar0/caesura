"""Command line entry point.

    python -m caesura "the man in the grey coat is my uncle"
    python -m caesura --system both --json "..."
    python -m caesura --eval
"""

from __future__ import annotations

import argparse
import sys

from .api import SYSTEMS, run


def _print_human(result) -> None:
    print(result.text)
    if result.decisions:
        print()
        width = max(len(d.boundary) for d in result.decisions)
        for d in result.decisions:
            line = f"  {d.boundary:<{width}}  {d.value}  {d.rule}  {d.score:.3f}"
            for alt in d.alternatives:
                line += f"  [{alt.system}={alt.value} {alt.score:.3f} {alt.rule}]"
            print(line)
    emphasis = result.meta.get("emphasis") or {}
    if emphasis:
        print()
        print("  emphasis: " + ", ".join(f"{tok} ({rule})" for tok, rule in emphasis.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="caesura",
        description="Predict phrase breaks and emphasis on punctuation-free text",
    )
    ap.add_argument("text", nargs="*", help="text to mark up")
    ap.add_argument("--system", default="rules", choices=SYSTEMS)
    ap.add_argument("--model", default=None, help="local dir or Hub id for System B")
    ap.add_argument("--json", action="store_true", help="emit the full StageResult as JSON")
    ap.add_argument("--no-emphasis", action="store_true")
    ap.add_argument("--eval", action="store_true", help="run the full evaluation instead")
    args, rest = ap.parse_known_args(argv)

    if args.eval:
        from .evaluate import main as eval_main

        forwarded = list(rest)
        if args.model:
            forwarded += ["--model", args.model]
        return eval_main(forwarded)

    text = " ".join(args.text) if args.text else sys.stdin.read()
    if not text.strip():
        ap.error("no text given")

    result = run(
        text,
        system=args.system,
        model_path=args.model,
        emphasis=not args.no_emphasis,
    )
    if args.json:
        print(result.to_json())
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
