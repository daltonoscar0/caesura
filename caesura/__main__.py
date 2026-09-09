"""Command line entry point.

    python -m caesura "the man in the grey coat is my uncle"
    python -m caesura --system both --pretty "..."
    python -m caesura --eval

Per the pipeline contract, the default output is the StageResult as JSON on
stdout. ``--pretty`` prints the human-readable form instead.
"""

from __future__ import annotations

import argparse
import json
import sys

from .api import SYSTEMS, run


def _print_human(result) -> None:
    print(result["output"])
    decisions = result["decisions"]
    if decisions:
        print()
        width = max(len(d["surface"]) for d in decisions)
        for d in decisions:
            line = f"  {d['surface']:<{width}}  {d['result']}  {d['rule']}  {d['score']:.3f}"
            for alt in d["alternatives"]:
                line += f"  [{alt['system']}={alt['result']!r} {alt['score']:.3f}]"
            print(line)
    emphasis = result["meta"].get("emphasis") or {}
    if emphasis:
        print()
        print("  emphasis: " + ", ".join(f"{tok} ({rule})" for tok, rule in emphasis.items()))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # --eval hands the whole remaining command line to the evaluation parser.
    # It cannot go through the parser below, whose positional `text` takes
    # nargs="*" and would swallow the value of any flag it does not know.
    if "--eval" in argv:
        from .evaluate import main as eval_main

        argv.remove("--eval")
        return eval_main(argv)

    ap = argparse.ArgumentParser(
        prog="caesura",
        description="Predict phrase breaks and emphasis on punctuation-free text",
    )
    ap.add_argument("text", nargs="*", help="text to mark up; reads stdin if omitted")
    ap.add_argument("--system", default="rules", choices=SYSTEMS)
    ap.add_argument("--model", default=None, help="local dir or Hub id for System B")
    ap.add_argument("--pretty", action="store_true", help="human-readable instead of JSON")
    ap.add_argument("--no-emphasis", action="store_true")
    ap.add_argument(
        "--eval",
        action="store_true",
        help="run the full evaluation instead; remaining flags go to caesura.evaluate",
    )
    args = ap.parse_args(argv)

    text = " ".join(args.text) if args.text else sys.stdin.read()
    if not text.strip():
        ap.error("no text given")

    result = run(
        text,
        system=args.system,
        model_path=args.model,
        emphasis=not args.no_emphasis,
    )
    if args.pretty:
        _print_human(result)
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
