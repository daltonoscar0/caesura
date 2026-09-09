"""Render the README's tables from outputs/eval.json.

Kept separate from the README so the numbers in the README are always something
that came out of a run rather than something that was typed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "outputs", "eval.json")

SET_TITLES = {
    "libritts_test": "LibriTTS-R test-clean (held out, in domain)",
    "switchboard": "Switchboard conversational (out of domain)",
    "adversarial": "Adversarial set (hand annotated)",
}

SYSTEM_TITLES = {
    "final_only": "baseline: b3 on final token",
    "rules": "A: parser rules",
    "model": "B: distilroberta",
    "both": "A+B union",
}

CONSTRUCTION_TITLES = {
    "reduced_relative": "reduced relative",
    "np_z": "NP/Z",
    "long_subject": "long subject",
    "coordination": "coordination scope",
    "appositive": "appositive",
    "fronted_pp": "fronted PP",
    "non_punctuation_break": "break at no punctuation site",
}


def headline_table(report) -> str:
    """One compact table for the top of the README: any-break F1 everywhere."""
    sets = list(report["results"])
    lines = [
        "| system | " + " | ".join(SET_TITLES.get(s, s).split(" (")[0] for s in sets) + " |",
        "|---|" + "---:|" * len(sets),
    ]
    for system in ("final_only", "rules", "model", "both"):
        cells = []
        for name in sets:
            block = report["results"][name]["systems"].get(system)
            cells.append(f"{block['any']['f1']}" if block else "-")
        lines.append(f"| {SYSTEM_TITLES.get(system, system)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def results_tables(report) -> str:
    out = []
    for name, block in report["results"].items():
        out.append(f"**{SET_TITLES.get(name, name)}** "
                   f"({block['n_utterances']} utterances, {block['n_boundaries']} boundaries)\n")
        out.append("| system | b2 P | b2 R | b2 F1 | b3 P | b3 R | b3 F1 | any P | any R | any F1 |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for system, s in block["systems"].items():
            out.append(
                f"| {SYSTEM_TITLES.get(system, system)} "
                f"| {s['b2']['p']} | {s['b2']['r']} | **{s['b2']['f1']}** "
                f"| {s['b3']['p']} | {s['b3']['r']} | **{s['b3']['f1']}** "
                f"| {s['any']['p']} | {s['any']['r']} | **{s['any']['f1']}** |"
            )
        out.append("")
    return "\n".join(out)


def baseline_table(report) -> str:
    out = [
        "| published system | test set | comma F1 | period F1 | question F1 | overall micro F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for entry in report["punctuation_restoration_reference"]:
        out.append(
            f"| {entry['system']} | {entry['test_set']} | {entry['comma_f1']} "
            f"| {entry['period_f1']} | {entry['question_f1']} | {entry['overall_micro_f1']} |"
        )
    return "\n".join(out)


def taxonomy_table(report) -> str:
    tax = report["error_taxonomy"]
    systems = [s for s in ("rules", "model", "both") if s in tax]
    constructions = sorted(
        {c for s in systems for c in tax[s]},
        key=lambda c: list(CONSTRUCTION_TITLES).index(c) if c in CONSTRUCTION_TITLES else 99,
    )
    header = "| construction | gold breaks |" + "".join(
        f" {SYSTEM_TITLES[s]} missed | {SYSTEM_TITLES[s]} wrong level | {SYSTEM_TITLES[s]} spurious |"
        for s in systems
    )
    sep = "|---|---:|" + "---:|" * (3 * len(systems))
    lines = [header, sep]
    totals = {s: [0, 0, 0] for s in systems}
    gold_total = 0
    for c in constructions:
        gold = tax[systems[0]].get(c, {}).get("gold_breaks", 0)
        gold_total += gold
        row = f"| {CONSTRUCTION_TITLES.get(c, c)} | {gold} |"
        for s in systems:
            b = tax[s].get(c, {})
            row += f" {b.get('missed', 0)} | {b.get('wrong_level', 0)} | {b.get('spurious', 0)} |"
            totals[s][0] += b.get("missed", 0)
            totals[s][1] += b.get("wrong_level", 0)
            totals[s][2] += b.get("spurious", 0)
        lines.append(row)
    row = f"| **all** | **{gold_total}** |"
    for s in systems:
        row += f" **{totals[s][0]}** | **{totals[s][1]}** | **{totals[s][2]}** |"
    lines.append(row)
    return "\n".join(lines)


def exact_match_table(report) -> str:
    tax = report["error_taxonomy"]
    systems = [s for s in ("rules", "model", "both") if s in tax]
    constructions = sorted(
        {c for s in systems for c in tax[s]},
        key=lambda c: list(CONSTRUCTION_TITLES).index(c) if c in CONSTRUCTION_TITLES else 99,
    )
    lines = [
        "| construction | " + " | ".join(SYSTEM_TITLES[s] for s in systems) + " |",
        "|---|" + "---:|" * len(systems),
    ]
    for c in constructions:
        cells = " | ".join(str(tax[s].get(c, {}).get("exact_sentences", 0)) for s in systems)
        lines.append(f"| {CONSTRUCTION_TITLES.get(c, c)} | {cells} |")
    return "\n".join(lines)


def garden_path_table(report) -> str:
    lines = [
        "| id | type | gold | System A rules | System B model | A ok | B ok |",
        "|---|---|---|---|---|:-:|:-:|",
    ]
    for row in report["garden_paths"]:
        lines.append(
            f"| {row['id']} | {CONSTRUCTION_TITLES.get(row['construction'], row['construction'])} "
            f"| `{row['gold']}` | `{row['rules']}` | `{row['model']}` "
            f"| {'yes' if row['rules_match'] else 'no'} "
            f"| {'yes' if row['model_match'] else 'no'} |"
        )
    return "\n".join(lines)


def speed_table(report) -> str:
    meta = report["speed"].get("_meta", {})
    lines = [
        f"Measured on CPU, one utterance at a time, over "
        f"{meta.get('utterances', '?')} LibriTTS-R test utterances "
        f"({meta.get('tokens', '?')} tokens).\n",
        "| system | tokens/second | ms per utterance |",
        "|---|---:|---:|",
    ]
    for system, s in report["speed"].items():
        if system.startswith("_"):
            continue
        lines.append(f"| {system} | {s['tokens_per_second']} | {s['ms_per_utterance']} |")
    return "\n".join(lines)


def rule_firing_table(report) -> str:
    libri = report.get("rule_firing_libritts", {})
    adv = report.get("rule_firing_adversarial", {})
    names = sorted(set(libri) | set(adv))
    lines = [
        "| rule | fires on LibriTTS-R test | fires on adversarial |",
        "|---|---:|---:|",
    ]
    for name in names:
        lines.append(f"| `{name}` | {libri.get(name, 0)} | {adv.get(name, 0)} |")
    return "\n".join(lines)


def emphasis_table(report, limit: int = 14) -> str:
    lines = ["| sentence | marked |", "|---|---|"]
    for row in report["emphasis_adversarial"][:limit]:
        marks = ", ".join(f"`{t}` ({r.replace('emph_', '')})" for t, r in row["emphasis"].items())
        lines.append(f"| `{row['text']}` | {marks} |")
    return "\n".join(lines)


SECTIONS = {
    "headline": headline_table,
    "results": results_tables,
    "baselines": baseline_table,
    "taxonomy": taxonomy_table,
    "exact": exact_match_table,
    "garden": garden_path_table,
    "speed": speed_table,
    "firing": rule_firing_table,
    "emphasis": emphasis_table,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("section", nargs="?", choices=sorted(SECTIONS) + ["all"], default="all")
    ap.add_argument("--eval", default=DEFAULT)
    args = ap.parse_args()

    with open(args.eval) as fh:
        report = json.load(fh)

    names = sorted(SECTIONS) if args.section == "all" else [args.section]
    for name in names:
        if args.section == "all":
            print(f"\n<!-- {name} -->\n")
        print(SECTIONS[name](report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
