"""Download the LibriTTS-R transcripts.

LibriTTS-R itself ships as multi-gigabyte audio tarballs, and the LibriSpeech
transcripts that the task description offers as a fallback are uppercased and
already stripped of punctuation, which makes them useless as a source of weak
break labels. What is needed is LibriTTS-R's ``text_original`` field, so this
pulls the text-only mirror, which is four parquet files totalling ~40 MB.

If the mirror is unreachable, point CAESURA_LIBRITTS_PARQUET_DIR at a local
directory of parquet files with ``id`` and ``text_original`` columns instead.
"""

from __future__ import annotations

import os
import sys
import urllib.request

REPO = "ylacombe/libritts-r-text-tags-v2"
FILES = [
    "clean/dev.clean-00000-of-00001.parquet",
    "clean/test.clean-00000-of-00001.parquet",
    "clean/train.clean.100-00000-of-00001.parquet",
]
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for rel in FILES:
        name = os.path.basename(rel)
        dest = os.path.join(OUT, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"have {name}")
            continue
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}"
        print(f"get  {name}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:  # pragma: no cover - network path
            print(f"failed: {exc}", file=sys.stderr)
            if os.path.exists(dest):
                os.remove(dest)
            return 1
    print(f"ok -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
