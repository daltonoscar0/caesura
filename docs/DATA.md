# Data

Part of the [Caesura](../README.md) documentation.

**Before reading any numbers here, read [the caveat](../README.md#read-this-before-the-results):** break labels for training and for two of the three test sets are derived from punctuation, and punctuation is not prosody.

## Data

| split | source | utterances / passages | boundaries | none | b2 | b3 |
|---|---|---:|---:|---:|---:|---:|
| train | LibriTTS-R train-clean-100 | 10,196 | 575,700 | 488,618 | 50,034 | 37,048 |
| dev | LibriTTS-R dev-clean | 1,779 | 99,597 | 84,509 | 8,766 | 6,322 |
| test (a) | LibriTTS-R test-clean | 1,565 | 90,525 | 77,359 | 7,818 | 5,348 |
| test (b) | Switchboard conversational | 500 | 5,537 | 4,767 | 264 | 506 |
| test (c) | adversarial, hand annotated | 60 | 566 | 441 | 63 | 62 |

**LibriTTS-R.** The task description offered `librispeech_asr` text as a simpler
fallback. It is not usable: LibriSpeech transcripts are uppercased and already
stripped of punctuation, so there is nothing to derive labels from. What is
needed is LibriTTS-R's `text_original` field, which preserves the punctuation of
the underlying Project Gutenberg text. The full LibriTTS-R release is
multi-gigabyte audio, so `scripts/fetch_data.py` pulls a text-only mirror
(`ylacombe/libritts-r-text-tags-v2`, four parquet files, about 40 MB).

Consecutive utterances from the same chapter are stitched into passages of about
45 tokens. This matters more than it sounds: LibriTTS-R segments are mostly
single sentences, so a model trained on them one at a time learns "the last
token is `<b3>`" and nothing else, and every evaluation number becomes a
measurement of that shortcut. Stitching puts `<b3>` in the middle of the
sequence where it has to be predicted from syntax. The `final_only` baseline in
the results tables is there to show how much of the score that shortcut is worth.

**Switchboard.** 500 utterances of at least six tokens, sampled with a fixed
seed from the disfluency-cleaned, punctuated Switchboard text in the sibling
`mend` project, read only. Punctuation is the gold signal, with the same caveat
as above. This set is not redistributed here (Switchboard is licensed);
`scripts/build_data.py` rebuilds it from the sibling checkout.

**Adversarial set.** 60 sentences written by hand for this project, in
`data/adversarial.jsonl`, each with a gold break annotation and a one-clause
rationale. Seven construction types: reduced relative (10), NP/Z garden path
(8), long subject (8), coordination scope (8), appositive and parenthetical and
vocative (8), fronted PP or adverbial (8), and break at no punctuation site
(10). The last bucket is the one that no punctuation-derived corpus can contain.
Gold is my own judgement about where a competent reader pauses, and the
rationale is there so that a disagreement can be argued with rather than merely
noted.
