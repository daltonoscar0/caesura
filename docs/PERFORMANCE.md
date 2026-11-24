# Speed

Part of the [Caesura](../README.md) documentation.

**Before reading any numbers here, read [the caveat](../README.md#read-this-before-the-results):** break labels for training and for two of the three test sets are derived from punctuation, and punctuation is not prosody.

## Speed

Measured on CPU, one utterance at a time, over 300 LibriTTS-R test utterances (17014 tokens).

| system | tokens/second | ms per utterance |
|---|---:|---:|
| rules | 1299 | 43.65 |
| model | 461 | 123.08 |
| model+emphasis | 349 | 162.64 |

Streaming rather than batched, because a front end sees one utterance at a time.
The `rules` figure includes the spaCy parse, which is nearly all of it. The
`model` figure is the transformer forward pass alone; `model+emphasis` adds the
parse back, because emphasis marking is rule-based and needs a parse whichever
system placed the breaks. So the honest cost of the model path in a system that
also wants emphasis is 349 tokens per second, not 461.

Both are comfortably faster than real time (English speech runs at roughly 3
words per second) and neither is close to being the bottleneck in a TTS pipeline.
The interesting number is that they are within a factor of four of each other:
the parse is not cheap, and choosing System A does not buy much speed.
