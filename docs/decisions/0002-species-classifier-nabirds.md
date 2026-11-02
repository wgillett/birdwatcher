# 0002: Species classifier — NABirds EfficientNetV2-S via ONNX Runtime

## Status

Accepted. Implemented alongside this ADR.

## Context

`src/bird/classify.py` only had `StubClassifier`, which always answers
`"unknown"`. The detector stage was already resolved (see
`docs/decisions/0001-bird-detector-yolo-onnx.md`); this is the next cascade
stage per AGENTS.md: "Classifier: local species model runs on each crop."
AGENTS.md's "Open Questions" left the species-classifier choice open too:
NABirds-trained vs. iNaturalist-trained.

The user supplied real backyard photos, informally labeled by filename, to
drive a real-image test: `bird_blue_jay.jpg` / `bird_two_blue_jays.jpg` (Blue
Jay), `bird_common_grackle.jpg` (Common Grackle), `bird_male_cardinal.jpeg`
(Northern Cardinal — the official common name for what's informally called
"cardinal" in North America, *Cardinalis cardinalis*).

## Decision

Use the NABirds-trained classifier from **Ornimetrics' `ornimetrics-edge`**
project on Hugging Face (https://huggingface.co/Ornimetrics/ornimetrics-edge):
`species_classifier_nabirds.onnx` — EfficientNetV2-S, 256x256 input, 555
North American species (NABirds taxonomy), 86.9% top-1 on the NABirds
validation set. Ships with `species_classifier_nabirds.json` (class list +
RGB mean/std normalization stats), both committed to `models/`. Run via
`onnxruntime.InferenceSession` with `CPUExecutionProvider`, same pattern as
the detector. No export step was needed — the ONNX file is published
directly by the model author.

Implemented as `NABirdsOnnxClassifier` in `src/bird/classify.py`, alongside
— not replacing — `StubClassifier`.

### Why NABirds over iNaturalist

The same Hugging Face repo also ships an iNaturalist-trained alternative
(`species_classifier_inat.onnx`, 302 species, ~73% top-1, CC-BY/CC-BY-NC
licensed — explicitly "the commercial path" per that model's card). NABirds
was chosen for meaningfully better accuracy and species coverage; see
**License** below for why this is fine for this project specifically.

### Label collapsing

NABirds' 555 classes are finer-grained than plain species — many species are
split into sex/age/plumage sub-classes (e.g. `Northern_Cardinal_Adult_Male`,
`Northern_Cardinal_Female_Juvenile`), others just carry an opaque catalog id
suffix (e.g. `Blue_Jay_ID_950`). `_species_name()` collapses a raw label down
to a plain species common name by splitting on `_` and truncating at the
first token that's a known qualifier (`ID`, `Adult`, `Male`, `Female`,
`Juvenile`, `Immature`, `Breeding`, `Nonbreeding`, `Winter`, `Summer`,
`Plumage`, `Morph`, `Eclipse`) or a bare digit.

`classify()` doesn't just rename the top class — it **merges predicted
probability mass by normalized species name** before ranking (e.g. the two
cardinal sub-classes' probabilities are summed under `"Northern Cardinal"`).
This makes top-1 selection reflect total species-level confidence rather
than whichever single sub-class happened to score highest, and is what
"official species name" means at the modeling level, not just display.

Known limitation, not fixed: a few NABirds names use `_s_` for possessives
(e.g. `Allen_s_Hummingbird` collapses to `"Allen s Hummingbird"` instead of
`"Allen's Hummingbird"`). Doesn't affect the three species this project has
test coverage for.

## License

**The NABirds classifier is licensed for non-commercial research use only**
(NABirds dataset terms) — not a permissive open-source license. This project
(`wgillett/birdwatcher`) is a personal hobby project; **the user has
confirmed directly that it is not, and will not be, monetized.** On that
basis the non-commercial-research-use license is satisfied. The model weight
file is committed to this public GitHub repo; the NABirds dataset terms don't
explicitly address redistribution of derived model weights (only usage), so
this is accepted as a reasonable-but-not-airtight interpretation for a
personal project — see the decision log in the originating conversation for
the full tradeoff discussion.

**If this project's purpose ever changes** (e.g. any commercial use, or
redistribution terms turn out to matter more than assumed here), swap this
classifier for the iNaturalist (CC-BY/CC-BY-NC) alternative in the same
Hugging Face repo, or another permissively-licensed model — not a drop-in
guarantee of equal accuracy/coverage, but same architecture and same
`Classifier` protocol, so the swap is mechanical.

## Consequences / follow-ups

- `NABirdsOnnxClassifier` is not yet wired into `cli.py`/`pipeline.py`'s
  default build — `_build_pipeline` still hardcodes both `StubDetector` and
  `StubClassifier`. Wiring both real stages into the CLI together is a
  natural follow-up, not done here.
- The `_s_` possessive-name collapsing bug affects species outside this
  project's current test coverage; not fixed.
- `top_k` (default 5) and the stop-word list are untuned/unvalidated beyond
  the three species this project has photos for.
