Benchmarks used in J-Lens correlation analysis

## Pre-computed EleutherAI benchmarks (26 checkpoints)

Scores taken from the official Pythia evaluation harness. All are
zero-shot unless noted.

### ARC-Easy (AI2 Reasoning Challenge — Easy)
- **Task:** Multiple-choice science questions (grade-school level)
- **Metric:** Accuracy (↑, 0–1)
- **Example:** "Which property of a mineral can be determined just by looking at it? (A) luster (B) mass (C) weight (D) hardness"
- **Interpretability signal:** Tests basic factual recall and simple reasoning

### ARC-Challenge (AI2 Reasoning Challenge — Hard)
- **Task:** Harder multiple-choice science questions
- **Metric:** Accuracy (↑, 0–1)
- **Note:** Very low scores for 160M model (~0.2, near chance)

### PIQA (Physical Interaction QA)
- **Task:** Multiple-choice physical commonsense reasoning
- **Metric:** Accuracy (↑, 0–1)
- **Example:** "How to separate egg whites? (A) use a water bottle to suck the yolk (B) crack egg and let white drain through fingers"
- **Interpretability signal:** Physical world knowledge + procedural reasoning

### WinoGrande
- **Task:** Pronoun resolution (fill-in-the-blank with two options)
- **Metric:** Accuracy (↑, 0–1)
- **Example:** "The trophy didn't fit in the suitcase because _ was too big. (A) the trophy (B) the suitcase"
- **Interpretability signal:** Coreference resolution, world knowledge

### SciQ
- **Task:** Multiple-choice science questions with supporting paragraph
- **Metric:** Accuracy (↑, 0–1)
- **Interpretability signal:** Science fact retrieval + reading comprehension

### LAMBADA (OpenAI)
- **Task:** Last-word prediction given a long passage context
- **Metric:** Accuracy (↑, 0–1)
- **Example:** "...she opened the door and saw a _"
- **Interpretability signal:** Long-range context integration

### LogiQA
- **Task:** Multiple-choice logical reasoning (translated from Chinese civil service exams)
- **Metric:** Accuracy (↑, 0–1)
- **Interpretability signal:** Deductive and categorical reasoning

### WSC (Winograd Schema Challenge)
- **Task:** Pronoun disambiguation (binary choice)
- **Metric:** Accuracy (↑, 0–1)
- **Note:** Very small test set, high variance

---

## WMT14 fr-en (153 checkpoints, our eval)

We run inference directly with Pythia-160M (greedy decoding, max 128
new tokens) on the full 3003-example WMT14 French→English test set.
Scores computed with sacreBLEU.

### BLEU (Bilingual Evaluation Understudy, ↑ better)
- **What:** n-gram overlap (n=1..4) between generated output and reference
- **Range:** 0–100. Score of 0 means zero 4-gram matches with reference
- **Pythia-160M range:** 0.0–0.4 (essentially zero everywhere)
- **Why so low:** Pythia is monolingual English; it never produces correct
  French→English translations. The model often generates unrelated English
  or garbled text.

### chrF (character n-gram F-score, ↑ better)
- **What:** Character-level n-gram (n=1..6) precision/recall against reference
- **Range:** 0–100
- **Pythia-160M range:** 0.0–11.2
- **What this captures:** Even without translating, the model learns to
  produce English text that shares substrings with the reference (common
  words like "the", "and", "of"). The chrF score weakly tracks the model's
  general English fluency.

### TER (Translation Edit Rate, ↓ better)
- **What:** Minimum number of insertions/deletions/substitutions/shifts
  needed to turn the hypothesis into the reference, divided by reference
  length. Expressed as percentage (×100).
- **Range:** ≥ 0. TER ≈ 100 means the edit cost equals the reference length
  (common when output is empty or very short). TER < 20 is decent translation.
- **Pythia-160M range:** 100–389
- **What this captures:** As the model learns to generate longer, more
  fluent English, TER drops because the output shares more tokens with the
  reference. But it never reaches translation quality (TER stays > 100).

