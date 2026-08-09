# Independent audit of the `clean` labels

> **Question:** *"You collected the clean cases through your own system. Aren't
> you checking your system with your system?"*
>
> **Answer:** no for selection, and now no for verification either. 186 of 188
> clean labels are confirmed by a referee that shares no code with the detector.
> **2 were mislabelled and are listed below.**

Run 2026-08-09 · reads `../contradiction_result/labeled_test_set.jsonl` ·
writes only into this folder · **v1's reported artifacts are untouched**.

---

## 1. What was already safe

Of the 1,346 labels, **1,158 (86%) are certain by construction** — a seeded
script produced every corruption and every paraphrase. The detector had no say.

The remaining **188 `clean` rows** were the weak point: real LLM output,
labelled clean on the presumption that the assistant got it right.

**But the detector did not select them.** Checked directly:

```
captured turns                    : 188
clean cases in the test set       : 188      ← every one
turns the LIVE detector flagged   : 10
  ...still used as clean          : 10       ← none dropped
  ...excluded from the clean set  : 0
clean rows using raw LLM text     : 188      ← pre-correction, always
```

A circular design would have kept only the turns the detector passed, quietly
converting its own blind spots into free true negatives. This kept all of them,
including the ten it flagged, using the text from *before* any correction.

So the only open question was whether the originals were themselves correct.

---

## 2. The referee

`audit_clean_cases.py`. Deliberately dumb: **no NLI, no embeddings, no
thresholds, and no imports from the detector.** It compares literal values
against `product_refs` ∪ `graph_before` and nothing else.

| Rule | Check | Severity |
|---|---|---|
| **R1** | every `£` amount is a known price, a difference between two known prices, or a budget figure the user asked for | **failure** |
| **R2** | every colour word is a known colour (product names blanked first, since names embed colours) | **failure** |
| **R3** | for multi-item turns, each item's name appears | advisory |

### Two exemptions, both added after a first pass produced only false alarms

**(a) Hypernym.** `Brown` where the truth is `Yellowish Brown` is a more general
rendering of the same colour, not a different one — exactly the reasoning the
test set already uses for its hard negatives (`Dress` → `maxi dress` is benign).
A component word of a known compound colour is allowed.

*Caught 11 false alarms:* `Brown` ⊂ `Yellowish Brown`, `Pink` ⊂ `Light Pink`,
`Purple` ⊂ `Light Purple`, `Grey` ⊂ `Dark Grey`.

**(b) Styling aside.** *"pairs well with black dresses"* while recommending a
**boot** is advice about other garments, not a claim about this one. Detected as:
the colour is followed by a garment noun that is not this product's type.

*Caught 3 false alarms:*

```
ccase_0630  "pairs well with ... including black dresses"     product is a boot
ccase_1194  "perfect for dressing up a Black Dress"           product is a blouse
ccase_1224  "similar to the Black Dress items you often buy"  user's history
```

Budget phrases (`under £50`, `cheaper than £30`) are exempted the same way, and
derived arithmetic (`£4.04 cheaper`) is allowed — the same allowance the
hallucination referee makes.

---

## 3. Result

```
188 clean cases
  186 pass
    2 FAIL          <- genuinely mislabelled
   27 minor notes    (hypernyms, asides, re-cased names — reported, not counted)
```

### The two failures

**`ccase_1080`** — `catalog_search`

```
evidence : FESTIVAL Tee £10.08  |  "Printed  tee 9.99" £10.08
response : "Option 2: Printed tee, White, £9.99"
```

The product's **name** contains `9.99`. The LLM lifted it as the price. The
actual price is £10.08. **A genuine contradiction labelled clean.**

**`ccase_0789`** — `item_compare`

```
evidence : Skinny Midprice No Fade Black £29.49
response : "...compared to Skinny Midprice No Fade Black at £24.99
            (not £29.49, as per evidence)"
```

It asserts £24.99 while parenthetically noting the evidence says £29.49. The
primary claim contradicts the evidence. **Also genuine.**

Both are **price** errors, both caught by R1 — the least ambiguous rule.

---

## 4. What to do with them

They are **2 of 188 clean rows**, and 2 of 333 negatives overall. The effect on
any metric is well inside the confidence intervals already reported.

The correct handling is to **exclude them rather than relabel them**. Relabelling
to `contradiction` would put a case into the positive class whose corruption was
not script-generated, breaking the property that makes every positive label
certain. Excluding keeps that property intact and simply removes two rows whose
label cannot be trusted either way.

**v1's reported numbers are computed on the test set as it stands and are not
restated.** This audit records the defect; it does not silently rewrite the file
the report was built from.

---

## 5. What this referee cannot do

**It is stricter than the detector on invented values and weaker on
association.** A cross-item swap in a clean response — two products exchanging
correct values — passes R1 and R2, because every value present is a known value.
The same limitation the hallucination referee documents.

**Colour prose is genuinely ambiguous.** The two exemptions above cover the
patterns observed here; a colour used in some other rhetorical way could still
produce a false alarm. This is why R3 is advisory and why the 27 minor notes are
reported rather than counted.

**It cannot verify completeness** — that the response omitted nothing it should
have said. Only stated values are checked.

---

## 6. What to say

> *"86% of the labels are certain by construction — a seeded script produced
> every corruption. The clean labels were not selected by the detector: all 188
> captured turns became clean cases regardless of its verdict, using the raw
> pre-correction text. We then verified them with a rule-based referee sharing
> no code with the detector. 186 of 188 confirmed; 2 were mislabelled, both
> price errors, and both are documented."*

Finding two errors in your own negatives is a stronger position than asserting
there are none.

---

## 7. Files

| File | Contents |
|---|---|
| `audit_clean_cases.py` | the referee |
| `results_clean_audit.json` | per-case verdicts, problems and minor notes |
| `clean_audit_report.txt` | human-readable report of the failures |

```bash
python test_result/contradiction_result_v2/audit_clean_cases.py
```

No models, no network, runs in under a second.
