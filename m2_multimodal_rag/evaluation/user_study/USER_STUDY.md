# M2 Novelty User Study — Protocol & Form Questions

**Goal:** human evidence for three novelties that automated metrics can't fully
capture: explanation trustworthiness (N4 guard), emotional-style match
(N5 Kansei KB), and diversity after rejection (N3 bandit).

**Participants:** 10 (friends/classmates fine; note age range + familiarity
with online shopping). **Duration:** ~15 min. **Format:** Google Form,
anonymous. All pairs are shown as "Option A / Option B" with the underlying
configuration hidden and left/right order randomised per question
(record the mapping in `stimuli_key.csv` — never in the form).

---

## Preparing stimuli (before sending the form)

1. Start M2 (warmed). For each part below, run the listed prompts twice —
   once normally, once with the ablation env var — and screenshot the result
   cards / copy the explanation text.
2. Save screenshots as `stimuli/q<part><num>_<a|b>.png`; record which side is
   which config in `stimuli_key.csv` (columns: question_id, side_a, side_b).

**Part 1 — Explanations (N4 guard):** 8 items. For each item, capture the
explanation with the guard ON (default) and OFF (`M2_ABLATE_GUARD=none`).
Choose items where the raw explanation contains an error if possible (check
against the product image/metadata).

**Part 2 — Emotional queries (N5 Kansei):** 6 queries, one per style
(elegant / casual / sporty / romantic / professional / bold), e.g.
"something elegant for a gala". Capture top-3 result cards with KB ON
(default) and OFF (`M2_ABLATE_KB=1`).

**Part 3 — Diversity (N3 bandit):** 4 scenarios. Script: initial search →
reject two items ("not those") → capture the NEXT recommendation set with
the adaptive bandit (default) and with `M2_ABLATE_BANDIT=0.9`
(relevance-only baseline).

---

## Form questions

**Intro:** "You will compare pairs of fashion recommendations produced by two
versions of the same system. There are no right answers — choose what YOU
find better."

Demographics (2): age range; how often do you shop for clothes online?
(never / occasionally / monthly / weekly)

### Part 1 — Which explanation do you trust? (8 questions)
Show: product image + Explanation A + Explanation B.
- Q: "Which explanation describes this product more accurately?"
  (A / B / both equal)
- Q: "Rate your trust in Explanation A" (1 = not at all – 5 = fully)
- Q: "Rate your trust in Explanation B" (1–5)

### Part 2 — Which results match the style? (6 questions)
Show: the query text + result-set screenshot A + screenshot B.
- Q: "The shopper asked for **<style>**. Which set of items better matches
  that style?" (A / B / both equal)

### Part 3 — Better follow-up suggestions? (4 questions)
Show: "The shopper rejected these two items ▸ [rejected items] — the system
then suggested:" set A + set B.
- Q: "Which new set gives the shopper better alternatives?" (A / B / both equal)
- Q: "Which set feels more repetitive?" (A / B / neither)

Closing (optional): "Any comments on the recommendations or explanations?"

---

## Analysis (put responses in `responses.csv`, one row per participant-question)

- **Preferences:** win rate per novelty (system vs ablated), report with
  binomial 95% CI; a two-sided sign test against 50% gives significance.
- **Likert (Part 1):** paired per participant-item → Wilcoxon signed-rank
  (scipy.stats.wilcoxon), report medians and p.
- Report: n participants, demographics summary, win-rate bar chart per part
  (make_figures.py `fig_n5_kansei` style), and 2–3 quoted comments.

**Report wording tip:** describe it as a "small-scale preliminary user study
(n=10)" and avoid over-claiming significance — examiners reward calibrated
claims backed by the automated metrics + human corroboration together.
