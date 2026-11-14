# Amazon ML Challenge 2026 — Guidelines, Submission & Documentation Rules

Source: `guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf` (organiser-issued).
This file is the single source of truth for competition constraints. `CLAUDE.md` points here.

---

## 1. Timeline

| Item | Detail |
|---|---|
| Challenge window | **25 Sep 2026, 12:00 AM IST → 27 Sep 2026, 11:59 PM IST** |
| Day 1 | Problem statement + dataset released |
| Day 3 | Final deadline to build and submit |
| Submission cap | **5 submissions per day**, across 3 days; submit button disables after the cap |
| Leaderboard | Live team rankings during the challenge; **final leaderboard revealed after** the window closes |

Queries during the hackathon go through the organisers' Google Form (linked in the PDF).
Technical/platform issues: `support@unstop.com` with a screenshot + registered email ID.
Organisers explicitly will **not** make decisions on your behalf.

---

## 2. Required submission artefacts

For the **best solution submitted by the team**, both of these must be shared:

### 2.1 Approach document — 1 to 2 pages
Must explain:
- the **ML approach**
- **ML models used**
- **experiments** run
- **conclusion**

Keep it to 1–2 pages. This is a length limit, not a suggestion — write tight, lead with results.

### 2.2 Source code
- Covers **experiments, training and inference**.
- **Proper comments describing the functions.** Judges read this code; undocumented functions are a
  scored weakness, not a style nit.

### 2.3 Top-100 follow-up submission
If the team reaches the top 100, these additional details/documents are required:
- **Methodology used**
- **Candidate generation / blocking strategy**
- **Model architecture and feature engineering**
- Any other relevant information about the approach

> The explicit call-out of *candidate generation / blocking* signals this is a
> matching / retrieval / entity-resolution-flavoured problem. Whatever the final task turns out to
> be, keep a written record of how the candidate set is narrowed before ranking — it is directly
> asked for.

---

## 3. Evaluation & shortlisting

- **Two leaderboards: Public and Private.** Evaluation and shortlisting are based on performance
  **across both**.
- Shortlisting is based on **submitted solutions** — so version history must be maintained.
- Participants **may be required to submit final source code at a later stage**.
- Top 100 teams announced after: artefacts submitted + leaderboard score + each member meeting
  eligibility criteria.

**Practical consequence:** optimise local cross-validation, not the public leaderboard. A public-LB
overfit that collapses on private LB loses the shortlist. Track the CV↔public-LB correlation; when
they disagree, trust CV.

---

## 4. Submission budget protocol

15 total slots. Treat each as a scarce resource.

Before spending a slot, record in `experiments/log.md`:
1. Run id and config diff vs. the previous best
2. Local CV score (fold-wise, not just mean)
3. Why this is worth a slot (what hypothesis it tests)
4. After submitting: the public LB score and the CV↔LB delta

Suggested allocation, adjust once the metric is known:
- **Day 1:** 1 slot on a trivial baseline (validates the submission format end-to-end, early), rest
  on first real models.
- **Day 2:** feature/model exploration; keep at least 1 slot in reserve.
- **Day 3:** ensembles/blends and the final pick; **never leave the final submission to the last
  hour** — platform issues are your problem, not the organisers'.

Naming: `submissions/sub_<NNN>_<short-tag>.csv`. **Never overwrite a submission file.** Each one
pairs with a config snapshot in `experiments/`.

---

## 5. Version history requirement

Shortlisting is based on submitted solutions and code may be re-requested later. Therefore each
submitted solution must be replayable from:
- the exact code state (git commit hash, or a zipped snapshot if not using git),
- the config file used,
- the random seed,
- the data preprocessing step versions.

Record the mapping `submission file → commit/snapshot → config → CV score → LB score` in
`experiments/log.md`. If that chain is broken for the final best solution, the submission is
effectively unverifiable.

---

## 6. Platform & conduct rules

- **Desktop or laptop only.** Mobile devices are not supported.
- **No simultaneous logins.** One machine per participant; detected simultaneous logins may
  terminate the challenge attempt entirely and produce only error messages.
- **Cheating, plagiarism, or multiple registration IDs → instant disqualification.**
  - Any external code, dataset, or pretrained weights must be permissibly licensed and cited in the
    approach document.
- Technical troubleshooting: clear browser cache, try another browser or incognito, or switch
  network (hotspot/wifi).

---

## 7. Pre-flight checklist (run before every submission)

- [ ] Submission file matches the required format exactly (columns, order, id dtype, row count)
- [ ] Row count equals the test set row count; no NaNs; no duplicate ids
- [ ] Inference ran from saved artefacts, not from in-memory training state
- [ ] Seed set, config snapshotted, run logged in `experiments/log.md`
- [ ] Slot count for today checked (≤ 5)
- [ ] Code committed / snapshotted before submitting

## 8. Pre-flight checklist (run before the final deadline)

- [ ] `docs/approach.md` written, **1–2 pages**, covers approach / models / experiments / conclusion
- [ ] All functions in `src/` have docstrings describing what they do
- [ ] `train.py` and `inference.py` run end-to-end from a clean state
- [ ] `README.md` gives exact reproduction steps (env, data placement, commands)
- [ ] Top-100 notes drafted: methodology, candidate generation/blocking, model architecture,
      feature engineering
- [ ] Best submission identified and its full version chain verified
