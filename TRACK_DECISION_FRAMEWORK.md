# TRACK DECISION FRAMEWORK: AV vs NLI

## Strategy
Build BOTH tracks (AV and NLI), Categories A+B+C for each. After dev set evaluation, submit the track with the highest combined rubric score.

## Plans
- **AV Plan:** `IMPLEMENTATION_PLAN.md` (974 lines, 3 solutions)
- **NLI Plan:** `NLI_IMPLEMENTATION_PLAN.md` (1,363 lines, 3 solutions)

## Baselines

| Track | SVM (Cat A target) | LSTM (Cat B target) | BERT (Cat C target) |
|-------|-------------------|--------------------|--------------------|
| AV    | 0.5610            | 0.6226             | 0.7854             |
| NLI   | 0.5846            | 0.6603             | 0.8198             |

## Expected Performance (macro_f1)

| Solution | AV Expected | NLI Expected | Notes |
|----------|------------|-------------|-------|
| Cat A    | 0.65-0.75  | 0.65-0.75   | Similar range, different feature types |
| Cat B    | 0.70-0.78  | 0.78-0.85   | ESIM purpose-built for NLI >> char-CNN for AV |
| Cat C    | 0.78-0.84  | 0.83-0.88   | Both use DeBERTa-v3; NLI is easier for transformers |

## Rubric Scoring Projection

| Criterion (max) | AV (A+B) | NLI (A+B) | Notes |
|-----------------|----------|-----------|-------|
| **Performance Sol 1** (3) | 3.0 | 3.0 | Both easily beat SVM |
| **Performance Sol 2** (3) | 2.5-3.0 | 3.0 | AV Cat B has more execution risk |
| **Organisation** (3) | 3.0 | 3.0 | Track-independent |
| **Completeness** (3) | 3.0 | 3.0 | Track-independent |
| **Soundness Sol 1** (3) | 2.5-3.0 | 3.0 | AV Cat A more features = more to go wrong |
| **Soundness Sol 2** (3) | 2.5-3.0 | 2.5-3.0 | AV GRL complex; NLI ESIM+KIM complex but well-documented |
| **Creativity Sol 1** (3) | 2.5-3.0 | 2.5-3.0 | AV: novel features (rhythm, syntactic). NLI: alignment+natlog |
| **Creativity Sol 2** (3) | 2.5-3.0 | 2.5-3.0 | AV: GRL disentanglement. NLI: ESIM+KIM+charCNN |
| **Evaluation** (3) | 3.0 | 3.0 | Both have rich evaluation landscapes |
| **Formatting** (3) | 3.0 | 3.0 | Track-independent |
| **Informativeness Sol 1** (3) | 3.0 | 2.5-3.0 | AV features richer to describe |
| **Informativeness Sol 2** (3) | 3.0 | 3.0 | Both complex architectures |
| **Accurate Repr** (4) | 3.5-4.0 | 3.5-4.0 | Depends on verification protocol execution |
| **TOTAL** (40) | **35-38** | **35-38** | **Very close — performance will decide** |

## Decision Rules

### Primary: Dev Set Performance (after training both tracks)

1. **Compute gap over baseline for each solution:**
   - `gap_A = solution_f1 - baseline_f1` (Cat A over SVM, Cat B over LSTM)

2. **Compute combined gap:** `total_gap = gap_sol1 + gap_sol2`

3. **Compute McNemar's p-value** for each solution vs its baseline

### Decision Matrix

| Condition | Decision |
|-----------|----------|
| Both AV solutions significant AND both NLI solutions significant | Choose track with higher `total_gap` |
| Only one track has both solutions significant | Choose that track |
| AV total_gap > NLI total_gap by >0.05 | Choose AV |
| NLI total_gap > AV total_gap by >0.05 | Choose NLI |
| Gaps within 0.05 of each other | Choose NLI (lower execution risk, ESIM performance ceiling) |

### Tiebreaker: Creativity Self-Assessment

For each solution, honestly score creativity 1-3 based on:
- 1: Standard approach for this task domain
- 2: Draws from literature, combines known techniques
- 3: Goes beyond standard, novel combinations, task-specific innovations

Choose the track with higher combined creativity score.

### Fallback: Category C

If Cat B underperforms on either track (fails to beat LSTM baseline significantly):
- Swap to Cat A + Cat C for that track
- Cat C (DeBERTa) is the performance safety net

## Implementation Order

### Phase 1: Shared Infrastructure (Day 1)
- Project structure, data loading, scorer integration
- Load BOTH AV and NLI training data

### Phase 2: Category A — Both Tracks in Parallel (Days 1-3)
- AV Cat A: Feature engineering (all 9 groups including novel features) → stacking ensemble
- NLI Cat A: Feature engineering (all 9 tiers) → stacking ensemble
- **These share code patterns** — data_utils, ensemble framework, evaluation harness

### Phase 3: Category B — Both Tracks in Parallel (Days 2-4)
- AV Cat B: Adversarial Style-Content Disentanglement Network (char-CNN+BiLSTM+GRL)
- NLI Cat B: ESIM + KIM (BiLSTM + cross-attention + WordNet knowledge)
- **These share code patterns** — training loop, GRL implementation, evaluation

### Phase 4: Evaluate and Decide (Day 5)
- Run all solutions on dev sets
- Run full evaluation suite (McNemar's, confusion matrices, error analysis)
- Apply decision framework → choose track
- Generate test predictions for chosen track (test data released March 24)

### Phase 5: Category C Backup (Days 4-5, if needed)
- Train DeBERTa for the chosen track only
- Only if Cat B underperforms

### Phase 6: Polish (Days 6-9)
- Model cards (with verification protocol)
- Evaluation commentary
- Code documentation
- Poster
- Submission assembly

## Key Risk: GloVe Embeddings for NLI

NLI Cat B (ESIM) requires GloVe 840B 300d pre-trained embeddings. These are ~2GB to download. This is a general-purpose word embedding, NOT task-specific external data, so it is allowed under closed mode (same reasoning as DeBERTa being allowed for Cat C).

**Action:** Download GloVe early. If not available, fall back to GloVe 6B 300d (~1GB) or train word2vec on the NLI training data (worse but compliant).

## CSF3 GPU Strategy

- AV Cat B: ~1-3 hours on A100
- NLI Cat B (ESIM): ~1-2 hours on A100
- Cat C (DeBERTa): ~2-4 hours on A100
- **Total GPU time needed:** ~8-12 hours across all solutions
- **Strategy:** Submit as batch jobs, run overnight
