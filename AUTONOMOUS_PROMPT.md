# Master Prompt — Paste This Into a Fresh Claude Code Session

```
You are working on COMP34812 NLU Coursework at the University of Manchester. This is worth 50% of the unit grade (40 marks). The deadline is 2026-03-31 14:00. Time is NOT a constraint — optimize purely for maximizing the grade (target 95-100%).

## What Exists Already

Read these files first — they contain exhaustive implementation plans:

1. `CLAUDE.md` — All development conventions, CSF3 setup, rubric, baselines, git rules
2. `IMPLEMENTATION_PLAN.md` — AV (Authorship Verification) track plan, 974 lines, 3 solutions
3. `NLI_IMPLEMENTATION_PLAN.md` — NLI (Natural Language Inference) track plan, 1,363 lines, 3 solutions
4. `TRACK_DECISION_FRAMEWORK.md` — How to compare and choose between tracks

Read all four files COMPLETELY before starting any work.

## What You Must Build

Build BOTH tracks (AV and NLI), all 3 category solutions per track (6 total), evaluate on dev sets, and submit the best track's top 2 solutions. The full deliverables for the winning track:

1. Two prediction CSV files (Category A + Category B)
2. Training notebooks (one per solution)
3. Evaluation notebook (confusion matrices, McNemar's, error analysis, ablation, all with written commentary)
4. Demo/inference notebooks (one per solution)
5. Two model cards (using provided template at `archive_extracted/COMP34812_modelcard_template.md`)
6. Poster (A1 PDF)
7. README.md
8. All trained models saved

## How to Work — Self-Recursive Loop

Follow this cycle for every task. NEVER skip steps:

1. **PLAN** — Read the relevant implementation plan section. If anything is unclear, spin up a research agent team.
2. **CODE** — Write the code. Commit AND push every ~10 lines changed. No Co-Authored-By.
3. **VERIFY** — Run the code. Check outputs are correct. If not, debug and fix (commit each fix).
4. **TEST** — Write minimal tests. Run them. Commit.
5. **EVALUATE** — For ML code: run on dev set, compare against baselines, check statistical significance.
6. **ITERATE** — If below target performance, spin up agent teams to research improvements. Implement the best ideas. Repeat from step 2.
7. **COMPLY** — At major milestones, spin up a compliance agent to verify: rubric coverage, spec requirements, naming conventions, file paths, model card accuracy vs code.
8. **NEXT** — Move to next task. If not obvious what's next, spin up an agent team to determine priorities.

## Agent Teams

When you need research or decisions, spin up specialized agent teams. Examples:

- **NLU Expert + ML Expert**: For architecture decisions, feature engineering choices
- **Evaluation Expert**: For designing evaluation beyond standard metrics
- **Literature Research Agent**: For finding papers, verifying citations, checking approach novelty
- **Compliance Agent**: For checking everything matches the coursework spec and rubric

Agents should be given specific, bounded tasks with clear deliverables.

## Git Rules — CRITICAL

- Push after EVERY commit (micro-commits, ~10 lines each)
- NO Co-Authored-By lines — commits should show only the repo owner
- Short imperative commit messages
- Never amend — always new commits
- The user expects 3,000-4,000 total commits

## CSF3 (GPU Cluster)

SSH is set up: `ssh csf3` works with no authentication (7-day socket). If it fails, re-establish using the expect script documented in the memory files.

- Project dir: `~/scratch/nlu-project/` (data already uploaded)
- Conda env: `nlu` (Python 3.11, PyTorch, sklearn, transformers, spaCy, xgboost, etc.)
- GPU jobs: `#SBATCH -p gpuA` for A100 80GB, max 4-day walltime
- Activate env: `module load apps/binapps/conda/miniforge3/25.3.0 && conda activate nlu`

Submit GPU training as batch jobs. Monitor with `squeue -u r36859ak`. Category A (traditional ML) runs locally or on CPU nodes. Categories B and C need GPU.

## Implementation Order

Follow `TRACK_DECISION_FRAMEWORK.md` Phase ordering:

### Phase 1: Shared Infrastructure (first)
- Project directory structure (as specified in both implementation plans)
- Data loading and preprocessing utilities
- Scorer integration (wrapper around local scorer)
- Evaluation harness (confusion matrix, McNemar's, bootstrap CI, etc.)

### Phase 2: Category A — Both Tracks
- AV Cat A: ~950 stylometric features + stacking ensemble
- NLI Cat A: ~280 features (alignment, natural logic, WordNet) + stacking ensemble

### Phase 3: Category B — Both Tracks
- AV Cat B: Adversarial Style-Content Disentanglement Network (char-CNN+BiLSTM+GRL)
- NLI Cat B: ESIM + KIM (BiLSTM + cross-attention + WordNet knowledge)

### Phase 4: Evaluate and Decide
- Run all solutions on dev sets
- Full evaluation suite
- Apply decision framework → choose track

### Phase 5: Category C Backup (if needed)
### Phase 6: Polish — model cards, evaluation commentary, documentation, poster

## Start Now

Begin with Phase 1. Read the plans, set up the project structure, and start coding. Commit and push everything. Work autonomously through the full self-recursive loop until all deliverables are complete.
```
