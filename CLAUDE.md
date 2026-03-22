# COMP34812 NLU Coursework — Autonomous Development Instructions

## Project Goal
Maximize marks (target 95-100%) on COMP34812 NLU Coursework (40 marks, 50% of unit grade). Build TWO tracks (AV + NLI), Categories A+B+C for each, evaluate both, submit the best. Deadline: 2026-03-31 14:00.

## Plans
- `IMPLEMENTATION_PLAN.md` — AV track (974 lines, all 3 categories)
- `NLI_IMPLEMENTATION_PLAN.md` — NLI track (1,363 lines, all 3 categories)
- `TRACK_DECISION_FRAMEWORK.md` — Comparison rubric, decision rules, phasing

## Git Conventions — CRITICAL
- **NO Co-Authored-By lines.** Commits should show only the user's name (ayush-kumar-prog).
- **Micro-commits.** Every 10-line change gets its own commit and push. Every file creation, every function written, every bug fix, every config change — commit immediately.
- **Commit messages:** Short, descriptive, imperative. e.g. "Add char n-gram feature extractor", "Fix Burrows Delta guard for short texts", "Update ESIM hidden size to 300"
- **Push after every commit.** The user expects 3,000-4,000 commits by the end.
- **Never amend commits.** Always create new ones.

## Self-Recursive Development Loop
Follow this loop for EVERY piece of work:

```
1. PLAN    → Read the relevant section of the implementation plan
2. CODE    → Write the code (micro-commit each piece)
3. VERIFY  → Run the code, check outputs match expectations
4. TEST    → Write and run tests
5. EVALUATE → Compare performance metrics against baselines
6. ITERATE → If performance is below target, research improvements
7. COMPLY  → Check against rubric and spec requirements
8. NEXT    → Move to next task or spin up research agents if unclear
```

When things are NOT obvious (e.g. hyperparameter choices, architecture decisions):
- Spin up agent TEAMS (not just independent sub-agents) to research, hypothesize, and test experiments
- Agents should be specialized (NLU expert, ML expert, fine-tuning expert, evaluation expert)
- Agents MUST communicate findings to each other — one agent's output should inform another's work
- Agent characteristics should VARY based on the task (e.g. a post-training specialist for fine-tuning decisions, an evals specialist for metrics design, a literature researcher for novelty checks)
- This is an agent TEAM, not a collection of independent workers

## Agent Team Specifications
Use specialized agents based on the task:

- **NLU/NLP Expert**: For task-specific design decisions (AV stylometry, NLI entailment patterns, feature selection)
- **ML Engineering Expert**: For model architecture, training loops, optimization, regularization
- **Evaluation Expert**: For metrics, statistical tests, error analysis, ablation design
- **Research Agent**: For literature review, finding papers, comparing approaches
- **Compliance Agent**: Run at major milestones to verify rubric coverage, spec requirements, naming conventions, file paths, model card accuracy

## CSF3 (Manchester HPC)
- SSH: `ssh csf3` (7-day ControlMaster socket, no re-auth needed)
- Project dir: `~/scratch/nlu-project/`
- Data uploaded: `~/scratch/nlu-project/av/` and `~/scratch/nlu-project/nli/`
- Conda env: `nlu` (Python 3.11, PyTorch 2.5.1+cu121, sklearn, transformers, spaCy, xgboost, lightgbm, etc.)
- GPU partition: `#SBATCH -p gpuA` (A100 80GB), max 4 days walltime
- Activate: `module load apps/binapps/conda/miniforge3/25.3.0 && conda activate nlu`

## Rubric (40 marks)
- System Predictions: 6 (3+3, must be statistically significant over baseline)
- Organisation & Documentation: 3
- Completeness & Reproducibility: 3
- Soundness: 6 (3+3)
- Creativity: 6 (3+3, "beyond typical standard approaches")
- Evaluation: 3 (beyond just Codabench — needs commentary)
- Model Card Formatting: 3
- Model Card Informativeness: 6 (3+3)
- Model Card Accurate Representation: 4 (HIGHEST SINGLE CRITERION)

## Baselines (macro_f1)
- AV: SVM=0.5610, LSTM=0.6226, BERT=0.7854
- NLI: SVM=0.5846, LSTM=0.6603, BERT=0.8198

## Local Scorer
`cd baseline_extracted/nlu_bundle-feature-unified-local-scorer && python3 -m local_scorer.main --task {av|nli} --prediction path/to/file.csv`

## AI Tool Declaration — REQUIRED BY SPEC
The README MUST include a section called **"Use of Generative AI Tools"** describing which AI tools were used and for what purposes. This is a university requirement (Section VI of spec). Failure to declare = academic malpractice.

## Key Constraints
- Closed mode: only provided training data, no external datasets
- Pre-trained models (GloVe, BERT, DeBERTa, spaCy) are allowed
- Must submit exactly 2 solutions from 2 DIFFERENT categories (A/B/C)
- Statistical significance test (McNemar's) is REQUIRED for performance marks
- Model cards must EXACTLY match implementations
- Time is NOT a constraint — optimize purely for grade
- Prediction files named `Group_n_A.csv` and `Group_n_B.csv` (n = Canvas group number)
- All deliverables compressed into ONE zip file for Canvas upload
- Models >10MB must NOT be in the zip — store on OneDrive, link in README
- Code attribution required — acknowledge any reused code or face academic malpractice
