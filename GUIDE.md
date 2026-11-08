# GUIDE: finish the project in 7 days (starting from basic Python)

Read this top to bottom once (15 min), then follow the day plan. The goal is not just to *run* the
code but to **understand every piece well enough to explain it in an interview**. Each day ends with a
"you should be able to explain" checklist, which is what a reviewer will actually ask.

## 0. The story your project tells

> "I built a chess engine and used it as a testbed to **measure** (1) how much each search technique
> saves, (2) whether an evaluation function *learned from data* beats a hand-written one, and
> (3) how playing strength scales with search depth, with proper statistics (confidence intervals)."

That is a research project: questions → experiments → numbers → honest limitations.
Do **not** present it as "I built a chess AI". Present the questions and the measurements.

## 1. Setup (Day 1, ~45 min)

1. Install Python 3.10 or newer (python.org). Check: `python --version` (on Mac/Linux it may be `python3`).
2. Unzip the project, open a terminal in the `chess-research-bot` folder, then:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

3. **Expected: `13 passed`.** The engine code was written without being executed against python-chess
   in my environment (it could not install there), so these tests are your first real check.
   **If any test fails, copy the full error text and send it to me; fixing it is quick.** Do not
   continue to experiments until all 13 pass.
4. Play your engine: `python scripts/play_cli.py --depth 3` (type moves like `e4` or `Nf3`).

## 2. Day-by-day plan

How long things take depends on your computer. Always **time a tiny run first** (e.g. `--pairs 2`),
then scale up.

### Day 1: Setup + understand the engine
- Do Section 1. Read `chessbot/evaluation.py` and `chessbot/search.py` slowly; run
  `python scripts/play_cli.py --depth 2` and `--depth 4` and notice the speed/strength difference.
- Concepts to learn today (30-40 min each; search these names on YouTube/Chessprogramming wiki):
  **minimax**, **negamax**, **alpha-beta pruning**, **iterative deepening**.
- You should be able to explain: what a "node" is; why alpha-beta returns the *same* answer as minimax
  but visits fewer nodes (that is exactly what `test_alpha_beta_gives_same_value_as_minimax` checks);
  why trying the best move first makes pruning stronger.

### Day 2: Experiment 1, search ablation (RQ1)
```bash
python scripts/exp_search_ablation.py --depths 3 4
```
- Outputs in `results/`: `search_ablation.txt` (table), `.csv` (raw), `.png` (bar chart).
- Read the table: "vs alpha-beta" column = nodes relative to plain alpha-beta (0.20x = 5x fewer).
  The "score differs" column should be 0 or very small: techniques must change *speed*, not answers.
  If it is not 0, mention and investigate (a transposition-table bug or repetition effects).
- Also read `search.py`'s `_order`, `_quiesce` and the TT code. Learn: **move ordering (MVV-LVA,
  killer, history)**, **Zobrist hashing / transposition table**, **quiescence search / horizon effect**.
- Write down: the 3-4 headline numbers for the README table.
- You should be able to explain: what each technique does and why it saves nodes; why we use the
  geometric mean of node counts (heavy-tailed data); what the horizon effect is.

### Day 3: Get data
**Option A (recommended, real human games):**
1. Go to https://database.lichess.org/ and download one **standard rated** monthly file
   (`lichess_db_standard_rated_YYYY-MM.pgn.zst`). Pick one of the **earliest months** because they are small;
   a few tens of MB is plenty. The data is CC0 (free to use, including in a portfolio). Save it in `data/`.
2. ```bash
   python scripts/build_dataset.py data/lichess_db_standard_rated_YYYY-MM.pgn.zst --max-games 30000 --min-elo 1500
   ```
   (Replace the file name. Processing takes a few minutes; it prints progress.) This creates
   `data/positions.npz` (features + labels) and `data/test_positions.fen`.

**Option B (fallback, no download):** the engine plays itself:
```bash
python scripts/gen_selfplay.py --games 400 --workers 4 --depth 2
python scripts/build_dataset.py data/selfplay.pgn
```
Self-play labels are noisier (weak, partly random play), so mention this in the report.

- Learn today: how a PGN file is structured; what a "quiet position" is and why we drop positions
  right after captures/checks (a static evaluation cannot judge a half-finished exchange).
- You should be able to explain: what one training example is (384 numbers + the game result);
  why we sample only a few positions per game; why we later split train/validation **by game**.

### Day 4: Train the evaluation (RQ2a)
```bash
python scripts/train_eval.py --data data/positions.npz --epochs 30
```
- Prints a table: coin-flip vs material-only vs learned (log-loss lower = better; sign-accuracy higher = better)
  measured on games the model never saw. Outputs `models/learned_pst.npy` and plots in `results/`.
- Look at `results/learned_pst.png`: does it make chess sense? (Typical: advanced pawns valued more,
  knights prefer the centre, kings prefer the corner in the middlegame.) Put 1-2 observations in the report.
- **Tune honestly:** try `--l2 0`, `--l2 2e-7`, `--l2 1e-6`, `--lr 0.5`. Choose the setting with the best
  **validation** loss, never by how well it looks on training data. Record all settings in a small table.
- Learn: **logistic regression**, **cross-entropy / log-loss**, **overfitting vs validation set**,
  **regularisation**, **data leakage** (why the split is by game).
- You should be able to explain: the formula P(win) = 1/(1+10^(-score/400)); why we start from material
  weights and pull toward them (L2); what the gradient step does; why log-loss rather than accuracy.

### Day 5: Matches (RQ2b and RQ3)
Time one tiny match first: `python scripts/run_match.py --a "eval=models/learned_pst.npy,depth=3" --b "eval=material,depth=3" --pairs 2`
then scale `--pairs` so the run fits in your time (use `--workers` = CPU cores minus 1; run long ones overnight).

```bash
# RQ2b-1: learned vs material, equal DEPTH
python scripts/run_match.py --a "eval=models/learned_pst.npy,depth=3" --b "eval=material,depth=3" --pairs 50 --workers 3 --tag learned_vs_material_depth3

# RQ2b-2: equal TIME (fairer: the learned eval is no slower here, but always check)
python scripts/run_match.py --a "eval=models/learned_pst.npy,depth=12,time=0.2" --b "eval=material,depth=12,time=0.2" --pairs 50 --workers 3 --tag learned_vs_material_time

# Does quiescence search help?
python scripts/run_match.py --a "eval=material,depth=3,quiescence=1" --b "eval=material,depth=3,quiescence=0" --pairs 50 --workers 3 --tag quiescence

# RQ3: strength vs depth ladder
python scripts/exp_depth_ladder.py --eval models/learned_pst.npy --max-depth 4 --pairs 20 --workers 3
```
- **How many games?** With ~100 games the 95% interval is roughly ±55 Elo; with ~400 games roughly ±27.
  If the interval includes 0, say "no significant difference", which is a legitimate result. Never claim a win from a small sample.
- You should be able to explain: why every opening is played twice with colours swapped; what the Elo
  formula means (64% score ≈ +100 Elo); what a confidence interval says; why "equal depth" and "equal time"
  can give different conclusions.

### Day 6: Write-up + repository
- Fill the README results table with **your** numbers. Copy figures from `results/`.
- Write the report with `REPORT_TEMPLATE.md` (3-5 pages, export to PDF, put it in the repo root).
- Create a GitHub repo, upload the project (`data/*.pgn`, `*.zst`, `*.npz` are git-ignored; keep
  `models/learned_pst.npy` and `results/`). Make sure someone else could reproduce by following the README.

### Day 7: Buffer, CV, interview practice, optional extras
- Re-run anything that failed; fix typos; re-check numbers in README match `results/*.json`.
- Add the CV lines (Section 4). Practise the questions in Section 5 out loud.
- Optional extras (only if everything above is done): use it in a chess GUI via UCI
  (`python -m chessbot.uci --eval models/learned_pst.npy --depth 4`, add as an engine in Cute Chess or Arena);
  try an extension from Section 6.

## 3. If something goes wrong

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: chess` | Virtual environment not active, or `pip install -r requirements.txt` not run |
| A test fails | Send me the full error text |
| `zstandard` error reading `.zst` | `pip install zstandard`, or decompress the file yourself and pass the `.pgn` |
| Matches take too long | Lower `--pairs` and `depth`, raise `--workers`, use `time=0.1` |
| Training loss does not go down | Check you have thousands of positions; try `--lr 0.5` |
| "No positions extracted" | `--min-elo` too high for that month; lower it |
| Windows: multiprocessing hangs | Run from a normal terminal (not a notebook); try `--workers 1` |

## 4. CV and profile text (fill in YOUR real numbers; delete anything you did not do)

**Project title:** *Chess Engine Research: Search Optimisation and Learned Evaluation (Python)*

- Implemented a chess engine (negamax alpha-beta, iterative deepening, MVV-LVA/killer/history move ordering,
  Zobrist transposition table, quiescence search); ablation study over [N] positions showed move ordering
  and transposition tables reduced nodes searched by [X]x at depth [d] with identical results.
- Trained a piece-square evaluation on [N] positions from [M] Lichess games via logistic regression on
  game outcomes (Texel tuning, split by game); reduced validation log-loss from [a] to [b].
- Measured strength with [n] colour-swapped games per experiment: learned evaluation scored [x]% vs a material
  baseline (Elo [±k], 95% CI [lo, hi]); quantified Elo gain per extra search ply.
- Skills line: Python, NumPy, Matplotlib, algorithm analysis, experiment design, statistics.

Say "linear evaluation trained with logistic regression", **not** "neural network" or "AI trained on
millions of games" unless that is what you did.

## 5. Interview questions you should be able to answer

1. **What is alpha-beta pruning?** It skips branches that cannot affect the final minimax decision: once
   one reply refutes a move, there is no need to see the opponent's other replies. Same result, fewer nodes.
2. **Why does move ordering matter?** Pruning is most effective when the best moves are searched first.
   With perfect ordering alpha-beta needs about the square root of minimax's nodes.
3. **What is a transposition table?** A hash table mapping a position (Zobrist hash) to its searched
   score/depth/best move, so positions reached by different move orders are not searched again.
4. **What is quiescence search / the horizon effect?** Stopping the search mid-capture-sequence gives
   wrong evaluations (e.g. winning a queen that is recaptured next move). Quiescence keeps searching
   captures until the position is quiet.
5. **How did you learn the evaluation?** Predict the game result from a position: P = 1/(1+10^(-s/400)),
   minimise cross-entropy with Adam, regularise toward material values, validate on unseen games.
6. **Why split by game?** Positions from the same game are correlated; splitting by position would leak
   information and inflate validation performance.
7. **How do you know the learned evaluation is better?** Two ways: lower validation loss on unseen games
   and match results with confidence intervals. (If the interval includes 0, you cannot claim it.)
8. **Why play each opening twice with colours swapped?** It cancels the first-move advantage and
   opening luck, reducing variance.
9. **What are the limitations?** Linear model, noisy labels (game results), pure-Python speed, relative Elo only.
10. **What would you do next?** Train on Stockfish evaluations, try a small neural network (NNUE-style),
    learn move ordering, rewrite the search core in C++/Rust.
11. **Why log-loss instead of accuracy?** It measures confidence calibration and gives smooth gradients.
12. **Why the geometric mean for nodes?** Node counts vary by orders of magnitude between positions;
    the geometric mean summarises ratios fairly.

## 6. Extensions (ranked by value for effort)

1. **Hyperparameter study** (cheap, rigorous): plot validation loss vs `--l2` and vs dataset size
   (1k, 5k, 20k, 50k positions): a data-scaling curve is a classic data-science figure.
2. **More features**: add mobility, pawn structure or king-safety features to `feature_vector`, retrain,
   and test whether validation loss and match results improve.
3. **Stockfish evaluations as labels**: the Lichess evaluation database
   (`lichess_db_eval.jsonl.zst`, hundreds of millions of positions) gives a much cleaner target than game
   results, but it is a very large file; read only the first N lines.
4. **Self-improvement loop**: generate games with the learned engine, retrain, and measure Elo per iteration.
5. **Small neural network** in NumPy/PyTorch replacing the linear model (slower inside a Python search).

## 7. Honesty checklist before you submit

- Every number in the README/CV is in a file under `results/` and was produced by your own run.
- Every technique you list, you can explain in your own words (practise Section 5).
- You state that Elo values are relative to your own engines, not to FIDE or Lichess ratings.
- The repo contains the exact commands to reproduce the results.
