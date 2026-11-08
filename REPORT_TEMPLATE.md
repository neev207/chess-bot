# Report template (aim for 3-5 pages; export to PDF and put it in the repo)

**Title:** Measuring Search Enhancements and a Learned Evaluation in a Python Chess Engine

**Abstract (4-5 sentences).** What you built, the three questions, the headline numbers, one limitation.
> Example shape: "I implemented a chess engine with alpha-beta search and evaluated (1) the node savings of
> move ordering and transposition tables, (2) a piece-square evaluation learned from N Lichess positions,
> and (3) strength as a function of depth. Move ordering reduced nodes by __x at depth __; the learned
> evaluation reduced validation log-loss from __ to __ and scored __% (Elo __ ± __) against a material
> baseline at equal depth. Results are relative to my own engines and limited by pure-Python speed."

## 1. Introduction (½ page)
- Why chess is a good testbed (clear objective, measurable strength, classic search + learning problem).
- Your 3 research questions, one sentence each.

## 2. Methods (1 page)
- **2.1 Search**: negamax, alpha-beta, iterative deepening, each technique in one sentence.
  Explain *why* move ordering matters: alpha-beta prunes most when the best move is tried first.
- **2.2 Evaluation**: linear piece-square model, 384 parameters, material baseline, colour-mirroring.
- **2.3 Learning**: P(win) = 1/(1+10^(-score/400)), cross-entropy loss against results, Adam,
  L2 regularisation toward material, quiet-position filtering, split by game.
- **2.4 Strength measurement**: colour-swapped pairs, random openings, Elo and 95% CI formulas.

## 3. Experimental setup (½ page)
- Data: source (Lichess database, month, rating filter, CC0 licence), number of games/positions, split sizes.
- Hardware, Python version, seeds, time per experiment. Everything needed to reproduce.

## 4. Results (1-1.5 pages)
- **RQ1** table + `results/search_ablation.png`. State node ratios and that scores were identical.
- **RQ2** validation table (coin-flip / material / learned) + `results/training_curve.png`
  + `results/learned_pst.png` (what did it learn? e.g. advanced pawns, central knights, king safety).
  Then match results at equal depth and equal time with CIs.
- **RQ3** `results/depth_ladder.png`.

## 5. Discussion and limitations (½ page)
- What the results mean; surprises; whether CIs include zero (be honest!).
- Limitations (see README) and 2-3 concrete next steps.

## 6. Reproducibility
- Exact commands and seeds (copy from GUIDE.md), repo link.
