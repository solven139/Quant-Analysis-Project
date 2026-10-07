# Quant-Analysis-Project


Follow the steps below(we need not to use all the files:
Step 0: breadth data (one-time, shared by both universes)

1.wrds_breadth_pull.py pulls S&P 500 membership and constituent prices from WRDS.
2.breadth_data_prep.py builds sp500_breadth_pct_above_200dma.csv.

14-asset universe
3. condition_features.py is Step 2. It labels all 1,309 trades with the six conditions and writes condedge_trades_labeled_6cond_14assets.csv.
4. condedge_pricepath_mcpt.py is Step 4 (MCPT, N=168). It writes condedge_pricepath_mcpt_results_14assets.csv.
5. condedge_dsr.py is Step 5 (DSR). It reads the labeled bets from step 3, the MCPT results from step 4 and the price panel. It prints the survivor verdict.
6. condedge_metalabel.py is Step 7 (meta-labeling). It reads the labeled bets and breadth series and writes the rank, fold-result and importance CSVs.

10-sector universe
7. condedge_sector_universe_run.py runs Steps 2, 4 and 5 in one script for the sectors (N=120). It writes condedge_trades_labeled_6cond_10sectors.csv and the MCPT and DSR outputs.
8. condedge_metalabel_sector_universe_run.py is Step 7 for the sectors. It needs the labeled file from step 7.

Step 6 is only the sector universe run; there's no separate script for it.

All the test_*.py files are optional. They run on synthetic data and are worth running only after you change code.

Order constraints

condition_features.py must run before steps 4, 5 and 6, because they read its labeled-bets CSV.
condedge_dsr.py must run after condedge_pricepath_mcpt.py, because it merges the MCPT p-values.
The metalabel scripts must run after their universe's labeling step.
Steps 4 and 6 are independent of each other.
The 10-sector chain (7 and 8) is independent of the 14-asset chain (3–6) apart from sharing the breadth file and the yields file.
