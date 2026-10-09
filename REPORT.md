# Mini Project Report

Wenyin He (wenyinh) · Cosette Zhou (cosettez) · Vidhi Kothari (vidhikot) · Meet Bhatt (mbbhatt)

## 1. The Plant and the Task

The Tennessee Eastman Process (TEP) is a process-monitoring benchmark: a simulation of an Eastman Chemical plant, with the chemistry disguised. Feeds A, C, D, E and an inert B react to liquid products G and H and a by-product F. The effluent passes through a condenser, a separator and a stripper, and a compressor recycles unreacted gas while controllers hold temperatures, pressures and levels at their set points. We monitor 52 channels every 3 minutes: 41 measurements (`xmeas_1`–`xmeas_41`) and 11 manipulated variables (`xmv_1`–`xmv_11`).

The simulator has 20 fault types, including feed steps, a lost feed, random variation, sticking valves, a kinetics drift, and five unknown faults. We have 500 fault-free runs and 20 runs per fault. Each run has 500 samples (25 h); in faulty runs the fault starts after sample 20. Detectors are trained **only on normal data**. Fault-free runs are split into training (1–300), threshold validation (301–400) and false-alarm testing (401–500). We compare PCA ($T^2$, SPE) with a Ridge model that predicts each sample from the two before it.

## 2. The Two Detectors

Both detectors standardize all 52 channels with the training mean and sample standard deviation (`ddof=1`). Each threshold is the 99th percentile of the statistic on validation runs 301–400. A sample is an alarm only when it and the two samples before it in the same run are all strictly above the threshold.

**PCA.** PCA checks each sample against the normal correlation structure. We kept the smallest number of components that explains at least 90% of the training variance: **k = 31** (90.18%). With loadings $P$ ($52\times31$), scores $t = P^\top z_t$, and training score variances $\lambda_i$:

$$
T^2_t = \sum_{i=1}^{k}\frac{t_i^2}{\lambda_i}, \qquad
\mathrm{SPE}_t = \lVert z_t - P P^\top z_t \rVert^2 .
$$

$T^2$ measures movement *inside* the model space, and SPE measures what the model *cannot* rebuild. The thresholds are 51.80 ($T^2$) and 11.77 (SPE).

**Ridge.** A multi-output `Ridge(alpha=1.0)` predicts the current standardized sample (52 outputs) from the previous two (104 inputs). Inputs never cross runs, so the first two samples of a run are not scored. The score sums the squared prediction errors, each divided by that channel's training error standard deviation:

$$
s_t = \sum_{j=1}^{52}\left(\frac{z_{t,j}-\hat z_{t,j}}{\sigma_{e,j}}\right)^2, \qquad \text{threshold } 112.44 .
$$

![PCA T² and SPE for Fault 1, Run 1](results/pca_fault1_plot.png){width=5.6in}

![Ridge score for Fault 1, Run 1](results/ridge_fault1_plot.png){width=5.6in}

*Fault 1, Run 1. The fault starts after sample 20 (dashed thresholds). Ridge alarms at sample 24 (12 min), SPE at 26 (18 min) and $T^2$ at 29 (27 min); all three stay above threshold.*

## 3. Results

Each cell is the detection rate (% of post-fault samples in alarm, averaged over 20 runs) and, in parentheses, the median delay in minutes, $(\text{first alarm} - 20)\times 3$, over runs with a post-fault alarm. The best rate for each fault is in bold.

| Fault | Description | Ridge | PCA $T^2$ | PCA SPE |
| ---: | --- | ---: | ---: | ---: |
| 1 | A/C feed ratio, step | **99.4** (12) | 98.5 (25.5) | 99.2 (15) |
| 2 | B composition, step | 96.5 (51) | **97.3** (39) | 95.9 (57) |
| 3 | D feed temperature, step | 0.0 (—) | 0.0 (—)\* | 0.0 (—) |
| 4 | Reactor cooling-water inlet temp., step | 96.9 (9) | 9.1 (64.5) | **99.4** (12) |
| 5 | Condenser cooling-water inlet temp., step | **99.6** (9) | 33.2 (9) | 12.5 (15) |
| 6 | A feed loss | **99.6** (9) | 97.9 (33) | **99.6** (9) |
| 7 | C header pressure loss | **99.6** (9) | **99.6** (9) | **99.6** (9) |
| 8 | A, B, C feed composition, random | **95.5** (52.5) | 94.2 (70.5) | 82.2 (67.5) |
| 9 | D feed temperature, random | 0.0 (—) | 0.0 (—)\* | 0.0 (—) |
| 10 | C feed temperature, random | **69.8** (67.5) | 8.5 (471) | 17.7 (177) |
| 11 | Reactor cooling-water inlet temp., random | **45.4** (33) | 25.8 (43.5) | 43.4 (40.5) |
| 12 | Condenser cooling-water inlet temp., random | **95.7** (30) | 94.4 (37.5) | 86.7 (48) |
| 13 | Reaction kinetics, slow drift | **90.2** (138) | 88.4 (159) | 89.3 (136.5) |
| 14 | Reactor cooling-water valve, sticking | **99.5** (9) | 97.8 (12) | 78.4 (13.5) |
| 15 | Condenser cooling-water valve, sticking | 0.0 (—) | 0.0 (—)\* | 0.0 (—) |
| 16 | Unknown | **71.2** (45) | 1.9 (690)† | 10.1 (156) |
| 17 | Unknown | 86.7 (118.5) | 66.6 (138) | **86.8** (118.5) |
| 18 | Unknown | **89.3** (141) | 87.1 (175.5) | 88.4 (151.5) |
| 19 | Unknown | **2.9** (121.5) | 1.0 (471) | 0.3 (610.5)† |
| 20 | Unknown | **63.5** (151.5) | 13.6 (186) | 43.2 (165) |
| — | **False-alarm rate**, normal runs 401–500 | 0.00 | 0.052 | 0.00 |

\* $T^2$ alarmed in only one of 20 runs (231 min, rate 0.01–0.02%), no higher than its false-alarm rate, so the fault is missed. † No post-fault alarm in 3 runs for $T^2$ on Fault 16 and 6 runs for SPE on Fault 19. Every other nonzero cell had an alarm in all 20 runs.

## 4. Comparison

**$T^2$ versus SPE.** $T^2$ responds to normal patterns that are too large; SPE responds to new patterns. In Fault 4 the controller raises cooling-water flow (`xmv_10`) while holding reactor temperature (`xmeas_9`), so SPE catches 99.4% and $T^2$ only 9.1% (Russell et al. 2000, Table 4: Q missed 3.8%, $T^2$ missed 95.6%). In Fault 14 those channels swing in their usual opposite-sign pattern, only harder, so $T^2$ catches 97.8% and SPE 78.4%. The split depends on k, the thresholds and the data.

**Ridge versus PCA.** On large step and drift faults (1, 2, 6–8, 12, 13, 17, 18) the best PCA statistic is within a few points of Ridge. Ridge is much better when the fault is dynamic or compensated. On Fault 5, PCA alarms stop near sample 200 once the loops settle (`xmv_11` is only about 1.9 standard deviations off), while Ridge keeps alarming; Russell et al. (2000, pp. 90–91) report the same for PCA versus dynamic CVA. On Faults 10, 16, 19 and 20 each sample looks normal and only the time evolution changes, which memoryless PCA misses (their Q misses 55–87%; CVA $T_r^2$ misses 2–10%). Fault 19 stays weak for Ridge (2.9%).

## 5. Faults Nobody Catches

All three statistics miss **Faults 3, 9 and 15**: a D-feed temperature step, a D-feed temperature random variation, and a sticking condenser cooling-water valve. The loops absorb these disturbances, or the sensors barely register them. Russell, Chiang and Braatz ([2000](https://web.mit.edu/braatzgroup/36_Fault_detection_in_industrial_processes_using_canonical_variate_analysis_and_dynamic_principal_component_analysis.pdf), Table 4 and p. 90) saw the same: every PCA, dynamic PCA and CVA statistic missed 90.3–99.8% of samples, with no visible change in any variable's mean or variance. A monitor only finds faults that leave a trace in the measured channels.

## 6. Diagnosis

Contributions are averaged over post-fault alarm samples from the 20 runs: squared normalized prediction error for Ridge, and squared residual $(z_{t,j}-[PP^\top z_t]_j)^2$ for SPE. Channel meanings were checked against the [original TEP code](https://github.com/camaramm/tennessee-eastman-profBraatz/blob/master/teprob.f).

| Fault | Top Ridge channels | Top SPE channels | Matches the description? |
| --- | --- | --- | --- |
| 4: reactor CW inlet temp. step | `xmv_10` reactor CW flow (120.7); `xmeas_9` reactor temp. (5.6) | `xmv_10` (13.6); `xmeas_9` (11.7); next only 0.5 | **Yes, for both.** The temperature loop raises cooling-water flow to hold the temperature. |
| 6: A feed loss | `xmeas_1` A-feed flow (318,639); `xmv_3` A-feed valve (234,050) | `xmv_5` recycle valve (3744); `xmv_10` (1547); `xmeas_9` (1372); `xmeas_1` 5th (365) | **Ridge yes; SPE partly.** SPE ranks downstream effects above the root cause. |
| 14: reactor CW valve sticking | `xmeas_21` CW outlet temp. (121.8); `xmeas_9` (77.2); `xmv_10` (70.1) | `xmeas_21` (49.9); `xmeas_2` D-feed flow (12.9); `xmv_1` D-feed valve (12.7) | **Ridge yes; SPE mostly.** D-feed barely moves; it ranks high by residual smearing. |

Contributions show where a model is surprised, not what failed. SPE can smear a fault onto correlated channels.

## 7. Limits

Both models are linear and were trained on one operating mode of a simulation with stationary noise. A real plant changes throughput, grade and catalyst activity, and real sensors drift, so the near-zero false-alarm rates are optimistic. Each run has one fault with a known start. Settings were fixed, not tuned: k, the 99th-percentile thresholds, two lags and `alpha=1`. PCA is static; Ridge sees only two past samples.

## 8. AI Use

Wenyin used GenAI to help refactor the Ridge code, check the assignment requirements and numerical results, and draft the Ridge report sections. Vidhi used GenAI to draft and to condense the PCA sections, and to run sanity checks. Meet used GenAI to understand PCA and get reference code, and to convert this report from Markdown to PDF.

## Appendix: Member Contributions

- **Wenyin He — `wenyinh`.** Wrote `src/ridge.py` (standardization, within-run lags, Ridge, `results/ridge_model.npz`), `src/evaluate_ridge.py` and `src/diagnose_ridge.py`; produced the Ridge CSV rows, Makefile targets and report §§2–8.
- **Cosette Zhou — `cosettez`.** Wrote the first `src/run_forecast_score.py` (`results/scores_ridge.parquet`) and `src/plot_ridge_score.py` (`results/ridge_fault1_plot.png`).
- **Vidhi Kothari — `vidhikot`.** Wrote `src/pca.py` (checksums, split, standardization, k at 90% variance, $T^2$/SPE scores) and `make pca`; wrote the PCA sections and §1, and checked Fault 5 timing, Fault 14 smearing and Russell et al. (2000).
- **Meet Bhatt — `mbbhatt`.** Built the original PCA notebook `notebooks/PCA_v0.1.ipynb`. Wrote `src/evaluate_pca.py`, `src/diagnose_pca.py` and `src/plot_pca.py`, and produced the PCA CSV rows and `results/pca_fault1_plot.png`.
