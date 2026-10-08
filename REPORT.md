# Mini Project Report

Group Member:

Wenyin He(wenyinh)

Cosette Zhou(cosettez)

Vidhi Kothari(vidhikot)

Meet Bhatt(mbbhatt)

## 1. The Plant and the Task

The Tennessee Eastman Process (TEP) is a widely used benchmark for process monitoring. It is a computer simulation of a real chemical plant at the Eastman Chemical Company, with the chemistry and operating conditions disguised for proprietary reasons. Gaseous feeds (A, C, D, E and an inert B) react in a reactor to form two liquid products (G and H) and a by-product (F). The reactor effluent is cooled in a condenser, split in a vapor–liquid separator and cleaned in a stripper, and unreacted gas is sent back to the reactor by a recycle compressor. A control system runs the plant at all times, adjusting valves and flows to keep temperatures, pressures and levels at their set points. We monitor 52 channels sampled every 3 minutes: 41 measurements (flows, temperatures, pressures, levels and gas compositions, named `xmeas_1`–`xmeas_41`) and 11 manipulated variables, which are the controllers' valve and flow settings (`xmv_1`–`xmv_11`).

The simulator can introduce 20 types of faults (numbered 1–20). Examples include a step change in a feed composition, a loss of the A feed, random variation in a cooling-water temperature, a sticking valve, a slow drift in reaction kinetics, and five faults of unknown type. We have 500 fault-free runs and, for each fault, 20 faulty runs. Every run has 500 samples (25 hours), and in the faulty runs the fault starts after sample 20. Our task is to detect faults using **only normal data for training**, as in a real plant where examples of every future fault are not available. Each detector learns what normal operation looks like and raises an alarm when a new sample does not fit. We split the fault-free runs into training (1–300), threshold validation (301–400) and false-alarm testing (401–500). We compare two detectors: a PCA model with two statistics ($T^2$ and SPE), and a Ridge model that predicts each sample from the two before it. For each fault we report the detection rate, the median detection delay and the false-alarm rate on normal data. We then ask which faults each method catches or misses, why, and which channels drive the alarms.

## 2. The Two Detectors

### Ridge forecast monitor

We trained a multi-output `Ridge(alpha=1.0)` model on normal runs 1–300. We used the training mean and sample standard deviation (`ddof=1`) to standardize all 52 channels. The model uses the previous two samples to predict the current sample. This gives 104 inputs and 52 outputs. We built the inputs separately for each run, so they never include data from another run. The first two samples of each run have no score.

For each channel, we calculated the prediction error and divided it by the standard deviation of that channel's training errors. We then squared these values and added them to get the Ridge score:

$$
s_t = \sum_{j=1}^{52}\left(\frac{z_{t,j}-\hat z_{t,j}}{\sigma_{e,j}}\right)^2.
$$

We set the threshold to the 99th percentile of scores on normal validation runs 301–400, which was **112.4423**. A sample was marked as an alarm only if its score and the previous two scores in the same run were all strictly above this threshold. We used normal runs 401–500 to check false alarms.

![Ridge score and validation threshold for Fault 1, Run 1](results/ridge_fault1_plot.png)

*Ridge score for Fault 1, Run 1. The fault starts after sample 20. The first alarm is at sample 24, giving a delay of 12 minutes.*

### PCA monitor

PCA looks at each sample on its own and checks whether the 52 channels still move together the way they did in normal operation. We used the same split and scaling as Ridge: we computed the mean and sample standard deviation (`ddof=1`) of each channel on normal runs 1–300 and used them to standardize every run. We fitted PCA on the standardized training data with the exact (`full`) SVD solver. We kept the smallest number of components whose eigenvalues explain at least 90% of the training variance, which gave **k = 31** components (90.18%). The other 21 directions carry only 10% of the normal variance, so they should stay close to zero when the plant is healthy.

Each standardized sample $z_t$ is split into a part inside the 31-dimensional model space and a residual part outside it. Let $P$ be the $52 \times 31$ matrix of loadings, $t = P^\top z_t$ the scores, and $\lambda_i$ the variance of score $i$ in training. We computed two statistics:

$$
T^2_t = \sum_{i=1}^{k}\frac{t_i^2}{\lambda_i}, \qquad
\mathrm{SPE}_t = \lVert z_t - P P^\top z_t \rVert^2 .
$$

$T^2$ measures how far the sample is from normal *inside* the model space. It grows when the variables move in a normal pattern but by an abnormally large amount. Dividing by $\lambda_i$ means a move along a quiet direction counts more than the same move along a noisy one. SPE (also called the Q statistic) measures the part of the sample the model *cannot* rebuild. It grows when the usual relationships between channels break, even if each channel is still in its normal range. Keeping all 52 components would rebuild every sample exactly and make SPE zero, which is why $k < 52$ matters.

We set each threshold to the 99th percentile of its statistic on normal validation runs 301–400: **51.7971** for $T^2$ and **11.7745** for SPE. We used the same alarm rule as Ridge: an alarm needs the current value and the two values before it in the same run to be strictly above the threshold. Normal runs 401–500 were used only to measure false alarms.

![PCA T² and SPE with validation thresholds for Fault 1, Run 1](results/pca_fault1_plot.png)

*PCA $T^2$ (top) and SPE (bottom) for Fault 1, Run 1. The fault starts after sample 20. SPE first alarms at sample 26 (18 min delay) and $T^2$ at sample 29 (27 min delay). Both stay far above their thresholds for the rest of the run.*

## 3. Results

### Ridge results

Ridge had no false alarms in the normal test runs. None of the 49,800 scored samples met the alarm rule, so the false-alarm rate was **0.00%**. In the table, detection rate is the percentage of samples after sample 20 that were in alarm, averaged over the 20 runs. We calculated delay as `(first alarm sample − 20) × 3` minutes and report the median for runs with a post-fault alarm. The last column counts runs with no post-fault alarm. A dash means all runs were missed.

| Fault | Detection rate (%) | Median delay (min) | Runs missed / 20 |
| --- | ---: | ---: | ---: |
| 1 | 99.35 | 12 | 0 |
| 2 | 96.51 | 51 | 0 |
| 3 | 0.00 | — | 20 |
| 4 | 96.89 | 9 | 0 |
| 5 | 99.58 | 9 | 0 |
| 6 | 99.58 | 9 | 0 |
| 7 | 99.58 | 9 | 0 |
| 8 | 95.45 | 52.5 | 0 |
| 9 | 0.00 | — | 20 |
| 10 | 69.76 | 67.5 | 0 |
| 11 | 45.35 | 33 | 0 |
| 12 | 95.69 | 30 | 0 |
| 13 | 90.18 | 138 | 0 |
| 14 | 99.54 | 9 | 0 |
| 15 | 0.00 | — | 20 |
| 16 | 71.19 | 45 | 0 |
| 17 | 86.69 | 118.5 | 0 |
| 18 | 89.27 | 141 | 0 |
| 19 | 2.92 | 121.5 | 0 |
| 20 | 63.51 | 151.5 | 0 |

### PCA results

The PCA statistics were scored on all 500 samples of each run, so the normal test set has 50,000 samples. SPE had no alarms on it, a false-alarm rate of **0.00%**. $T^2$ had 26 alarm samples, a false-alarm rate of **0.052%**. Both are well below the 1% that a 99th-percentile threshold allows on single samples, because the three-in-a-row rule removes isolated spikes. Detection rate, median delay and missed runs are defined in the same way as for Ridge. A dash means no run had a post-fault alarm.

| Fault | $T^2$ rate (%) | $T^2$ median delay (min) | $T^2$ runs missed / 20 | SPE rate (%) | SPE median delay (min) | SPE runs missed / 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 98.54 | 25.5 | 0 | 99.19 | 15 | 0 |
| 2 | 97.28 | 39 | 0 | 95.90 | 57 | 0 |
| 3 | 0.02 | 231 | 19 | 0.00 | — | 20 |
| 4 | 9.14 | 64.5 | 0 | 99.42 | 12 | 0 |
| 5 | 33.16 | 9 | 0 | 12.54 | 15 | 0 |
| 6 | 97.88 | 33 | 0 | 99.58 | 9 | 0 |
| 7 | 99.59 | 9 | 0 | 99.58 | 9 | 0 |
| 8 | 94.19 | 70.5 | 0 | 82.19 | 67.5 | 0 |
| 9 | 0.01 | 231 | 19 | 0.00 | — | 20 |
| 10 | 8.49 | 471 | 0 | 17.66 | 177 | 0 |
| 11 | 25.77 | 43.5 | 0 | 43.36 | 40.5 | 0 |
| 12 | 94.41 | 37.5 | 0 | 86.72 | 48 | 0 |
| 13 | 88.45 | 159 | 0 | 89.33 | 136.5 | 0 |
| 14 | 97.81 | 12 | 0 | 78.39 | 13.5 | 0 |
| 15 | 0.02 | 231 | 19 | 0.00 | — | 20 |
| 16 | 1.94 | 690 | 3 | 10.11 | 156 | 0 |
| 17 | 66.59 | 138 | 0 | 86.78 | 118.5 | 0 |
| 18 | 87.08 | 175.5 | 0 | 88.44 | 151.5 | 0 |
| 19 | 0.99 | 471 | 0 | 0.33 | 610.5 | 6 |
| 20 | 13.63 | 186 | 0 | 43.22 | 165 | 0 |

For Faults 3, 9 and 15, $T^2$ alarmed after the fault in only one of the 20 runs, with an overall rate of 0.01–0.02%. That is no higher than its false-alarm rate on normal data, so we count these faults as missed.

## 4. Comparison

### Ridge response

Ridge checks how much the current values differ from its predictions based on the previous two samples. Ten faults had detection rates above 90%. For Fault 19, every run had at least one alarm, but only 2.92% of samples after the fault were in alarm. The median delay was 121.5 minutes. Ridge detected this fault at least once in every run, but most samples after the fault were not in alarm.

### PCA response, and PCA versus Ridge

**$T^2$ versus SPE.** Faults 4 and 14 show the difference between the two PCA statistics most clearly. Both faults involve the reactor cooling water. In Fault 4 (step in cooling-water inlet temperature) the temperature controller raises the cooling-water flow (`xmv_10`) and holds the reactor temperature (`xmeas_9`) near its set point. Only one channel moves, and it moves without its usual partner. That breaks a normal correlation, so SPE catches 99.4% of samples while $T^2$ catches only 9.1%. Russell et al. (2000, Table 4, Fig. 2) found the same for Fault 4: PCA Q missed 3.8% of samples and PCA $T^2$ missed 95.6%. In Fault 14 (sticking cooling-water valve) the reactor temperature and the cooling-water flow both swing strongly, in the same opposite-sign pattern seen in normal operation, only much larger. That is a large move *inside* the model space, so $T^2$ catches 97.8% and SPE 78.4%. Here our result differs from Russell et al. (2000, Table 4): for Fault 14 their PCA Q missed no samples and their $T^2$ missed 15.8%. Their setup is different. They chose a different number of components, set thresholds from distributional formulas and later adjusted them, and used a different data set. The split between $T^2$ and SPE depends on which directions are kept in the model, so it can change with these choices. In short, $T^2$ responds to normal patterns that are too large, and SPE responds to new patterns. Faults 10, 11, 16, 17 and 20 also favour SPE, while Faults 2, 5, 8 and 12 favour $T^2$.

**Faults Ridge catches and PCA misses.** Ridge does much better on Faults 5, 10, 16 and 20 (99.6%, 69.8%, 71.2% and 63.5%, against at most 33%, 18%, 10% and 43% for the best PCA statistic).

- *Fault 5* is a step in condenser cooling-water inlet temperature. In our data the PCA alarms stop at about sample 200, roughly nine hours after the fault starts. By then the control loops have settled at a new steady state in which most channels are back near normal. The condenser cooling-water flow (`xmv_11`) stays shifted by only about 1.9 training standard deviations, which is not enough to cross either PCA threshold. Russell, Chiang and Braatz (2000, Table 4 and pp. 90–91) report the same behavior. Their PCA $T^2$ and Q statistics miss 77.5% and 74.6% of Fault 5 samples, and the Q statistic drops below its threshold about 10 h after the fault starts. Dynamic PCA misses just as much, and only their state-space CVA statistics catch the fault throughout. Ridge keeps alarming, so the new steady state still breaks the sample-to-sample relationships it learned from normal data.
- *Faults 10, 16 and 20* (and Fault 19 to a lesser degree) are random-variation or unknown faults. In each sample the values stay mostly inside the normal range and the normal correlation pattern. What changes is how values evolve over time. Static PCA treats each sample independently and has no memory, so it cannot see this. Ridge predicts each sample from the previous two, so a change in the dynamics shows up directly as prediction error. Russell et al. (2000, Table 4) show the same split. PCA Q misses 65.9%, 75.5%, 87.3% and 55.0% of samples for Faults 10, 16, 19 and 20, while their residual CVA statistic $T_r^2$ misses only 9.9%, 8.4%, 1.9% and 8.7%. CVA is a dynamic method that, like Ridge, models how the plant moves over time.

**Faults PCA catches as well as Ridge.** On large step faults (1, 2, 6, 7, 8, 12, 13, 17, 18) the best PCA statistic is within a few points of Ridge. These faults move the plant far enough that both a static and a dynamic model notice. Ridge is usually faster. For example, on Fault 1 its median delay is 12 min, against 15 min for SPE and 25.5 min for $T^2$. On Fault 2 $T^2$ is faster than Ridge (39 vs 51 min). On Faults 2 and 7 $T^2$ has a slightly higher detection rate than Ridge.

## 5. Faults Nobody Catches

### Ridge misses

Ridge missed Faults 3, 9, and 15 in all 20 runs. Fault 3 is a step change in D-feed temperature, and Fault 9 is random variation in the same temperature. Fault 15 is a sticking condenser cooling-water valve. [Russell, Chiang and Braatz (2000), Table 1 and p. 90](https://web.mit.edu/braatzgroup/36_Fault_detection_in_industrial_processes_using_canonical_variate_analysis_and_dynamic_principal_component_analysis.pdf) also found these faults hard to detect, with little visible change in the means or variances of individual variables. Our findings are similar, but the methods and settings differ, so the numbers cannot be compared directly.

### PCA misses

PCA also misses Faults 3, 9 and 15. SPE never alarmed after the fault in any run. $T^2$ alarmed in one run out of 20, at a rate no higher than on normal data. So **all three statistics miss the same three faults**. A static model (PCA) and a dynamic model (Ridge) both fail, and they use very different information. This points to the data rather than the detectors: the effect of these faults is too small to separate from normal noise, either because the control loops absorb the disturbance (Faults 3 and 9 act on the D-feed temperature) or because the sticking valve barely changes what the sensors record (Fault 15). This matches Russell, Chiang and Braatz (2000, Table 4 and p. 90). For Faults 3, 9 and 15, every PCA, dynamic PCA and CVA statistic missed between 90.3% and 99.8% of samples, which is close to what the threshold alone would flag on normal data. The authors also found no visible change in the mean or variance of any variable compared with fault-free data, and left these three faults out of their comparison for that reason. Data-driven monitoring can only find faults that leave a trace in the measured channels. These three leave almost none.

## 6. Diagnosis

### Ridge contributions

For each fault, we collected all alarm samples after sample 20 from the 20 runs. We averaged each channel's squared normalized residual over these samples and saved the top five channels. Faults 3, 9, and 15 have no ranking because they had no post-fault alarms. We used the [original TEP simulation code](https://github.com/camaramm/tennessee-eastman-profBraatz/blob/master/teprob.f) to check the channel meanings and fault definitions.

| Fault | Main Ridge contributions | Physical interpretation |
| --- | --- | --- |
| 4: reactor cooling-water inlet temperature step | `xmv_10` (120.74): reactor cooling-water flow input; `xmeas_9` (5.64): reactor temperature | The main channels are related to reactor cooling. The large contribution from `xmv_10` may come from the controller responding to the temperature change. |
| 6: loss of A feed | `xmeas_1` (318638.93): measured A-feed flow; `xmv_3` (234049.54): manipulated A-feed input | Both channels measure or control the A feed, which matches the fault. |
| 14: reactor cooling-water valve sticking | `xmeas_21` (121.85): cooling-water outlet temperature; `xmeas_9` (77.19): reactor temperature; `xmv_10` (70.08): cooling-water flow input | These channels suggest a change in heat removal when the cooling-water valve sticks. |

The contributions show which channels had large normalized prediction errors. They do not prove the root cause. Because the errors are squared, they also do not show whether a variable increased or decreased.

### PCA (SPE) contributions

For SPE, a channel's contribution is its squared residual $(z_{t,j} - [PP^\top z_t]_j)^2$. These contributions add up to SPE. We averaged them over every post-fault SPE alarm sample from the 20 runs and kept the top five. We used the same three faults as Ridge so that the two can be compared.

| Fault | Main SPE contributions | Matches the description? |
| --- | --- | --- |
| 4: reactor cooling-water inlet temperature step | `xmv_10` (13.63): reactor cooling-water flow; `xmeas_9` (11.69): reactor temperature; next channel only 0.50 | **Yes.** Almost all of SPE comes from the two channels of the reactor temperature loop. This is the same pair that Ridge ranked first. The controller raising cooling-water flow while the reactor temperature holds is exactly the broken correlation SPE responds to. |
| 6: loss of A feed | `xmv_5` (3744.33): compressor recycle valve; `xmv_10` (1546.81): reactor cooling-water flow; `xmeas_9` (1372.12): reactor temperature; `xmeas_20` (594.25): compressor work; `xmeas_1` (365.08): A-feed flow | **Partly.** The A-feed flow is in the top five, but it ranks fifth. Losing the A feed throws the whole plant far outside normal operation: pressures rise, the recycle compressor and reactor cooling saturate. SPE ranks these large downstream effects above the root cause. Ridge, which compares each sample with its recent past, put `xmeas_1` and `xmv_3` (A feed) first. |
| 14: reactor cooling-water valve sticking | `xmeas_21` (49.87): reactor cooling-water outlet temperature; `xmeas_2` (12.94): D-feed flow; `xmv_1` (12.65): D-feed flow valve | **Mostly.** The top channel is in the cooling-water loop. The D-feed channels are false leads: on their own they barely move during this fault. They rank high because the residual direction of `xmeas_21` loads on `xmeas_2` and `xmv_1`, so an error in `xmeas_21` "smears" onto them. The channels that really move (`xmv_10`, `xmeas_9`) move in a normal pattern, so they show up in $T^2$ (97.8% detection) rather than in SPE. |

Fault 14 shows a known weakness of SPE contributions. The residual projection $I - PP^\top$ spreads a fault in one channel onto channels that are correlated with it, so channels that did not move can appear high in the ranking. As with Ridge, the ranking shows where the model is surprised, not which part of the plant failed.

## 7. Limits

### Ridge limitations

Ridge is a linear model and only uses the previous two samples. It may miss changes that need a longer history to detect. It can also miss a fault if the recent values still give accurate predictions. We saw no false alarms in the normal test runs, but this may change under other operating conditions. These simulations contain one fault at a time. They do not tell us how the model would perform with sensor failures, several faults at once, or changes in the plant over time.

### PCA limitations

PCA is linear and static. It assumes the normal channels are related by fixed linear correlations, and it ignores the order of samples. It cannot see faults that change how variables evolve over time without changing their spread (Faults 10, 16 and 20). It also loses faults that the control loops compensate for (Fault 5). The model has a single operating mode. A real plant changes throughput, product grade and catalyst activity, and each change would move the normal correlation structure, making PCA raise false alarms or need retraining. The choice of k (90% variance) and the 99th-percentile thresholds were set once and not tuned. A different k changes which faults land in $T^2$ and which in SPE. Finally, the false-alarm rates and thresholds come from simulated normal runs with stationary, well-behaved noise. Real sensors drift, freeze and drop out, so our false-alarm rates are likely optimistic for a real plant.

## 8. AI Use

Wenyin used GenAI to help refactor the Ridge code, check the assignment requirements and numerical results, and draft the Ridge report sections. The scores, evaluation results, and channel contributions were checked against the official reference calculations. The fault descriptions and channel names were checked against the sources linked above.

## Appendix: Member Contributions

- **Wenyin He — Andrew ID: `wenyinh`.** Wrote `src/ridge.py` for data preparation, standardization with training statistics, lagged features from the previous two samples within each run, Ridge training, residual scaling, and saving/loading `results/ridge_model.npz`. Corrected the evaluation runs, score format, and validation threshold in the plot. Wrote `src/evaluate_ridge.py` (validation thresholds, the three-point alarm rule, detection rates, delays and false-alarm rates) and `src/diagnose_ridge.py` (channel rankings over post-fault alarm samples). Produced the Ridge rows in `results/thresholds.csv`, `results/detection.csv` and `results/contributions.csv`. Added the Ridge Makefile targets and wrote the Ridge report sections (§2–§8).
- **Cosette Zhou — Andrew ID: `cosettez`.** Wrote the first version of `src/run_forecast_score.py`, using the saved Ridge model to predict all 52 channels, calculate squared normalized residuals for each channel, and sum them into anomaly scores for `results/scores_ridge.parquet`. Wrote the first version of `src/plot_ridge_score.py` and created the initial figure `results/ridge_fault1_plot.png`.
- **Vidhi Kothari — Andrew ID: `vidhikot`.** Wrote `src/pca.py`, the script version of the PCA pipeline: data verification against the official checksums, row counts and run ranges; the train/validation/test split; standardization with training statistics; choosing k for 90% variance; and $T^2$/SPE scoring to `results/scores_pca.parquet`. Added the `make pca` target. Wrote the PCA report sections (§2–§7) and §1, and checked the PCA analysis: the timing of Fault 5 alarms, the smearing of SPE contributions in Fault 14, and the literature comparison with Russell et al. (2000).
- **Meet Bhatt — Andrew ID: `mbbhatt`.** Built the original PCA implementation in `notebooks/PCA_v0.1.ipynb`. Wrote `src/evaluate_pca.py` (validation thresholds, the three-point alarm rule, detection rates, delays and false-alarm rates), `src/diagnose_pca.py` (SPE channel contributions over alarm samples) and `src/plot_pca.py` (the Fault 1 $T^2$/SPE figure). Produced the PCA rows in `results/thresholds.csv`, `results/detection.csv` and `results/contributions.csv`, and the figure `results/pca_fault1_plot.png`.
