# CodeAlpha_UnemploymentAnalysis

Task 2 of the **CodeAlpha Data Science Internship**: unemployment analysis with Python, focused on the impact of Covid-19 on India's labour market (May 2019 - October 2020).

## Data
| File | Coverage | Granularity |
|---|---|---|
| `data/Unemployment in India.csv` (A) | May 2019 - Jun 2020 | State x month x Rural/Urban |
| `data/Unemployment_Rate_upto_11_2020.csv` (B) | Jan 2020 - Oct 2020 | State x month, with zone and coordinates |

Variables: estimated unemployment rate (%), estimated employed persons, labour force participation rate (LFPR, %).

## Data cleaning
- Dataset A had **28 completely blank rows** (768 -> 740 real rows) and stray spaces in headers and values.
- Dataset B's `longitude` / `latitude` columns are **swapped** in the source (values 10-34 and 71-93); fixed. It also has two columns named `Region` (state and zone).
- Coverage is uneven: Chandigarh, Sikkim, J&K and Goa have missing months. Every total across states uses a **balanced panel** (A: 20 states, B: 25 states), otherwise lines move just because states enter or leave the file.
- National, zone and rural/urban rates are **weighted** (total unemployed / total labour force, with labour force reconstructed as employed / (1 - rate)), not simple averages of state rates. Outside the April-May peak a simple average overstates the rate by roughly 1-3 points (e.g. 10.0% vs 7.4% in July 2020) because small states count as much as Uttar Pradesh.
- 11 rural rows in A show exactly 0% unemployment (mainly Puducherry, Goa, Sikkim); kept but treated as suspect small-sample values.
- The two files agree closely where they overlap (Jan-Jun 2020, 158 state-months, correlation 0.97, median difference 0.0 pp) but 12 state-months differ by more than 5 pp (e.g. Telangana May 2020: 34.8% in A vs 14.7% in B). So the files are analysed separately and never spliced.

## Key findings
All figures are weighted, balanced-panel results.

**1. Covid-19 caused a sudden, very large but short-lived spike.**

| Phase (Dataset B) | Unemployment rate | LFPR |
|---|---|---|
| Pre-Covid (Jan-Feb 2020) | 7.4% | 42.8% |
| Lockdown starts (Mar 2020) | 8.7% | 41.9% |
| **Lockdown peak (Apr-May 2020)** | **22.9%** (24.0% in April) | 37.0% |
| Unlock begins (Jun 2020) | 10.2% | 40.4% |
| Rebound (Jul-Oct 2020) | 7.3% | 40.8% |

April 2020 was about **3.3x** the pre-Covid rate. By October the rate (6.9%) was back at or below the January-February level.

**2. The headline rate understates the damage.** Employed persons fell **32%** between January and April 2020 (403 m to 275 m in the 25 balanced states). By October, employment was still **3.2% below January**, and LFPR was 40.7% against 42.9% in January. People who left the labour force are not counted as unemployed, so the October rate looks fully recovered while fewer people were working.

**3. Urban areas were hit harder than rural areas** (Dataset A). Urban unemployment was higher before Covid (8.8% vs 7.0%; higher in every pre-Covid month, with October 2019 virtually tied), in April (25.8% vs 23.8%), May (26.2% vs 23.6%) and June (12.3% vs 10.8%).

**4. The shock was broad but very uneven across states.** Unemployment rose in **23 of 25 states** (paired Wilcoxon test, p ~ 1e-6, comparing each state's Apr-May average to its Jan-Feb average). The largest rises: Puducherry (+66 pp, from a tiny base), Jharkhand (+42), Tamil Nadu (+40), Bihar (+36), Karnataka (+22). By zone, April rates were highest in the East (32.5%) and South (28.6%), compared with the North (20.8%), West (16.5%) and Northeast (15.7%).

**5. Recovery was uneven too.** By October the median state was 0.7 pp *below* its baseline (not significantly different from it, p = 0.60), but Rajasthan (+11 pp), Goa (+6), Puducherry (+5), Haryana (+4), West Bengal (+4) and Uttarakhand (+4) were still more than 3 pp above. The North had the highest zone rate in October (11.3% vs 3.1% in the South).

**6. Seasonality cannot be established from this data.** Before Covid the national rate stayed in a narrow 7.1-8.1% band (mean 7.6%), and the series *zigzags* up and down in every one of the 9 month-to-month changes by about 0.7 pp. A pattern that regular looks more like a survey/measurement artefact than a real seasonal cycle. The data contains only 10 pre-Covid months and no repeated calendar month except May/June 2019 vs 2020, so no seasonal cycle can be tested; the only same-month comparison available is dominated by Covid.

## Insights for economic and social policy
These are inferences from the patterns above, not causal estimates.
1. **Support during shutdowns, not retraining afterwards.** Unemployment spiked within one month and fell back within three. That looks like a mobility/demand shock, so income support, wage subsidies and liquidity for firms during a shutdown fit better than long-term structural measures.
2. **Track participation and employment, not only the rate.** The rate recovered while employment and LFPR did not. Policy dashboards should report all three.
3. **Do not treat urban areas as safer.** Urban unemployment was higher in every phase, which supports extending employment guarantees and social protection to cities, not only rural areas.
4. **Target hotspots.** Puducherry, Jharkhand, Bihar and Tamil Nadu saw the largest jumps. A plausible (untested here) explanation is dependence on migrant and informal workers; this needs state-level data on occupations to confirm.
5. **Keep watching the laggards.** Rajasthan, Haryana, Goa and West Bengal had not returned to baseline by October.
6. **Improve the data.** The two files disagree materially for some state-months and several small states have gaps or implausible 0% rates; policy decisions need consistent monthly series.

## Limitations
- Only 14 (A) and 10 (B) months, so no full seasonal cycle and a single Covid wave.
- State-level estimates for small states (Puducherry, Goa, Sikkim, Chandigarh) are noisy.
- The balanced-panel totals exclude some states (A: Assam, Chandigarh, Goa, J&K, Meghalaya, Puducherry, Sikkim, Uttarakhand; B: J&K, Sikkim), so they approximate, not equal, national figures.
- Employment counts are reconstructed estimates, and nothing here proves the lockdown *caused* each change.

## Charts (`outputs/`)
| File | Shows |
|---|---|
| `01_national_trend.png` | National unemployment, May 2019 - Oct 2020, with the lockdown date |
| `02_jobs_and_participation.png` | Employment index and LFPR |
| `03_rural_vs_urban.png` | Rural vs urban rate |
| `04_state_impact_dumbbell.png` | Each state: baseline vs lockdown vs October |
| `05_state_month_heatmap.png` | State x month rate heatmap |
| `06_zones.png` | Regional trends |
| `07_april_map.png` | April 2020 rate by state location |
| `08_seasonality_check.png` | Pre-Covid pattern and same-month comparison |
| `09_unemployment_vs_participation.png` | Unemployment vs LFPR by state-month |

Tables and logs: `phase_summary.csv`, `state_impact.csv`, `summary.json`, `cleaning_and_findings_log.txt`.

## Run it
```bash
pip install -r requirements.txt
python unemployment_analysis.py
```

## Files
- `unemployment_analysis.py`: full pipeline (cleaning, analysis, charts)
- `data/`: the two source CSVs
- `outputs/`: charts, tables, logs
- `requirements.txt`: dependencies
