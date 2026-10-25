# ChaseIQ - Exam Guide: explain it, demo it, defend it

## 1. The 60-second pitch

> "ChaseIQ - *Inside Every Chase* - estimates, ball by ball, an IPL chasing team's probability of winning, and
> shows what decided the match. I cleaned 18 seasons of ball-by-ball data, engineered features such as required
> run rate, wickets in hand and an Elo team rating, and trained the **IPL Second-Innings Chase Model**: a blend of
> logistic regression and a regularised XGBoost. It reaches about 80% accuracy on 209 matches from 2023-25 that it
> never saw. The model is deliberately limited to IPL second innings - ChaseIQ refuses to apply it to ODI, T20I or
> Test cricket. A FastAPI service serves it, SQLite stores history, and a Streamlit dashboard adds a ball-by-ball
> **Simulator** and a **Live Cricket** page for India Men, Women and A teams built on
> free data sources, with the daily request quota managed."

## 2. Five-minute demo script

Start: `python run.py`.

1. **Live Cricket (60 s).** Show the three labels - LIVE, SCHEDULED, HISTORICAL. Open *Recent results* (real
   India men's and women's matches from Cricsheet), toggle **India Women** and **ODI** to show the filters. Point at the
   source line: *CricketData.org free tier, 100 requests/day; Cricsheet, ODC-BY*. If live data is unavailable, read the
   yellow notice: the app says so and shows setup steps instead of inventing scores. Mention: international matches
   show scores only - the chase model never predicts them.
2. **Simulator (60 s).** Hypothetical chase: bowl with **4 / 6 / W**; show *Why this probability?* and the pressure map.
3. **Replay + Model (45 s).** Press Play on a 2023-25 match (hold-out badge). On *Model*, point at the scope notice,
   the evaluation protocol and the calibration curve.
4. **API + tests (45 s).** `/docs` → `POST /predict` with `"competition": "ODI"` returns 422 - the guard in action.
   Finish with `python run.py --test` (83 tests pass).

## 3. Concepts in plain language

**Win probability.** For each ball of the 2nd innings: *given the score, wickets, balls left and target, how
often did chasing teams in similar situations win?* This is binary classification with a probabilistic output.

**Features.**
* *Required run rate* = runs needed ÷ overs left; *current run rate* = runs ÷ overs bowled. Their gap is the
  single strongest signal.
* *Wickets in hand* = resources. 8 runs/over with 2 wickets left is far harder than with 8 left.
* *Target vs ground average* - 170 is a big total at Chepauk and a small one at Chinnaswamy.
* *Elo gap* - a chess-style rating updated after every match: win → gain points, more if you beat a stronger side.
* *Form / head-to-head / ground chase bias* - smoothed win ratios computed from **earlier** matches only.

**Logistic regression.** Weighted sum of features → squashed to 0-1 by the sigmoid. Simple and interpretable.

**XGBoost.** Many small decision trees, each correcting the previous ones. Captures non-linear effects
(e.g. pressure rising sharply in the last 3 overs). We use shallow trees, strong regularisation and
**monotone constraints**: runs needed ↑ can never raise win probability; wickets in hand ↑ never lowers it.

**Blend in log-odds space.** `logit(p) = 0.5·logit_LR + 0.5·logit_XGB`. Both parts are additive in log-odds, so
the blend is too: `base rate + contribution of each feature = prediction`. Tree contributions come from
XGBoost's built-in SHAP values; logistic contributions are coefficient × standardised value. That is the
"Why this probability?" chart, and a test proves the parts sum exactly to the prediction.

**Metrics.**
* *Accuracy* - share of correct win/lose calls (depends on the 50% threshold).
* *ROC-AUC* - can the model rank winning situations above losing ones? 0.5 = random, 1 = perfect.
* *Brier score* - mean squared error of the **probabilities**; lower is better.
* *Log-loss* - heavily punishes confident wrong predictions; used to choose the model.
* *Calibration* - do predicted 70% events happen 70% of the time?

**Chronological split & leakage.** A random split would put balls from the same match in train and test, and
let Elo/form "see the future". We train on past seasons and test on later ones - exactly the deployment situation.

## 4. Database design

```
teams ─┬─< matches >── venues
       │       │
       │       └─< chase_states (match_id, balls, runs, wickets, win_prob)
       └─< head_to_head
prediction_log   live_snapshots      (written by the API at run time)
```

* **Reference/history tables** (`teams`, `venues`, `matches`, `head_to_head`) are built once by the pipeline.
* **`chase_states`** stores the model's probability for every historical ball (125k rows) - this is what makes
  replay instant and lets analytics be pure SQL (e.g. `MIN(win_prob)` per match = "greatest escape").
* **`teams.elo / form`** hold the *current* ratings, so live predictions use real team strength.
* **`prediction_log` / `live_snapshots`** are operational tables (CRUD from the API).
* Foreign keys, indexes on `season` and `venue_id`, composite primary key on `(match_id, balls)`.

*Why SQLite?* Zero-install, one portable file, full SQL - perfect for a demo laptop and a free cloud host.

## 5. Likely viva questions

**Why did logistic regression nearly match XGBoost?** There are ~110 rows per match but only one outcome, so
rows are highly correlated and deep trees memorise matches. Our evidence: tuned/regularised trees beat the
original XGBoost, and the simple blend wins on validation log-loss.

**How do you know it isn't overfitting?** The test seasons (2023-25) were not used for fitting or model
selection (selection used 2020-22). Test results are close to validation (accuracy 79.8% vs 80.1%, AUC 0.878 vs
0.885, log-loss 0.440 vs 0.423), so there is no sign of a large gap. The test numbers are slightly worse, as
expected for unseen later seasons, and the guide reports that rather than hiding it.

**Isn't 80% accuracy low?** Cricket is high-variance. Accuracy is 69% before a ball is bowled, 87% with 5
overs left. Judge probabilities with Brier/log-loss/calibration, not accuracy alone.

**Why add monotone constraints?** Domain knowledge as a regulariser. It guarantees sensible behaviour (the
simulator can never show win % rising when more runs are needed).

**What does the toss do?** Almost nothing (toss winners win 51.6% of the time) - the model learns that.

**How is the target defined?** `target = first-innings total + 1`. Training and serving share one feature
function (`src/features.py`) so they cannot disagree.

**What did you remove from the data and why?** Super overs, ties, no-results and rain-shortened chases
(label is undefined or not comparable); terminal states (outcome already known).

**What is Elo?** Rating that rises when you beat a stronger opponent. K = 24; 25% pulled back to the mean each
season because squads change at auctions.

**Why is early-over run rate "shrunk"?** After one ball "4 runs in 0.17 overs" = 24 runs/over, which is
noise. We blend in a prior of 8 rpo worth one over; a first-ball boundary used to swing the probability
by ~15 points and now the average first-ball swing is ~3 points.

**How does the live feature work?** A background task asks CricketData.org for the current matches (one request), normalises them (India Men / Women / A, format), caches them in SQLite and re-reads at an adaptive interval that protects the free 100-requests/day budget. If a live IPL match is in its second innings, the same response is converted (15.3 overs → 93 balls) and scored by the IPL Second-Innings Chase Model; every other match is shown with scores only. The dashboard re-reads the server's cache every minute without spending provider requests.

**What if the venue is unknown?** It is matched fuzzily to a known ground; otherwise neutral league-average
ground statistics are used and the response says `venue_known: false`.

**How is it tested?** 81 pytest tests: monotonic behaviour, exact additivity of explanations, no train/serve skew,
terminal states, every API route (bad input → 422), the IPL-only guards, India Men/Women/A
classification, provider parsing, the daily request budget, and 'never invent data' rules. Tests run on a temporary
copy of the database and never call a real provider. An optional script (`tests/e2e_browser.py`) drives the real UI
in Chrome (26 checks).

## 6. Honest limitations (say these before they ask)

* Scoreboard-only: no players, pitch, dew, powerplay/death-over bowler types.
* Only second-innings chases (first-innings prediction would need a different model).
* Elo is team-level; a star player's absence is invisible.
* Replay probabilities for 2008-2022 matches are *in-sample* (the badge says so); 2023-25 are out-of-sample.
* Live and fixture data depend on CricketData.org's free tier (100 requests/day, a few minutes behind) and on a valid key.

## 7. What changed from version 1

| Area | Before | Now |
|---|---|---|
| Data | numeric team ids, 59 venue spellings, ties/rain included, target off by one | franchises mapped, 37 canonical venues, clean labels, one shared feature function |
| Model | XGBoost only, venue label-encoding, accuracy-based reporting | 5 candidates, chronological selection, monotone constraints, calibration + stage metrics |
| Explainability | none | exact per-factor contributions, what-if, pressure map |
| Storage | pickles/CSVs only | SQLite with 10 tables: history, per-match context, provider cache, quota counter |
| Backend | 5 endpoints in one file | 20+ endpoints in 5 routers, validation, logging, background poller |
| Frontend | 2 tabs, broken buttons | 5 views: Live Cricket, Simulator, Replay, Analytics, Model |
| Quality | no tests | 83 tests + CI workflow + browser checks |

## 8. ChaseIQ additions - likely questions

**Why does ChaseIQ not predict ODIs or T20Is?** The model was trained only on IPL second innings. Cricket formats
and sides differ (pitch, overs, team strength), so applying it elsewhere would give confident but unfounded
numbers. The API refuses it (`competition` must be IPL; international teams → 422) and the UI shows scores only.

**Where does live data come from, and what are the limits?** CricketData.org - free forever, **100 requests/day**
(checked on its pricing page), data a few minutes behind. ChaseIQ spends that budget deliberately: a 10-minute poll
while something is live, 30 minutes otherwise, a persisted counter in SQLite and a 10-request reserve. Results come
from Cricsheet (free, ODC-BY, attribution shown, delayed by days). ESPN, Cricinfo and TheSportsDB were tried and
rejected (empty / blocked / key disabled).

**What if the provider is down or has no key?** Each source reports `ok / no_key / invalid_key / quota / error`.
The page shows a notice and the last cached data (time-stamped); it never fills the gap. The project's original key
was actually rejected ("Invalid API Key") - the app detects that once and does not retry for hours.

**Is the what-if exact?** No - it applies each ball as chosen (4 = four runs) and cannot model extras or run-outs on
non-legal balls, and boundaries in the historical timeline are inferred from runs added on a ball (4 or more).

**What can you not verify?** Women's and India A fixtures from CricketData.org (needs a valid key to test); India A
results appear only if Cricsheet carries the match. Both limits are stated in the app and README.

**Why remove the Prediction Log page?** It was developer plumbing, not product value. The table and `/history`
endpoints stay in the backend (the API still logs `POST /predict`), but the UI no longer exposes them.
