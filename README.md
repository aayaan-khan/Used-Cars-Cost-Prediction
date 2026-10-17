# Used Car Deal Score (India)

Is this used-car listing fairly priced? This project predicts a fair price for Indian used-car listings **together with an honest uncertainty range**, then turns the gap between the asking price and that range into a 0-100 *Deal Score*.

**Status: in progress.** Data audit, cleaning, EDA, validation design, model selection and conformal prediction intervals are done. Error analysis, SHAP, the final locked-test result, the Deal Score write-up and the demo app are still to come. The numbers below are development cross-validation, **not** test results.

## Data
- Source: Kaggle "Vehicle dataset" by Nehal Birla (CarDekho), file `Car details v3.csv`, 8,128 rows. The file is not committed: download it into `data/raw/`.
- Licence: to be confirmed from the Kaggle page, see [`data/README.md`](data/README.md).
- Limits: prices are asking prices, not transaction prices; there is no city column; the data ends in 2020, so 2020 is treated as "now" and all prices are 2020-era INR.

## Approach
- **Cleaning:** 1,202 exact duplicate rows removed before any split; units parsed to numbers; physically impossible usage removed (4 rows); the 209 rows with missing specs kept, because they are older and cheaper, not random.
- **Target:** log(price), since depreciation works in percentages.
- **Validation:** 20% of the *car models* are locked away as the final test set. Development uses 5-fold cross-validation grouped by car model, so the model is judged on cars it has never seen. Every model is also scored on a random split ("typical repeat listing"). All preprocessing sits inside the pipeline.
- **Why the validation matters:** the same random forest scores R² 0.934 with duplicates and a random split, 0.906 once de-duplicated, and 0.810 when whole car models are held out.
- **Intervals:** conformalized quantile regression (nominal 90%), calibrated on held-out car models, with a locally adaptive variant selected by a rule fixed before looking at the results.
- Every decision, including the ones that did not pay off, is in [`docs/decision_log.md`](docs/decision_log.md).

## Results (development cross-validation, 5,530 listings)

| Model | Unseen car models: RMSE (log price) | Unseen car models: median error | Repeat listings: median error |
|---|---|---|---|
| Baseline: linear on age + log km | 0.528 | 28.7% | not measured |
| Ridge | 0.280 | 16.8% | 15.0% |
| **LightGBM (selected)** | 0.294 | 17.6% | 11.9% |

Ridge and the boosted models are statistically indistinguishable on unseen car models (fold standard deviation 0.02-0.04); boosting is clearly better on repeat listings. Repeat-listing figures keep near-duplicate relistings of the same car inside one fold (notebook Section 8); doing so changed no result by more than 0.22 points.

| 90% interval (adaptive CQR) | Coverage | Median width (% of price) |
|---|---|---|
| Unseen car models | 88.2% | 86% |
| Repeat listings | 90.5% | 65% |

Known weakness, disclosed rather than fixed: coverage on diesel listings is 85.6% against 91.3% for petrol on unseen models, and none of the three methods tried closed the gap.

## Repository layout
```
notebooks/Used_Cars_Cost_Prediction.ipynb   the full analysis, top to bottom
src/conformal.py                            interval classes (CQR variants)
src/deal_score.py                           deal_score(features, listed_price)
docs/decision_log.md                        every modelling decision, numbered
data/README.md                              data source, licence, caveats
models/                                     generated, git-ignored
reports/figures/                            figures saved by the notebook
```

## How to run
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # pinned to the versions this was verified with
# download "Car details v3.csv" into data/raw/
cd notebooks
jupyter nbconvert --to notebook --execute --inplace Used_Cars_Cost_Prediction.ipynb   # about 5-10 minutes, depending on your CPU
jupyter nbconvert --to html Used_Cars_Cost_Prediction.ipynb
```
Running the notebook also **regenerates the four files in `models/`**, which are not committed (pickled models only load under compatible library versions). Old `.joblib` files from before `src/conformal.py` existed will not load; delete them and re-run.

Try the Deal Score once the notebook has run (from the repo root):
```python
import math
from src.deal_score import deal_score

car = {"age": 6, "log_km": math.log1p(70000), "log_engine": math.log(1248), "log_power": math.log(81.8),
       "seats": 5, "mileage_val": 19.5, "fuel": "Diesel", "transmission": "Manual",
       "owner": "First Owner", "seller_type": "Individual", "brand": "Maruti"}
deal_score(car, listed_price=450000)   # fair price, 90% range, gap %, 0-100 score and label
```

## Limitations
- A Deal Score compares an asking price with what similar *listings* cost. It cannot see condition, accident history, city or negotiation, so it is not a verdict on the car.
- Two listings of the same car differ by a median of about 11.6%, which bounds how precise any fair-price estimate can be.
- 2020-era prices; no city.
