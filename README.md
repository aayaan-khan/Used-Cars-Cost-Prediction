# Used Car Deal Score (India)

Is this used-car listing fairly priced? This project predicts a fair price for Indian used-car listings **together with an honest uncertainty range**, then turns the gap between the asking price and that range into a 0-100 *Deal Score*.

**Status: in progress.** Data audit, cleaning, EDA, validation design, model selection, conformal prediction intervals, error analysis and SHAP explanations are done. The final locked-test result, the Deal Score write-up and the demo app are still to come. The numbers below are development cross-validation, **not** test results.

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

## Does the Deal Score work? (notebook Section 12)

There are no labelled bargains, so the Deal Score was tested by changing the asking price of held-out listings by a known amount and counting how often it notices, using out-of-fold predictions only. The rules were fixed before the results were seen.

| Deal Score check (development, out-of-fold) | Unseen car models | Repeat listings |
|---|---|---|
| Untouched listings flagged either way (false-alarm upper bound) | 11.8% | 9.5% |
| Detected after a -30% price change | 39% | 54% |
| Detected after a -20% price change | 22% | 30% |
| Detected after a -10% price change | 12% | 13% |
| Detected after a +10% price change | 12% | 13% |
| Detected after a +20% price change | 21% | 27% |
| Detected after a +30% price change | 31% | 41% |
| Median price error, low-confidence vs other listings | 27.6% vs 16.8% | 20.2% vs 10.8% |

![Deal Score detection](reports/figures/deal_score_detection.png)

- **It detects large gaps, not small ones.** A 10% change is barely distinguishable from untouched listings; roughly half of a 50% markup is caught on unseen car models. Read "Overpriced" as "clearly outside what similar listings ask".
- **The false-alarm figure is an upper bound,** because some untouched listings may really be mispriced.
- **A low-confidence notice** (car model rare or unknown in the data, car 13+ years old, specs missing) marks listings where the price error is about 1.7-1.9 times larger. It does not lower the false-alarm rate in every regime, so it is shown as a warning next to the label rather than replacing it.
- Open point: the saved interval model is calibrated on held-out car models, so for well-known models it is wider than the repeat-listing figures above; this is settled before the demo app is built (decision D45).

## Where the model is wrong (notebook Section 9)
- Most large errors are whole car models that the price model mis-prices as a group when it has not seen them: budget models such as the Tata Nano (+130% median over-prediction), Mahindra Xylo (+78%) and Tata Indica (+42%), and the reverse for a model with strong resale such as the Toyota Innova (-40%).
- Error rises with age: 12.5% typical error up to 3 years old, 29.9% beyond 12 years. Listings with missing specs are worse (25.5% vs 17.3%).
- There is no systematic bias by predicted price. The costly failure for a Deal Score is a *false bargain*: on unseen models the fair price is at least 30% above the listing for 14.3% of listings (9.2% on repeat listings).
- Adding the car model's identity as a feature (Section 10, tested against a rule fixed in advance) lowers the repeat-listing error from 11.9% to 11.4% and cannot help models the system has not seen. It is not yet used by the saved models.

## What drives the price (notebook Section 11)

![SHAP summary](reports/figures/shap_summary.png)

- Age (39% of the model's price movement) and power (22%) dominate; fuel, brand and engine size follow with about 8% each. The ranking is the same on all five held-out folds of car models and agrees with permutation importance (rank correlation 0.94).
- Each listing gets a breakdown such as "age +30%, power +10%, diesel +6%", and the factors multiply exactly to the fair price (`src/explain.py`). The Deal Score then says how far the asking price sits from that fair price.
- These are descriptions of the model on 2020 asking prices, not causal claims. Rated mileage and seats add no usable signal on unseen car models.

## Repository layout
```
notebooks/Used_Cars_Cost_Prediction.ipynb   the full analysis, top to bottom
src/conformal.py                            interval classes (CQR variants)
src/deal_score.py                           deal_score(features, listed_price): fair price, range, label, confidence notice
src/deal_validation.py                      price-injection test and confidence-flag tools (notebook Section 12)
src/explain.py                              explain_listing(features): why the fair price is what it is (SHAP)
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
