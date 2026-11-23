# Student Mental Health Risk Prediction Using Machine Learning

BCA project prototype (BBAU Satellite Centre, Amethi) - Aditya Jaiswal, AC-036/24.
Pipeline from the synopsis: preprocessing -> Logistic Regression / Decision Tree /
Random Forest comparison -> Flask questionnaire app.

> Academic screening prototype only. NOT a medical diagnosis.

## Run it

```bash
pip install -r requirements-train.txt   # website + training libraries
python prepare_data.py                  # data/raw CSV -> cleaned data/student_mental_health.csv
python train.py                         # CV, hold-out, leakage experiment, saves model + plots
python app.py                           # open http://127.0.0.1:5000
python -m unittest discover -s tests    # 11 tests
```

## The website

| Page | URL | What it does |
|---|---|---|
| Home | `/` | Explains the tool, example result, helplines |
| Check-in | `/check-in` | 32 questions in 5 steps: About you, Studies, Daily life, Mood (PHQ-2 + GAD-2), 14 pressure ratings (1-5). CGPA is entered on the 10-point scale |
| Result | `POST /check-in` | Summary, strengths, pressure profile, personalised plan, first-week list, ML indicator, support numbers |
| How it works / The model | `/how-it-works`, `/the-model` | Method, metrics tables, charts, leakage experiment |
| Get help, About, Privacy | `/get-help`, `/about`, `/privacy` | Support, project info, data handling |

Three separate parts on the result page:
1. **Pressure profile + mood screen**: rule-based summary of the user's own answers (not ML).
   PHQ-2 / GAD-2 totals of 3 or more (the usual cut-off) flag "worth a check", never a diagnosis.
2. **Personalised plan** (`recommendations.py`): a transparent rule-based recommendation engine
   (about 18 rules). Each suggestion has a priority score, a "why you see this" sentence built from
   the user's own answers, 2-4 steps and a time estimate; the top ones become a first-week list.
   It is not generative AI, nothing leaves the server, and it never diagnoses or advises treatment.
3. **ML indicator**: Random Forest on background answers only (the dataset has no pressure
   scores). Its probability is turned into a percentile of the training data's out-of-fold
   probabilities: Lower (<40th), Moderate (40-70th), Higher (>=70th).
The "talk to someone" box is driven by the user's own ratings; the model alone never triggers it.

## Deploy on Vercel

1. Push this folder to GitHub (the repo root must contain `app.py`, `vercel.json`, `.python-version`).
2. On vercel.com: Add New > Project > import the repo. Framework preset: Other/auto-detected Flask.
   Leave build/output settings empty. Click Deploy.
3. Static files live in `public/` (Vercel serves that folder from its CDN; Flask's own static folder is not used there).
4. `.python-version` pins Python 3.13; `scikit-learn` is pinned to 1.9.1 to match the saved model.
5. Check `/healthz`, then run one full check-in.

## Deploy elsewhere (Render, Railway, any Procfile host)

```bash
# Any host that runs a Procfile (Render, Railway, Heroku-style):
#   build:  pip install -r requirements.txt
#   start:  gunicorn app:app --workers 2 --timeout 60
```
- Commit `model/` (risk_model.joblib, metrics.json, oof_probs.json) and `public/img/`.
- `scikit-learn` is pinned: the saved model must load with the version that trained it.
  If you retrain on another machine, retrain with the same version or update the pin.
- Health check: `GET /healthz`. No database, no cookies, no stored answers.
- Security: strict Content-Security-Policy (no inline styles/scripts), 16 KB request cap,
  server-side validation of every field.
- The only external request is the Lexend font from Google Fonts (falls back to system fonts).

## Data decisions (for the report)

- **Target `Risk`** = 1 if `Depression == "Yes"` (27% positive).
- **Model inputs**: Age, Current semester, CGPA (10-point scale converted x0.4 to the 4.0 scale), Gender,
  Family background, Relationship status, Marital status, Living style, Friend circle. `Cost_without_study` was dropped (currency unknown, almost no effect).
- **Excluded `Health_condition` (data leakage)**: "Critical" -> 100% depressed, "Good" -> 100% not.
  With it ROC-AUC ~0.99; without it ~0.74. `train.py` reports both.
- **Excluded `Home_division`**: Bangladesh-specific, not generalisable. 3 duplicate rows removed.

## Limitations

- Self-reported survey from one country; correlations, not causes.
- Modest predictive power (ROC-AUC ~0.74): it is a screening aid, never a diagnosis.
- Baseline (always "Lower Risk") has higher accuracy (73%) but zero recall, so ROC-AUC and
  recall are used for selection instead of accuracy.
- **Cite the exact source of the dataset in your references.**
