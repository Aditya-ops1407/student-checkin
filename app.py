"""
Student Mental Health Risk Screening - Flask app.

Pages: home, check-in (3 steps) + result, how it works, model performance,
get help, about, privacy.  Nothing a user enters is stored or logged.

Local:       python app.py
Production:  gunicorn app:app   (see Procfile)
"""
import json
import os
from bisect import bisect_left

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

import config as C
from recommendations import build_plan

# Vercel serves public/ from its CDN at the site root; the same layout works locally.
app = Flask(__name__, static_folder="public", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024      # forms are tiny; reject anything bigger

model = joblib.load(C.MODEL_PATH)
with open(C.METRICS_PATH) as f:
    METRICS = json.load(f)
with open(C.OOF_PATH) as f:
    OOF = json.load(f)                              # sorted out-of-fold probabilities


# ----------------------------------------------------------------- helpers
def validate(form):
    """Return (clean, errors). errors maps field name -> message."""
    clean, errors = {}, {}

    for col, (lo, hi) in C.NUMERIC_RANGES.items():
        raw = form.get(col, "").strip()
        try:
            val = float(raw)
        except ValueError:
            errors[col] = "Enter a number."
            continue
        if not lo <= val <= hi:
            errors[col] = f"Enter a value between {lo:g} and {hi:g}."
            continue
        clean[col] = val

    for col, options in C.CATEGORICAL_FEATURES.items():
        val = form.get(col, "")
        if val not in options:
            errors[col] = "Choose one of the options."
        else:
            clean[col] = val

    # Lifestyle and study-hours choices (do not feed the ML model)
    life, study = {}, None
    for q in C.LIFESTYLE:
        val = form.get(q["key"], "")
        if val not in dict(q["options"]):
            errors[q["key"]] = "Choose one of the options."
        else:
            life[q["key"]] = val
    val = form.get(C.STUDY_HOURS["key"], "")
    if val not in dict(C.STUDY_HOURS["options"]):
        errors[C.STUDY_HOURS["key"]] = "Choose one of the options."
    else:
        study = val

    # Mood screen: PHQ-2 and GAD-2 items scored 0-3
    mood = {}
    for item in C.MOOD_ITEMS:
        raw = form.get(f"m_{item['key']}", "")
        if raw not in {"0", "1", "2", "3"}:
            errors[f"m_{item['key']}"] = "Choose how often this has bothered you."
        else:
            mood[item["key"]] = int(raw)

    # Stress-factor ratings 1-5
    ratings = {}
    for fac in C.FACTORS:
        raw = form.get(f"f_{fac['key']}", "")
        if raw not in {"1", "2", "3", "4", "5"}:
            errors[f"f_{fac['key']}"] = "Pick a rating from 1 to 5."
        else:
            ratings[fac["key"]] = int(raw)
    extras = {"ratings": ratings, "life": life, "study": study, "mood": mood}
    return clean, extras, errors


def level_for(score):
    if score >= 4:
        return "high"
    if score >= 3:
        return "moderate"
    return "low"


LEVEL_TEXT = {"low": "Low", "moderate": "Moderate", "high": "High"}


def factor_profile(ratings):
    """Rule-based summary of the self-rated stress factors (not a model)."""
    rows = []
    for fac in C.FACTORS:
        v = ratings[fac["key"]]
        rows.append({**fac, "value": v, "pct": v * 20, "level": level_for(v),
                     "level_text": LEVEL_TEXT[level_for(v)],
                     "answer": C.SCALE_LABELS[v]})
    mean = sum(r["value"] for r in rows) / len(rows)
    ordered = sorted(rows, key=lambda r: -r["value"])
    n_high = sum(r["value"] >= 4 for r in rows)
    overall = "high" if mean >= 3.5 or n_high >= 5 else "moderate" if mean >= 2.5 else "low"
    return {"rows": rows, "mean": round(mean, 1), "overall": overall,
            "overall_text": LEVEL_TEXT[overall], "n_high": n_high,
            "ordered": ordered, "shown": ordered[:6], "rest": ordered[6:]}


def model_indicator(clean):
    """Run the trained pipeline and express the result relative to the study data."""
    row = dict(clean)
    row["CGPA"] = min(max(row["CGPA"] * C.CGPA_TO_MODEL, C.CGPA_MODEL_RANGE[0]),
                      C.CGPA_MODEL_RANGE[1])
    df = pd.DataFrame([row])[C.NUMERIC_FEATURES + list(C.CATEGORICAL_FEATURES)]
    prob = float(model.predict_proba(df)[0][1])
    pct = round(100 * bisect_left(OOF, prob) / len(OOF))
    band = "higher" if pct >= 70 else "moderate" if pct >= 40 else "lower"
    return {"percentile": pct, "band": band,
            "band_text": {"higher": "Higher", "moderate": "Moderate", "lower": "Lower"}[band]}


@app.template_filter("ordinal")
def ordinal(n):
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@app.context_processor
def inject_globals():
    return {"helplines": C.HELPLINES, "nav": [
        ("home", "Home"), ("checkin", "Check-in"), ("how_it_works", "How it works"),
        ("model_page", "The model"), ("get_help", "Get help")]}


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; script-src 'self'; form-action 'self'; "
        "frame-ancestors 'none'; base-uri 'none'")
    if request.path.startswith("/check-in"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ----------------------------------------------------------------- pages
@app.route("/")
def home():
    return render_template("home.html", metrics=METRICS)


@app.route("/check-in", methods=["GET", "POST"])
def checkin():
    if request.method == "GET":
        return render_template("checkin.html", errors={}, values={},
                               config=C, ratings={}, mood={})
    clean, extras, errors = validate(request.form)
    if errors:
        return render_template("checkin.html", errors=errors, values=request.form,
                               config=C, ratings=extras["ratings"], mood=extras["mood"]), 400
    profile = factor_profile(extras["ratings"])
    plan = build_plan(extras["ratings"], extras["life"], extras["study"], extras["mood"],
                      {"friend_circle": clean["friend_circle"], "living_style": clean["Living_style"]})
    return render_template("result.html", profile=profile, plan=plan, ind=model_indicator(clean))


@app.route("/how-it-works")
def how_it_works():
    return render_template("how.html", metrics=METRICS)


@app.route("/the-model")
def model_page():
    return render_template("model.html", metrics=METRICS)


@app.route("/get-help")
def get_help():
    return render_template("help.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/healthz")
def healthz():
    return jsonify(status="ok")


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, title="We can't find that page",
                           msg="The link may be old or mistyped."), 404


@app.errorhandler(405)
def bad_method(_):
    return render_template("error.html", code=405, title="That page doesn't accept this action",
                           msg="Go back to the check-in and submit the form from there."), 405


@app.errorhandler(413)
def too_large(_):
    return render_template("error.html", code=413, title="That submission is too large",
                           msg="The check-in only needs short answers. Please try again."), 413


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500, title="Something went wrong on our side",
                           msg="Your answers were not saved. Please try the check-in again."), 500


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
