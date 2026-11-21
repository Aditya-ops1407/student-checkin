"""Shared configuration (Student Depression Dataset + stress-factor questionnaire)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   # absolute paths: works under gunicorn and Vercel

TARGET = "Risk"

# ---- Model inputs (columns that exist in the training data) --------------------
NUMERIC_FEATURES = ["Age", "Current_semester", "CGPA"]

CATEGORICAL_FEATURES = {
    "Gender": ["Female", "Male"],
    "Family_background": ["Lower Middle class", "Middle class", "Rich class"],
    "Relationship_status": ["Single", "In a Relationship", "Complicated"],
    "Marital_status": ["Unmarried", "Married", "Separated"],
    "Living_style": ["Family", "Bachelor"],
    "friend_circle": ["Yes", "No"],
}

# Suspected label-proxy column; used ONLY for the leakage experiment in train.py.
LEAKY = "Health_condition"

# ---- Web-form details ---------------------------------------------------------
# The dataset stores CGPA on a 4.0 scale; the form asks on the Indian 10-point scale
# and converts linearly (x 0.4). Values are clipped to the range seen in training.
CGPA_SCALE = 10
CGPA_TO_MODEL = 4.0 / CGPA_SCALE
CGPA_MODEL_RANGE = (2.25, 4.0)

NUMERIC_RANGES = {            # form (min, max)
    "Age": (16, 40),
    "Current_semester": (1, 12),
    "CGPA": (0.0, 10.0),
}

# Option labels shown to the user (value stored for the model -> friendly text)
OPTION_LABELS = {
    "Family_background": {"Lower Middle class": "Lower middle income",
                          "Middle class": "Middle income",
                          "Rich class": "Higher income"},
    "Marital_status": {"Unmarried": "Unmarried", "Married": "Married", "Separated": "Separated"},
    "Living_style": {"Family": "With family", "Bachelor": "On my own / with friends"},
    "friend_circle": {"Yes": "Yes", "No": "Not really"},
}

# ---- Stress factors (self-rated 1-5; NOT fed to the ML model) -------------------
SCALE_LABELS = {1: "Not at all", 2: "A little", 3: "Moderately", 4: "A lot", 5: "Extremely"}

FACTORS = [
    {"key": "academic", "label": "Academic pressure",
     "q": "How much pressure do exams, grades and assignments put on you?",
     "tip": "Break big tasks into small daily targets, and ask a teacher or senior about what matters most for the exam."},
    {"key": "financial", "label": "Financial pressure",
     "q": "How much do money worries (fees, living costs, family finances) weigh on you?",
     "tip": "Check scholarship, fee-instalment and part-time options with your college office; talking about money early usually helps."},
    {"key": "relationship", "label": "Relationship pressure",
     "q": "How much stress do your relationships (partner, friends, family) cause you?",
     "tip": "Pick one conversation you have been avoiding and plan a calm time to have it, or talk it through with someone neutral first."},
    {"key": "family", "label": "Family expectations",
     "q": "How heavy do expectations from your family feel?",
     "tip": "Share your real progress and limits with one trusted family member; expectations often ease once they are spoken."},
    {"key": "workload", "label": "Workload and deadlines",
     "q": "How overwhelming is your overall workload and the number of deadlines?",
     "tip": "Write every deadline in one place, rank them, and protect one short break each day."},
    {"key": "sleep", "label": "Sleep problems",
     "q": "How much trouble do you have sleeping (falling asleep, waking up, feeling rested)?",
     "tip": "Keep a fixed wake-up time, stop screens 30 minutes before bed, and avoid late-night caffeine."},
    {"key": "loneliness", "label": "Loneliness",
     "q": "How often do you feel alone or without people you can talk to?",
     "tip": "Join one low-pressure group (club, study circle, sports) and message one person today, even briefly."},
    {"key": "future", "label": "Worry about the future",
     "q": "How much do you worry about your career, placements or what comes after college?",
     "tip": "Turn the worry into one concrete next step this week (a skill, a project, a mentor chat)."},
]

HELPLINES = [
    {"name": "Tele-MANAS", "number": "14416", "note": "Free, 24x7 national mental-health support in many languages (also 1800-891-4416)."},
    {"name": "KIRAN", "number": "1800-599-0019", "note": "Free 24x7 mental-health rehabilitation helpline, Government of India."},
    {"name": "Emergency", "number": "112", "note": "If you or someone else is in immediate danger."},
]

MODEL_PATH = str(BASE_DIR / "model" / "risk_model.joblib")
METRICS_PATH = str(BASE_DIR / "model" / "metrics.json")
OOF_PATH = str(BASE_DIR / "model" / "oof_probs.json")
DATA_PATH = str(BASE_DIR / "data" / "student_mental_health.csv")
