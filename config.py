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

# ---- Questions that do NOT feed the ML model --------------------------------------
# They drive the stress profile, the mood screen and the personalised plan.
STUDY_HOURS = {"key": "study_hours", "q": "On an average day, how many hours do you spend studying on your own (classes excluded)?",
               "options": [("lt2", "Under 2 hours"), ("2-4", "2 to 4 hours"), ("5-7", "5 to 7 hours"), ("8+", "8 hours or more")]}

LIFESTYLE = [
    {"key": "sleep_hours", "q": "How many hours do you usually sleep at night?",
     "options": [("lt5", "Under 5 hours"), ("5-6", "5 to 6 hours"), ("7-8", "7 to 8 hours"), ("9+", "More than 8 hours")]},
    {"key": "activity", "q": "On how many days a week do you do at least 30 minutes of physical activity (walking, sports, gym, yoga)?",
     "options": [("0", "Almost never"), ("1-2", "1 to 2 days"), ("3-4", "3 to 4 days"), ("5+", "5 or more days")]},
    {"key": "screen", "q": "How much time do you spend on your phone or screens each day, not counting study?",
     "options": [("lt2", "Under 2 hours"), ("2-4", "2 to 4 hours"), ("4-6", "4 to 6 hours"), ("6+", "More than 6 hours")]},
    {"key": "meals", "q": "How regular are your meals?",
     "options": [("regular", "Regular most days"), ("sometimes", "I skip meals sometimes"), ("rarely", "I skip meals often")]},
]

# Standard two-question screeners (public domain): PHQ-2 (low mood) and GAD-2 (anxiety).
# Score each item 0-3; a total of 3 or more on either pair is the usual point for a fuller check.
MOOD_SCALE = [(0, "Not at all"), (1, "Several days"), (2, "More than half the days"), (3, "Nearly every day")]
MOOD_ITEMS = [
    {"key": "phq1", "q": "Little interest or pleasure in doing things", "group": "phq"},
    {"key": "phq2", "q": "Feeling down, depressed or hopeless", "group": "phq"},
    {"key": "gad1", "q": "Feeling nervous, anxious or on edge", "group": "gad"},
    {"key": "gad2", "q": "Not being able to stop or control worrying", "group": "gad"},
]
MOOD_CUTOFF = 3

# ---- Stress factors (self-rated 1-5; NOT fed to the ML model) -------------------
SCALE_LABELS = {1: "Not at all", 2: "A little", 3: "Moderately", 4: "A lot", 5: "Extremely"}

FACTORS = [
    {"key": "academic", "label": "Academic pressure", "q": "How much pressure do exams, grades and assignments put on you?"},
    {"key": "exam", "label": "Exam and result anxiety", "q": "How anxious do you feel before exams or when results are due?"},
    {"key": "workload", "label": "Workload and deadlines", "q": "How overwhelming is your overall workload and the number of deadlines?"},
    {"key": "financial", "label": "Financial pressure", "q": "How much do money worries (fees, living costs, family finances) weigh on you?"},
    {"key": "relationship", "label": "Relationship pressure", "q": "How much stress do your relationships (partner, friends, family) cause you?"},
    {"key": "family", "label": "Family expectations", "q": "How heavy do expectations from your family feel?"},
    {"key": "loneliness", "label": "Loneliness", "q": "How often do you feel alone or without people you can talk to?"},
    {"key": "support", "label": "Feeling unsupported at college", "q": "How unsupported do you feel by your teachers or college?"},
    {"key": "future", "label": "Worry about the future", "q": "How much do you worry about your career, placements or what comes after college?"},
    {"key": "compare", "label": "Comparing yourself with others", "q": "How much do you compare yourself with classmates or people online and feel worse?"},
    {"key": "focus", "label": "Trouble concentrating", "q": "How much trouble do you have concentrating on your studies?"},
    {"key": "motivation", "label": "Low motivation or energy", "q": "How often do you feel you have no energy or motivation to start things?"},
    {"key": "sleep", "label": "Sleep problems", "q": "How much trouble do you have sleeping (falling asleep, waking up, feeling rested)?"},
    {"key": "health", "label": "Health and tiredness", "q": "How much do health problems or constant tiredness affect you?"},
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
