"""Run:  python -m unittest discover -s tests -v   (from the project folder)"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from app import app, ordinal
from recommendations import build_plan

FACTOR_KEYS = [f["key"] for f in C.FACTORS]
BACKGROUND = {"Gender": "Female", "Age": "20", "Living_style": "Family", "Marital_status": "Unmarried",
              "Relationship_status": "Single", "friend_circle": "Yes", "Family_background": "Middle class",
              "Current_semester": "4", "CGPA": "7.8"}
HABITS_GOOD = {"study_hours": "2-4", "sleep_hours": "7-8", "activity": "3-4", "screen": "lt2", "meals": "regular"}
MOOD_OK = {f"m_{m['key']}": "0" for m in C.MOOD_ITEMS}
CALM = {**BACKGROUND, **HABITS_GOOD, **MOOD_OK, **{f"f_{k}": "1" for k in FACTOR_KEYS}}
STRESSED = {**CALM, **{f"f_{k}": "5" for k in ["academic", "exam", "financial", "sleep", "future", "workload"]}}
POSITIVE_SCREEN = {**CALM, "m_phq1": "2", "m_phq2": "2"}


def plan_for(form):
    """Run the recommendation engine directly from a form-like dict."""
    f = {k: int(form[f"f_{k}"]) for k in FACTOR_KEYS}
    life = {q["key"]: form[q["key"]] for q in C.LIFESTYLE}
    mood = {m["key"]: int(form[f"m_{m['key']}"]) for m in C.MOOD_ITEMS}
    return build_plan(f, life, form["study_hours"], mood,
                      {"friend_circle": form["friend_circle"], "living_style": form["Living_style"]})


class AppTests(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    # ---- pages
    def test_pages_load(self):
        for url in ["/", "/check-in", "/how-it-works", "/the-model", "/get-help", "/about", "/privacy", "/healthz"]:
            self.assertEqual(self.c.get(url).status_code, 200, url)

    def test_question_count(self):
        self.assertEqual(len(C.NUMERIC_FEATURES) + len(C.CATEGORICAL_FEATURES) + 1 + len(C.LIFESTYLE)
                         + len(C.MOOD_ITEMS) + len(C.FACTORS), 32)

    def test_404_page(self):
        r = self.c.get("/nope")
        self.assertEqual(r.status_code, 404)
        self.assertIn("find that page", r.get_data(as_text=True))

    # ---- results
    def test_calm_result(self):
        r = self.c.post("/check-in", data=CALM)
        t = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("Your personalised plan", t)
        self.assertIn("Your first week", t)
        self.assertNotIn("might help to talk", t)       # the model alone must not trigger the callout

    def test_stressed_shows_support_and_plan(self):
        t = self.c.post("/check-in", data=STRESSED).get_data(as_text=True)
        self.assertIn("might help to talk", t)
        self.assertIn("Make study pressure manageable", t)
        self.assertIn("Why you see this", t)

    def test_positive_screen_result(self):
        t = self.c.post("/check-in", data=POSITIVE_SCREEN).get_data(as_text=True)
        self.assertIn("Talk to a counsellor or doctor", t)
        self.assertIn("Tele-MANAS", t)

    def test_extremes_do_not_crash(self):
        for cgpa in ("0", "10"):
            self.assertEqual(self.c.post("/check-in", data={**CALM, "CGPA": cgpa}).status_code, 200)

    # ---- validation
    def test_validation(self):
        bad = {**CALM, "CGPA": "14", "Gender": "x", "f_sleep": "9", "m_phq1": "7", "sleep_hours": "zz"}
        r = self.c.post("/check-in", data=bad)
        self.assertEqual(r.status_code, 400)
        t = r.get_data(as_text=True)
        for msg in ("Enter a value between 0 and 10", "Pick a rating from 1 to 5",
                    "Choose how often this has bothered you", "Choose one of the options"):
            self.assertIn(msg, t)

    def test_missing_fields(self):
        self.assertEqual(self.c.post("/check-in", data={}).status_code, 400)

    def test_every_field_is_required(self):
        for key in list(CALM):
            partial = {k: v for k, v in CALM.items() if k != key}
            self.assertEqual(self.c.post("/check-in", data=partial).status_code, 400, key)

    # ---- recommendation engine
    def test_engine_always_gives_three_to_six(self):
        for form in (CALM, STRESSED, POSITIVE_SCREEN):
            n = len(plan_for(form)["items"])
            self.assertTrue(3 <= n <= 6, n)

    def test_engine_sleep_rule(self):
        p = plan_for({**CALM, "sleep_hours": "lt5", "f_sleep": "5"})
        self.assertIn("sleep", [s["id"] for s in p["items"]])

    def test_engine_professional_first_on_positive_screen(self):
        p = plan_for({**STRESSED, "m_gad1": "2", "m_gad2": "1"})
        self.assertTrue(p["screen_positive"])
        self.assertEqual(p["items"][0]["id"], "professional")

    def test_engine_no_screen_flag_below_cutoff(self):
        p = plan_for({**CALM, "m_phq1": "1", "m_phq2": "1"})
        self.assertFalse(p["screen_positive"])

    def test_engine_each_suggestion_has_reason_and_steps(self):
        for form in (CALM, STRESSED, POSITIVE_SCREEN):
            for s in plan_for(form)["items"]:
                self.assertTrue(s["why"] and 2 <= len(s["steps"]) <= 4 and s["micro"], s["id"])

    def test_engine_week_has_seven_days(self):
        w = plan_for(STRESSED)["week"]
        self.assertEqual([x["day"] for x in w][-1], len(w))
        self.assertTrue(5 <= len(w) <= 7)

    def test_engine_no_diagnostic_language(self):
        banned = ("you have depression", "you are depressed", "diagnos", "medication", "disorder")
        for form in (CALM, STRESSED, POSITIVE_SCREEN):
            p = plan_for(form)
            text = " ".join(s["why"] + " ".join(s["steps"]) for s in p["items"]).lower()
            for b in banned:
                if b == "diagnos":
                    self.assertNotIn("you have been diagnos", text)
                else:
                    self.assertNotIn(b, text)

    # ---- misc
    def test_ordinal(self):
        self.assertEqual([ordinal(n) for n in (1, 2, 3, 11, 21, 91, 100)],
                         ["1st", "2nd", "3rd", "11th", "21st", "91st", "100th"])

    def test_no_inline_styles(self):
        # The CSP blocks inline style attributes, so templates must not rely on them.
        for html in (self.c.get("/").get_data(as_text=True), self.c.get("/check-in").get_data(as_text=True),
                     self.c.post("/check-in", data=STRESSED).get_data(as_text=True)):
            self.assertNotIn('style="', html)

    def test_headers_and_no_cookie(self):
        r = self.c.get("/")
        self.assertEqual(r.headers["X-Frame-Options"], "DENY")
        self.assertIn("default-src 'self'", r.headers["Content-Security-Policy"])
        self.assertNotIn("Set-Cookie", r.headers)

    def test_too_large_rejected(self):
        self.assertEqual(self.c.post("/check-in", data={"x": "a" * 20000}).status_code, 413)


if __name__ == "__main__":
    unittest.main()
