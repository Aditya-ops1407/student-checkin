"""Run:  python -m unittest discover -s tests -v   (from the project folder)"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app

GOOD = {"Gender": "Female", "Age": "20", "Living_style": "Family", "Marital_status": "Unmarried",
        "Relationship_status": "Single", "friend_circle": "Yes", "Family_background": "Middle class",
        "Current_semester": "4", "CGPA": "7.8"}
CALM = {**GOOD, **{f"f_{k}": "1" for k in
        ["academic", "financial", "relationship", "family", "workload", "sleep", "loneliness", "future"]}}
STRESSED = {**CALM, **{f"f_{k}": "5" for k in ["academic", "financial", "sleep", "future"]}}


class AppTests(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    def test_pages_load(self):
        for url in ["/", "/check-in", "/how-it-works", "/the-model", "/get-help", "/about", "/privacy", "/healthz"]:
            self.assertEqual(self.c.get(url).status_code, 200, url)

    def test_404_page(self):
        r = self.c.get("/nope")
        self.assertEqual(r.status_code, 404)
        self.assertIn("find that page", r.get_data(as_text=True))

    def test_calm_result(self):
        r = self.c.post("/check-in", data=CALM)
        t = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("What weighs on you most", t)
        self.assertIn("Overall pressure from your ratings: <strong>low</strong>", t)
        self.assertNotIn("might help to talk", t)   # model alone must not trigger the callout

    def test_stressed_shows_support(self):
        t = self.c.post("/check-in", data=STRESSED).get_data(as_text=True)
        self.assertIn("might help to talk", t)
        self.assertIn("Academic pressure", t)

    def test_ordinal(self):
        from app import ordinal
        self.assertEqual([ordinal(n) for n in (1, 2, 3, 11, 21, 91, 100)],
                         ["1st", "2nd", "3rd", "11th", "21st", "91st", "100th"])

    def test_validation(self):
        bad = {**CALM, "CGPA": "14", "Gender": "x", "f_sleep": "9"}
        r = self.c.post("/check-in", data=bad)
        self.assertEqual(r.status_code, 400)
        t = r.get_data(as_text=True)
        self.assertIn("Enter a value between 0 and 10", t)
        self.assertIn("Pick a rating from 1 to 5", t)

    def test_missing_fields(self):
        self.assertEqual(self.c.post("/check-in", data={}).status_code, 400)

    def test_extremes_do_not_crash(self):
        for cgpa in ("0", "10"):
            self.assertEqual(self.c.post("/check-in", data={**CALM, "CGPA": cgpa}).status_code, 200)

    def test_no_inline_styles(self):
        # The CSP blocks inline style attributes, so templates must not rely on them.
        for html in (self.c.get("/").get_data(as_text=True),
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
