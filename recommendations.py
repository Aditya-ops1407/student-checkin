"""
recommendations.py
------------------
Built-in recommendation engine for the "personalised plan".

It is a transparent, rule-based expert system (a classical branch of AI), NOT a
generative model. Every suggestion is triggered by explicit rules over the student's
own answers, carries a plain-language reason ("why you are seeing this"), and has a
priority score. This keeps it private (nothing leaves the server), free, testable and
explainable in a viva.

Safety rules:
  * No diagnosis, no medication advice, no treatment claims.
  * A positive mood screen (PHQ-2 / GAD-2 total >= 3) always puts "talk to a
    professional" first.
  * The ML model's band never creates a suggestion on its own.
"""
import config as C

LABELS = {
    "sleep_hours": dict(C.LIFESTYLE[0]["options"]),
    "activity": dict(C.LIFESTYLE[1]["options"]),
    "screen": dict(C.LIFESTYLE[2]["options"]),
    "meals": dict(C.LIFESTYLE[3]["options"]),
    "study_hours": dict(C.STUDY_HOURS["options"]),
}


def sev(rating):
    """Map a 1-5 self-rating to priority points (ratings of 1-2 trigger nothing)."""
    return {1: 0, 2: 0, 3: 5, 4: 8, 5: 10}[rating]


def _rated(f, *keys):
    """'You rated X 5/5 and Y 4/5' for the keys the student rated 3 or more."""
    names = {x["key"]: x["label"].lower() for x in C.FACTORS}
    parts = [f"{names[k]} {f[k]}/5" for k in keys if f[k] >= 3]
    return "You rated " + ", ".join(parts) + "." if parts else ""


def _suggestion(sid, title, score, why, steps, micro, time):
    return {"id": sid, "title": title, "score": round(min(score, 10), 1), "why": why,
            "steps": steps, "micro": micro, "time": time,
            "priority": "high" if score >= 8 else "medium" if score >= 5 else "low"}


# ---------------------------------------------------------------------------------
# Rules. Each returns a suggestion dict (with a score) or None.
# ctx keys: f (factor ratings 1-5), life (lifestyle choices), study (study_hours choice),
#           phq, gad (totals 0-6), mood (item scores), bg (background answers)
# ---------------------------------------------------------------------------------
def r_professional(x):
    phq, gad = x["phq"], x["gad"]
    if phq >= C.MOOD_CUTOFF or gad >= C.MOOD_CUTOFF:
        return _suggestion(
            "professional", "Talk to a counsellor or doctor", 10,
            f"Your four mood answers add up to {phq} out of 6 on the low-mood questions and {gad} out of 6 on "
            f"the anxiety questions. A total of {C.MOOD_CUTOFF} or more on either pair is the usual point at "
            "which a fuller check with a professional is suggested. This is a screen, not a diagnosis.",
            ["Book a visit with your college counsellor or a doctor, or call Tele-MANAS on 14416 (free, 24x7).",
             "Tell them how long you have felt this way and what has been hardest.",
             "Let one person you trust know that you are doing this, so you are not carrying it alone."],
            ["Tell one trusted person how you have been feeling.",
             "Find the contact details of your college counsellor or a doctor."],
            "One call or visit")
    if phq >= 2 or gad >= 2:
        return _suggestion(
            "mood_watch", "Keep an eye on your mood", 6,
            f"Your mood answers add up to {phq}/6 (low mood) and {gad}/6 (anxiety). That is below the usual "
            "cut-off, but worth watching.",
            ["Each evening, note your mood from 1 to 10 in one line.",
             "Write down what helped and what made it worse.",
             "If low mood or worry lasts two weeks or grows stronger, speak to your counsellor or a doctor."],
            ["Start a one-line mood note tonight."], "2 min a day")
    return None


def r_low_interest(x):
    if x["mood"]["phq1"] >= 2:
        return _suggestion(
            "interest", "Schedule something you used to enjoy", 5 + x["mood"]["phq1"] - 2,
            "You said you have had little interest or pleasure in things on more than half the days.",
            ["Pick one small activity you used to like (music, a game, cooking, a walk, calling a friend).",
             "Put it in your timetable like a class, for 15 to 20 minutes.",
             "Do it even if you do not feel like it; mood often follows action, not the other way round."],
            ["Choose one enjoyable 15-minute activity and book it for today.",
             "Do it for the full 15 minutes, then note how you felt."], "15 min, 3 times a week")


def r_worry(x):
    if x["mood"]["gad2"] >= 2 or x["mood"]["gad1"] >= 2:
        return _suggestion(
            "worry", "Calm racing worry", 5 + max(x["mood"]["gad1"], x["mood"]["gad2"]) - 2,
            "You said you have felt nervous or unable to stop worrying on more than half the days.",
            ["Slow breathing: breathe in for 4 counts and out for 6, for 2 minutes.",
             "Keep a 15-minute 'worry window' each day: write each worry and one possible action, then stop.",
             "Grounding: name 5 things you can see, 4 you can touch and 3 you can hear."],
            ["Try 2 minutes of slow breathing (in for 4, out for 6).",
             "Write your worries down and give each one a next step."], "2 to 15 min a day")


def r_sleep(x):
    s = {"lt5": 6, "5-6": 4, "7-8": 0, "9+": 1}[x["life"]["sleep_hours"]] + sev(x["f"]["sleep"]) * 0.5
    if s < 4:
        return None
    return _suggestion(
        "sleep", "Protect your sleep", s,
        f"You usually sleep {LABELS['sleep_hours'][x['life']['sleep_hours']].lower()} and rated sleep problems "
        f"{x['f']['sleep']}/5.",
        ["Wake up at the same time every day, within an hour, even on weekends.",
         "Keep screens off for the last 30 minutes before bed, and the phone away from your pillow.",
         "Avoid tea, coffee and energy drinks after about 4 pm.",
         "If you are awake after about 20 minutes, get up, do something quiet in dim light, then try again."],
        ["Set the same wake-up alarm for tomorrow.", "Tonight, switch screens off 30 minutes before bed."],
        "10 min a day")


def r_movement(x):
    s = {"0": 6, "1-2": 4, "3-4": 1, "5+": 0}[x["life"]["activity"]] + (1 if x["phq"] >= 2 else 0)
    if s < 4:
        return None
    return _suggestion(
        "movement", "Move a little on most days", s,
        f"You are active {LABELS['activity'][x['life']['activity']].lower()} a week.",
        ["Start with a 10-minute walk after a meal and add 5 minutes each week.",
         "The WHO suggests about 150 minutes of moderate activity a week for adults; build towards it slowly.",
         "Choose something you enjoy (cricket, badminton, cycling, dancing, yoga) and invite a friend."],
        ["Take a 10-minute walk today.", "Try a 10-minute stretch or home workout."], "10 to 30 min a day")


def r_screen(x):
    s = {"lt2": 0, "2-4": 1, "4-6": 3, "6+": 6}[x["life"]["screen"]]
    if x["f"]["compare"] >= 3:
        s += 2
    if s < 4:
        return None
    return _suggestion(
        "screen", "Reset your screen habits", s,
        f"You spend {LABELS['screen'][x['life']['screen']].lower()} a day on your phone or screens outside study"
        + (f", and rated comparing yourself with others {x['f']['compare']}/5." if x["f"]["compare"] >= 3 else "."),
        ["Turn off notifications for apps that are not essential.",
         "Choose one screen-free time every day, for example during meals.",
         "Mute or unfollow accounts that leave you feeling worse about yourself."],
        ["Mute notifications for three apps today.", "Have one meal without your phone."], "5 min to set up")


def r_meals(x):
    s = {"regular": 0, "sometimes": 3, "rarely": 6}[x["life"]["meals"]]
    if s < 4:
        return None
    return _suggestion(
        "meals", "Make meals more regular", s,
        f"You said you {'often' if x['life']['meals'] == 'rarely' else 'sometimes'} skip meals.",
        ["Anchor your day with one proper meal at a fixed time.",
         "Keep simple snacks in your room (fruit, nuts, roasted chana).",
         "Drink water through the day; thirst and hunger can feel like tiredness or irritability."],
        ["Plan tomorrow's first meal and set a reminder."], "Ongoing")


def r_study_pressure(x):
    a = max(x["f"]["academic"], x["f"]["exam"], x["f"]["workload"])
    s = sev(a) + (1 if x["study"] == "8+" else 0)
    if s < 5:
        return None
    return _suggestion(
        "study", "Make study pressure manageable", s,
        _rated(x["f"], "academic", "exam", "workload"),
        ["Write every deadline in one place and rank them by date and marks.",
         "Study in 25-minute focus blocks with 5-minute breaks; stop after two or three blocks for something unrelated.",
         "Practise with past papers instead of only re-reading; familiarity reduces exam fear.",
         "Ask one teacher or senior what matters most for the exam."],
        ["List every deadline for the next two weeks.", "Do one 25-minute focus block."], "30 min to plan")


def r_study_balance(x):
    if x["study"] != "8+":
        return None
    return _suggestion(
        "balance", "Balance long study hours with rest", 5 + (2 if x["f"]["sleep"] >= 4 else 0),
        "You study 8 hours or more on an average day.",
        ["Schedule rest as you schedule study: a break after every 90 minutes and one free evening a week.",
         "Rested hours usually achieve more than long, tired ones, especially when sleep is short.",
         "End the day with a fixed stopping time."],
        ["Choose a stopping time for tonight and keep to it."], "Ongoing")


def r_financial(x):
    s = sev(x["f"]["financial"])
    if s < 5:
        return None
    return _suggestion(
        "money", "Face money worries early", s, _rated(x["f"], "financial"),
        ["Write down your fixed monthly costs and what comes in; a number feels smaller than a fear.",
         "Ask your college office about fee instalments, scholarships and waivers.",
         "Look at small earning options that fit your timetable.",
         "Tell one trusted person; money stress usually shrinks when it is shared."],
        ["Write your monthly costs on one page.", "Message the college fee or scholarship office with one question."],
        "30 min")


def r_relationship(x):
    s = sev(x["f"]["relationship"])
    if s < 5:
        return None
    return _suggestion(
        "relationship", "Ease relationship strain", s, _rated(x["f"], "relationship"),
        ["Name the one issue that bothers you most, in a single sentence.",
         "Choose a calm moment and start with how you feel rather than what they did wrong.",
         "If conversations keep escalating, talk it through first with a neutral person such as a counsellor or senior."],
        ["Write your one-sentence issue.", "Decide when and where you will talk."], "20 min")


def r_family(x):
    s = sev(x["f"]["family"])
    if s < 5:
        return None
    return _suggestion(
        "family", "Talk about family expectations", s, _rated(x["f"], "family"),
        ["Share your real progress and limits with one family member you trust.",
         "Agree on one realistic target instead of an open-ended expectation.",
         "Say what kind of support would help most (time, encouragement, fewer comparisons)."],
        ["Pick the family member you could talk to and a time to do it."], "One conversation")


def r_support_circle(x):
    s = max(sev(x["f"]["loneliness"]), sev(x["f"]["support"]))
    if x["bg"]["friend_circle"] == "No":
        s += 3
    if x["bg"]["living_style"] == "Bachelor" and x["f"]["loneliness"] >= 3:
        s += 1
    if s < 5:
        return None
    bits = [_rated(x["f"], "loneliness", "support")]
    if x["bg"]["friend_circle"] == "No":
        bits.append("You said you do not have a close circle of friends.")
    return _suggestion(
        "connect", "Build your support circle", s, " ".join(b for b in bits if b),
        ["Message one person today; a short 'how are you?' counts.",
         "Join one regular group (club, study circle, sports, hostel activity).",
         "If there is no one close, your mentor or college counsellor is there for exactly this."],
        ["Send one 'how are you?' message.", "Find one club or study group you could join."], "5 min to start")


def r_college_support(x):
    s = sev(x["f"]["support"])
    if s < 5:
        return None
    return _suggestion(
        "college", "Use the support your college offers", s, _rated(x["f"], "support"),
        ["Find out who your mentor or counsellor is and send a short message.",
         "Ask for what you need in one sentence (an extension, a clarification, a meeting).",
         "Ask whether there is a student welfare cell."],
        ["Find your mentor's or counsellor's contact."], "10 min")


def r_future(x):
    s = sev(x["f"]["future"])
    if s < 5:
        return None
    return _suggestion(
        "future", "Turn future worry into next steps", s, _rated(x["f"], "future"),
        ["Write the worry as one specific question, such as 'Will I get placed?'.",
         "List one small step you control and do it this week (a skill, a project, a chat with a mentor).",
         "Keep 'what if' thinking to a 15-minute slot, then switch to a task."],
        ["Write the worry as a question and one action for it."], "20 min a week")


def r_compare(x):
    s = sev(x["f"]["compare"])
    if s < 5:
        return None
    return _suggestion(
        "compare", "Stop measuring yourself against others", s, _rated(x["f"], "compare"),
        ["When you compare, ask what part of this is a highlight reel.",
         "Measure yourself against your own last month, not someone else's best day.",
         "Write down three things you did well this week."],
        ["Write three things you did well this week."], "5 min a day")


def r_focus(x):
    a = max(x["f"]["focus"], x["f"]["motivation"])
    s = sev(a) + (1 if x["mood"]["phq1"] >= 2 else 0)
    if s < 5:
        return None
    return _suggestion(
        "focus", "Rebuild focus and motivation", s, _rated(x["f"], "focus", "motivation"),
        ["Shrink the task: promise yourself only 5 minutes, and allow yourself to stop after that.",
         "Take one distraction away (phone in another room, tabs closed).",
         "Do your hardest task when your energy is highest.",
         "Give yourself a small reward after you finish."],
        ["Do a 5-minute start on the task you have been avoiding."], "5 to 25 min")


def r_health(x):
    s = sev(x["f"]["health"])
    if s < 5:
        return None
    return _suggestion(
        "health", "Look after your body", s, _rated(x["f"], "health"),
        ["Note when the problems happen and how long they last.",
         "If tiredness or symptoms last more than two weeks, see a doctor; many causes are easy to treat.",
         "Morning daylight, water and regular meals help more than most people expect."],
        ["Write down what you have noticed and when."], "10 min")


FALLBACKS = [
    _suggestion("routine", "Keep a simple daily routine", 3,
                "A steady routine is one of the easiest ways to protect your mood.",
                ["Fix a wake-up time, a main study slot and a stopping time.",
                 "Include one thing each day that is only for you.",
                 "Review your week every Sunday for five minutes."],
                ["Write tomorrow's routine in three lines."], "10 min a day"),
    _suggestion("kind", "Do one kind thing for yourself", 3,
                "Small acts of self-care help when pressure is around.",
                ["Choose something small: a favourite song, a walk, a call to a friend.",
                 "Speak to yourself as you would to a friend in your situation."],
                ["Do one small kind thing for yourself today."], "10 min"),
    _suggestion("stay", "Stay connected", 3,
                "People who keep in regular touch with others cope better with pressure.",
                ["Plan one meal or walk with a friend each week.",
                 "Say yes to one invitation you would normally skip."],
                ["Invite someone for a walk or a tea."], "30 min a week"),
]

RULES = [r_professional, r_low_interest, r_worry, r_sleep, r_movement, r_screen, r_meals,
         r_study_pressure, r_study_balance, r_financial, r_relationship, r_family,
         r_support_circle, r_college_support, r_future, r_compare, r_focus, r_health]


def strengths(x):
    """What is going well, drawn only from the student's own answers."""
    out = []
    if x["life"]["sleep_hours"] == "7-8":
        out.append("You usually get 7 to 8 hours of sleep.")
    if x["life"]["activity"] in ("3-4", "5+"):
        out.append("You are active on 3 or more days a week.")
    if x["life"]["meals"] == "regular":
        out.append("Your meals are regular.")
    if x["bg"]["friend_circle"] == "Yes":
        out.append("You have a close circle of friends.")
    if x["phq"] < 2 and x["gad"] < 2:
        out.append("Your mood answers are in the low range.")
    calm = [f["label"].lower() for f in C.FACTORS if x["f"][f["key"]] <= 2]
    if len(calm) >= 3:
        out.append(f"{len(calm)} of the {len(C.FACTORS)} pressures are low for you, including {calm[0]} and {calm[1]}.")
    return out[:5]


def build_plan(f, life, study, mood, bg, max_items=6):
    """Return the personalised plan: ordered suggestions, strengths, first-week list, flags."""
    phq = mood["phq1"] + mood["phq2"]
    gad = mood["gad1"] + mood["gad2"]
    x = {"f": f, "life": life, "study": study, "mood": mood, "phq": phq, "gad": gad, "bg": bg}

    items = [s for s in (rule(x) for rule in RULES) if s and s["score"] >= 4]
    items.sort(key=lambda s: -s["score"])
    items = items[:max_items]
    for fb in FALLBACKS:
        if len(items) >= 3:
            break
        items.append(fb)

    # First week: round-robin through the micro-actions of the top suggestions.
    week, i = [], 0
    pools = [list(s["micro"]) for s in items[:4]]
    while len(week) < 6 and any(pools):
        p = pools[i % len(pools)]
        if p:
            week.append({"text": p.pop(0), "from": items[i % len(pools)]["title"]})
        i += 1
    week.append({"text": "Review: keep what helped, drop what did not, and plan next week.", "from": "Weekly review"})
    for n, w in enumerate(week, 1):
        w["day"] = n

    return {"items": items, "strengths": strengths(x), "week": week,
            "phq": phq, "gad": gad, "screen_positive": phq >= C.MOOD_CUTOFF or gad >= C.MOOD_CUTOFF}
