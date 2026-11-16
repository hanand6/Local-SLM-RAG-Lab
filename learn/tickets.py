# ============================================================
# learn/tickets.py — small labelled dataset for Chapter 7
# Synthetic customer-support tickets, 4 categories × 16.
# About a third avoid the obvious keywords on purpose, so keyword
# rules miss them and you can see where ML and LLMs differ.
# ============================================================

CATEGORIES = ["billing", "technical", "account", "shipping"]

TICKETS: list[tuple[str, str]] = [
    # ── billing ──
    ("I was charged twice for my order last week", "billing"),
    ("Can I get a refund for the subscription I cancelled?", "billing"),
    ("My invoice shows the wrong amount", "billing"),
    ("Why did my monthly payment go up?", "billing"),
    ("The promo code didn't apply and I paid full price", "billing"),
    ("Please update the credit card on file", "billing"),
    ("I see a charge I don't recognise on my statement", "billing"),
    ("How do I get a VAT receipt for my purchase?", "billing"),
    ("You took money from my account after I cancelled", "billing"),
    ("Is there a discount for annual plans?", "billing"),
    ("My bank says the transaction was declined but I still lost the money", "billing"),
    ("I need a copy of last month's bill", "billing"),
    ("Why am I paying for two seats when we only use one?", "billing"),
    ("The price at checkout was higher than advertised", "billing"),
    ("Can I split the cost across two cards?", "billing"),
    ("I'd like my money back, the product never worked", "billing"),
    # ── technical ──
    ("The app crashes every time I open the camera", "technical"),
    ("I get error 500 when uploading a file", "technical"),
    ("The website won't load on Safari", "technical"),
    ("Sync between my phone and laptop stopped working", "technical"),
    ("Notifications aren't showing up anymore", "technical"),
    ("The export to PDF button does nothing", "technical"),
    ("Everything is really slow since the last update", "technical"),
    ("The screen goes blank after I log in", "technical"),
    ("Your API returns a timeout for every request", "technical"),
    ("Videos play without sound", "technical"),
    ("The search bar shows results from the wrong folder", "technical"),
    ("I keep getting a bug where my edits disappear", "technical"),
    ("The integration with Slack broke yesterday", "technical"),
    ("It freezes when I try to print", "technical"),
    ("Dark mode makes the text unreadable", "technical"),
    ("My dashboard charts are empty even though I have data", "technical"),
    # ── account ──
    ("I forgot my password and the reset email never arrives", "account"),
    ("How do I change the email address on my profile?", "account"),
    ("Please delete my account and all my data", "account"),
    ("I'm locked out after too many login attempts", "account"),
    ("Can I add a teammate to my workspace?", "account"),
    ("How do I turn on two-factor authentication?", "account"),
    ("Someone else may have accessed my account", "account"),
    ("I want to change my username", "account"),
    ("How do I transfer ownership to my colleague who is taking over?", "account"),
    ("I can't sign in with Google anymore", "account"),
    ("Please merge my two profiles into one", "account"),
    ("I never received the verification code", "account"),
    ("How do I remove a former employee's access?", "account"),
    ("My name is spelled wrong on my profile", "account"),
    ("Can I have a second login for my assistant?", "account"),
    ("I'd like to close my membership", "account"),
    # ── shipping ──
    ("My package hasn't arrived after two weeks", "shipping"),
    ("The tracking number doesn't work", "shipping"),
    ("Can I change the delivery address for my order?", "shipping"),
    ("The box arrived damaged", "shipping"),
    ("Do you deliver to Canada?", "shipping"),
    ("I received someone else's parcel", "shipping"),
    ("When will my order be dispatched?", "shipping"),
    ("The courier left it at the wrong house", "shipping"),
    ("Can I pick it up from a store instead?", "shipping"),
    ("Only half of the items came", "shipping"),
    ("How much is express delivery?", "shipping"),
    ("It says delivered but nothing is here", "shipping"),
    ("I need it before Friday, can you speed it up?", "shipping"),
    ("Where is my stuff?", "shipping"),
    ("The driver never showed up during the delivery window", "shipping"),
    ("How do I send an item back to you?", "shipping"),
]

# Simple keyword rules — what a team writes on day one.
RULES = {
    "billing":   ["charge", "charged", "refund", "invoice", "payment", "paid", "price", "bill", "card", "receipt", "discount"],
    "technical": ["crash", "error", "bug", "load", "slow", "broke", "freeze", "freezes", "not working", "stopped working", "api"],
    "account":   ["password", "login", "log in", "sign in", "account", "profile", "username", "two-factor", "verification", "access"],
    "shipping":  ["package", "parcel", "delivery", "deliver", "tracking", "shipped", "dispatched", "courier", "arrived"],
}
