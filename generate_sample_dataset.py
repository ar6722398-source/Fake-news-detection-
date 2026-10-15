"""Generates a SMALL synthetic starter dataset (data/news_dataset.csv) so the project runs
out-of-the-box with no internet.

For real results, download a real dataset (see README) and put Fake.csv / True.csv in data/ -
train_model.py will automatically prefer them.
"""
import random
from pathlib import Path
import pandas as pd

random.seed(42)

DOMAINS = {
    "health": ["vaccination programme", "dengue cases", "hospital capacity", "diabetes screening", "air quality advisory"],
    "politics": ["voter registration", "budget session", "parliamentary committee", "state election schedule", "transport bill"],
    "technology": ["data protection rules", "smartphone exports", "semiconductor plant", "AI safety guidelines", "5G rollout"],
    "science": ["lunar mission", "climate study", "telescope data", "coral reef survey", "antibiotic resistance research"],
    "economy": ["inflation figures", "interest rate decision", "unemployment data", "GDP growth estimate", "fuel prices"],
    "sports": ["cricket tournament schedule", "Olympic qualifiers", "football league", "athletics championship", "stadium renovation"],
    "education": ["exam schedule", "scholarship scheme", "curriculum revision", "university admissions", "digital classroom programme"],
    "environment": ["monsoon forecast", "river cleaning project", "forest cover report", "plastic ban", "heatwave advisory"],
}
INSTITUTIONS = ["The Ministry of Health", "The central bank", "The World Health Organization", "The space agency",
                "The Election Commission", "Researchers at a leading university", "The Supreme Court",
                "The Education Department", "The national weather service", "The statistics bureau", "A parliamentary panel"]
PERSONS = ["Dr. Meera Nair", "Rajesh Kumar", "Anita Sharma", "Prof. David Collins", "Sarah Thompson", "Vikram Mehta"]
MONTHS = ["January", "March", "May", "July", "September", "November"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

REAL_HEADLINES = [
    "{inst} releases {topic} report, shows {n} percent change from last year",
    "{inst} says {topic} figures remain under review",
    "{inst} outlines timeline for {topic}",
    "{topic} update: {inst} confirms revised figures",
    "{inst} publishes data on {topic} after {n}-week review",
]
REAL_SENTENCES = [
    "According to a statement released on {day}, {inst_l} said the {topic} would be reviewed over the next {n} weeks.",
    "Officials said the data was collected from {n} districts and will be published in full on the official website.",
    "{person}, a spokesperson, told reporters that the findings were preliminary and subject to independent verification.",
    "Analysts cautioned that it was too early to draw firm conclusions, noting that the sample size was limited.",
    "The report, reviewed by independent experts, found a {n} percent change compared with the previous quarter.",
    "Industry groups requested further details, which officials said would be shared after the review is complete.",
    "The announcement follows a consultation process that began in {month}.",
    "Further updates are expected after the committee meets next month, the statement added.",
    "{person} said the figures were compiled by the department and cross-checked with state records.",
    "The agency confirmed that the {topic} would continue as planned while feedback is collected.",
]
REAL_DRAMATIC = [
    "Critics described the {topic} situation as a crisis, though officials disputed that characterisation.",
    "The decision on {topic} triggered a sharp debate in parliament on {day}.",
]

FAKE_HEADLINES = [
    "SHOCKING: Government HIDING the truth about {topic}!!!",
    "You won't believe what they just admitted about {topic}",
    "BREAKING!!! Secret {topic} plan EXPOSED, share before it's deleted",
    "Doctors hate this: miracle cure linked to {topic} banned by officials",
    "{topic}: what the mainstream media is NOT telling you",
    "LEAKED: officials covering up {topic} scandal, insiders reveal",
    "WAKE UP people! The truth about {topic} is finally out",
]
FAKE_SENTENCES = [
    "Insiders say the {topic} is part of a secret plan that the mainstream media refuses to report.",
    "Share this with everyone you know before they delete it!!! Wake up people!",
    "An anonymous source who cannot be named revealed that officials have been lying about {topic} for years.",
    "This one weird trick is what the elites don't want you to know, and 100% of people who tried it were amazed.",
    "Thousands of people are already talking about it on social media, and the truth is finally coming out.",
    "Experts are SHOCKED and the government has gone silent after this video went viral.",
    "Forward this message to 10 friends to stop the cover-up.",
    "They don't want you to know the real {topic} numbers, but we have the proof right here.",
    "Do your own research, because the so-called experts are paid to hide the truth about {topic}.",
    "Nobody is talking about this, but a WhatsApp message from a friend of a friend confirms everything.",
]
FAKE_SUBTLE_HEADLINES = [
    "Insiders claim {topic} numbers were quietly altered",
    "Sources say {topic} decision was made behind closed doors",
]
FAKE_SUBTLE_SENTENCES = [
    "A source familiar with the matter, who asked not to be named, claimed the {topic} data had been changed.",
    "Sources say officials were told to stay silent, though no documents were shown.",
    "People close to the matter allege that the real figures are far worse than reported.",
    "The claim has been widely shared online but has not been confirmed by any official body.",
]


def fill(t, domain, topic):
    inst = random.choice(INSTITUTIONS)
    return t.format(inst=inst, inst_l=inst[0].lower() + inst[1:] if not inst.startswith(("The ", "A ")) else inst,
                    topic=topic, n=random.randint(2, 48), person=random.choice(PERSONS),
                    month=random.choice(MONTHS), day=random.choice(DAYS))


def make_article(label):
    domain = random.choice(list(DOMAINS))
    topic = random.choice(DOMAINS[domain])
    if label == "REAL":
        title = fill(random.choice(REAL_HEADLINES), domain, topic)
        pool = REAL_SENTENCES + (REAL_DRAMATIC if random.random() < 0.3 else [])
        sents = random.sample(pool, random.randint(3, 5))
    else:
        if random.random() < 0.15:  # subtle fake: calm tone, unverified claims
            title = fill(random.choice(FAKE_SUBTLE_HEADLINES), domain, topic)
            sents = random.sample(FAKE_SUBTLE_SENTENCES, 3) + random.sample(REAL_SENTENCES, 1)
        else:
            title = fill(random.choice(FAKE_HEADLINES), domain, topic)
            sents = random.sample(FAKE_SENTENCES, random.randint(3, 5))
    random.shuffle(sents)
    text = " ".join(fill(s, domain, topic) for s in sents)
    return {"title": title, "text": text, "label": label, "category": domain}


if __name__ == "__main__":
    rows = [make_article("REAL") for _ in range(500)] + [make_article("FAKE") for _ in range(500)]
    df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    out = Path(__file__).parent / "data" / "news_dataset.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
