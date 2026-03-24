"""
Local text analysis engine for interview transcripts.
Uses only Python's standard library.
"""

import re
import math
from collections import Counter, defaultdict

# ── Stop words ──────────────────────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "need", "that", "this", "these", "those", "i", "you",
    "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "what", "which", "who",
    "whom", "how", "when", "where", "why", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "not", "only",
    "same", "so", "than", "too", "very", "just", "also", "if", "as", "well",
    "like", "get", "got", "go", "going", "there", "here", "then", "now",
    "even", "back", "still", "actually", "yeah", "yes", "okay", "ok",
    "um", "uh", "kind", "sort", "thing", "things", "lot", "lots", "bit",
    "pretty", "quite", "much", "many", "see", "say", "said", "make", "made",
    "way", "because", "any", "one", "two", "three", "four", "five", "first",
    "second", "mean", "means", "really", "over", "re", "ve", "ll", "d",
    "m", "s", "t", "always", "never", "ever", "often", "usually",
    "sometimes", "maybe", "probably", "definitely", "especially",
    "basically", "generally", "specifically", "people", "someone",
    "something", "everything", "anything", "nothing", "know", "think",
    "thought", "feel", "felt", "use", "used", "using", "put", "come",
    "came", "take", "took", "give", "gave", "look", "looking", "try",
    "tried", "keep", "want", "wanted", "need", "needed", "let", "without",
    "between", "before", "after", "since", "while", "though", "although",
    "however", "therefore", "thus", "hence", "moreover", "furthermore",
    "instead", "otherwise", "already", "again", "another", "every",
    "just", "right", "left", "different", "new", "old", "big", "small",
    "good", "bad", "great", "little", "long", "high", "low", "next",
    "last", "own", "same", "important", "able", "example", "case",
    "point", "sense", "time", "times", "day", "week", "month", "year",
}

# ── Signal word lists ────────────────────────────────────────────────────────

PAIN_SIGNALS = [
    "frustrat", "annoying", "annoy", "difficult", "struggle", "problem",
    "issue", "broken", "confus", "hard to", "hard time", "pain", "slow",
    "takes too long", "waste", "inefficient", "can't", "cannot", "unable",
    "doesn't work", "don't work", "not working", "fail", "error", "crash",
    "bug", "worst", "terrible", "awful", "hate", "dislike", "disappoint",
    "missing", "lack", "limited", "complicated", "complex", "overwhelm",
    "tedious", "manual", "stuck", "blocker", "barrier", "challenge",
    "concern", "worry", "worried", "afraid", "fear", "risk", "unreliable",
    "inconsistent", "clunky", "awkward", "cumbersome", "messy", "unclear",
]

REQUEST_SIGNALS = [
    "would like", "wish", "hope", "hoping", "want to", "wanted to",
    "need to", "needed", "should have", "could have", "would be great",
    "would be nice", "would be helpful", "would love", "would appreciate",
    "it would help", "if only", "feature request", "request", "asking for",
    "add", "adding", "build", "building", "support", "integrate",
    "integration", "automate", "automation", "improve", "improvement",
    "better way", "easier way", "easier to", "ability to", "allow",
    "enable", "option to", "possibility", "plan to", "planning to",
    "looking for", "searching for", "missing feature", "need a way",
    "want a way", "be able to", "have the ability", "more flexible",
    "more customiz",
]

OBJECTION_SIGNALS = [
    "too expensive", "cost", "pricing", "price", "budget", "afford",
    "not sure", "unsure", "uncertain", "not convinced", "skeptic",
    "complex", "complicated", "steep learning", "learning curve",
    "time consuming", "takes too much time", "not enough time",
    "security", "privacy", "data", "compliance", "gdpr", "trust",
    "vendor lock", "lock-in", "commitment", "contract", "switching",
    "competitor", "alternative", "already have", "currently using",
    "not a priority", "low priority", "later", "eventually", "maybe later",
    "not yet", "not right now", "not ready", "not the right time",
    "management", "approval", "sign-off", "stakeholder", "team won't",
    "resistance", "pushback", "concerned about", "worried about",
    "not worth", "roi", "return on investment", "justify",
]


# ── Text utilities ───────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    text = text.lower()
    text = re.sub(r"[''']", "", text)        # remove apostrophes (don't → dont)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return [w for w in text.split() if len(w) > 2]


def content_words(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOP_WORDS and not t.isdigit()]


def sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    # Also split on newlines that look like turn boundaries
    result = []
    for part in parts:
        sub = re.split(r"\n{1,}", part)
        result.extend(s.strip() for s in sub if s.strip())
    return [s for s in result if len(s.split()) >= 4]


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


# ── TF-IDF keyword extraction ────────────────────────────────────────────────

def tfidf_keywords(docs: list[str], top_n: int = 30) -> list[tuple[str, float]]:
    """Return top keywords scored by TF-IDF across documents."""
    tokenized = [content_words(tokenize(d)) for d in docs]
    N = len(docs)

    # IDF: log(N / df)
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    idf = {w: math.log((N + 1) / (df[w] + 1)) + 1 for w in df}

    # TF-IDF sum across all docs (global importance)
    scores: Counter = Counter()
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) or 1
        for w, cnt in tf.items():
            scores[w] += (cnt / total) * idf.get(w, 1)

    return scores.most_common(top_n)


# ── Theme extraction ─────────────────────────────────────────────────────────

def extract_themes(transcripts: list[dict]) -> list[dict]:
    """
    Find recurring multi-word phrases (bigrams + trigrams) across transcripts.
    Group phrases that share a keyword into themes.
    """
    texts = [t["text"] for t in transcripts]
    n = len(texts)

    # Count bigrams and trigrams that appear in multiple transcripts
    phrase_docs: dict[tuple, set] = defaultdict(set)
    for idx, text in enumerate(texts):
        toks = content_words(tokenize(text))
        for gram in set(ngrams(toks, 2)) | set(ngrams(toks, 3)):
            phrase_docs[gram].add(idx)

    # Only keep phrases appearing in ≥2 docs (or ≥1 if only 1 transcript)
    min_docs = 2 if n > 1 else 1
    recurring = {
        gram: docs
        for gram, docs in phrase_docs.items()
        if len(docs) >= min_docs
    }

    # Score by (doc_frequency * phrase_length)
    scored = sorted(
        recurring.items(),
        key=lambda x: (len(x[1]), len(x[0])),
        reverse=True,
    )

    # Greedily merge phrases that share a keyword into theme clusters
    themes: list[dict] = []
    used_phrases: set[tuple] = set()

    for gram, doc_set in scored[:80]:
        if gram in used_phrases:
            continue
        label = " ".join(gram)
        core_words = set(gram)

        # Find related phrases (share ≥1 word)
        related = [" ".join(g) for g, _ in scored
                   if g != gram and g not in used_phrases
                   and core_words & set(g)][:4]

        for g, _ in scored:
            if core_words & set(g):
                used_phrases.add(g)

        themes.append({
            "label": label,
            "related": related,
            "doc_count": len(doc_set),
            "n": n,
        })

        if len(themes) >= 8:
            break

    return themes


# ── Sentence-level detectors ─────────────────────────────────────────────────

def _score_sentence(sentence: str, signals: list[str]) -> float:
    low = sentence.lower()
    return sum(1 for sig in signals if sig in low)


def detect_sentences(transcripts: list[dict], signals: list[str], top_n: int = 6) -> list[dict]:
    """
    Find sentences matching signal words across all transcripts.
    Return the top_n most signal-rich sentences with their source.
    """
    candidates = []
    for t in transcripts:
        for sent in sentences(t["text"]):
            score = _score_sentence(sent, signals)
            if score > 0:
                candidates.append({
                    "text": sent,
                    "score": score,
                    "source": t["label"],
                })
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate by content similarity (skip if >60% word overlap with a kept sentence)
    kept = []
    for c in candidates:
        words_c = set(tokenize(c["text"]))
        duplicate = any(
            len(words_c & set(tokenize(k["text"]))) / max(len(words_c), 1) > 0.6
            for k in kept
        )
        if not duplicate:
            kept.append(c)
        if len(kept) >= top_n:
            break
    return kept


# ── Key quotes ───────────────────────────────────────────────────────────────

def extract_quotes(transcripts: list[dict], top_n: int = 4) -> list[dict]:
    """
    Pick notable sentences: long, content-rich, and not already captured
    as pain points / requests / objections.
    """
    all_sents = []
    for t in transcripts:
        for sent in sentences(t["text"]):
            words = content_words(tokenize(sent))
            # Score: content word density, reasonable length
            if 8 <= len(sent.split()) <= 50:
                score = len(words) / max(len(sent.split()), 1)
                all_sents.append({"text": sent, "score": score, "source": t["label"]})

    all_sents.sort(key=lambda x: x["score"], reverse=True)
    kept = []
    for s in all_sents:
        words_s = set(tokenize(s["text"]))
        duplicate = any(
            len(words_s & set(tokenize(k["text"]))) / max(len(words_s), 1) > 0.5
            for k in kept
        )
        if not duplicate:
            kept.append(s)
        if len(kept) >= top_n:
            break
    return kept


# ── Recommendations ──────────────────────────────────────────────────────────

def generate_recommendations(
    themes: list[dict],
    pain_points: list[dict],
    requests: list[dict],
    objections: list[dict],
    keywords: list[tuple],
) -> list[str]:
    recs = []
    top_kws = [w for w, _ in keywords[:10]]

    if pain_points:
        top_pain = pain_points[0]["text"][:120]
        recs.append(
            f"Address the most frequently cited pain point: \"{top_pain}…\" — "
            "reduce friction at this step to improve retention and satisfaction."
        )

    if requests:
        top_req = requests[0]["text"][:120]
        recs.append(
            f"Prioritize the most-requested capability: \"{top_req}…\" — "
            "this appeared consistently across interviews."
        )

    if objections:
        obj_topics = list({o["text"].split()[0].lower() for o in objections[:3]})
        recs.append(
            f"Proactively address the top objection areas ({', '.join(obj_topics[:3])}) "
            "in your sales and onboarding materials."
        )

    if themes:
        top_theme = themes[0]["label"]
        recs.append(
            f"Invest in improving the experience around \"{top_theme}\" — "
            f"it surfaced across {themes[0]['doc_count']} of {themes[0]['n']} interview(s) "
            "as a central concern."
        )

    if top_kws:
        recs.append(
            f"Use the language customers use: '{top_kws[0]}', '{top_kws[1] if len(top_kws) > 1 else ''}', "
            f"'{top_kws[2] if len(top_kws) > 2 else ''}' — "
            "align your messaging and documentation to these terms."
        )

    return recs[:5]


# ── Main entry point ─────────────────────────────────────────────────────────

def synthesize(transcripts: list[dict]) -> str:
    """
    Analyze transcripts and return a markdown-formatted synthesis report.
    """
    n = len(transcripts)
    texts = [t["text"] for t in transcripts]
    all_text = "\n\n".join(texts)

    # Run analyses
    keywords = tfidf_keywords(texts, top_n=25)
    themes = extract_themes(transcripts)
    pain_points = detect_sentences(transcripts, PAIN_SIGNALS, top_n=6)
    requests = detect_sentences(transcripts, REQUEST_SIGNALS, top_n=6)
    objections = detect_sentences(transcripts, OBJECTION_SIGNALS, top_n=5)
    quotes = extract_quotes(transcripts, top_n=4)
    recs = generate_recommendations(themes, pain_points, requests, objections, keywords)

    # Total word count for context
    total_words = sum(len(t["text"].split()) for t in transcripts)

    # ── Build markdown output ──────────────────────────────────────────────

    lines = []

    # Header
    lines.append("## Summary")
    interview_word = "interview" if n == 1 else "interviews"
    lines.append(
        f"Analysis of **{n} {interview_word}** ({total_words:,} words total). "
        f"Top recurring keywords: {', '.join(f'**{w}**' for w, _ in keywords[:8])}. "
    )
    if n == 1:
        lines.append(
            "_Note: patterns become more reliable with multiple transcripts — "
            "add more interviews for richer synthesis._"
        )
    else:
        lines.append(
            f"Themes and patterns were cross-referenced across all {n} transcripts."
        )
    lines.append("")

    # Recurring Themes
    lines.append("## Recurring Themes")
    if themes:
        for theme in themes:
            freq = f"{theme['doc_count']}/{theme['n']} interview{'s' if theme['n'] > 1 else ''}"
            related_str = ""
            if theme["related"]:
                related_str = f" _(related: {', '.join(theme['related'][:3])})_"
            lines.append(f"- **{theme['label']}** — mentioned in {freq}{related_str}")
    else:
        lines.append("- No strong recurring phrases detected across transcripts.")
    lines.append("")

    # Pain Points
    lines.append("## Pain Points")
    if pain_points:
        for pp in pain_points:
            lines.append(f"- _{pp['text'].strip()}_ **[{pp['source']}]**")
    else:
        lines.append("- No explicit pain points detected.")
    lines.append("")

    # Feature Requests
    lines.append("## Feature Requests")
    if requests:
        for req in requests:
            lines.append(f"- _{req['text'].strip()}_ **[{req['source']}]**")
    else:
        lines.append("- No explicit feature requests detected.")
    lines.append("")

    # Objections
    lines.append("## Objections")
    if objections:
        for obj in objections:
            lines.append(f"- _{obj['text'].strip()}_ **[{obj['source']}]**")
    else:
        lines.append("- No explicit objections detected.")
    lines.append("")

    # Key Quotes
    lines.append("## Key Quotes")
    if quotes:
        for q in quotes:
            lines.append(f"> \"{q['text'].strip()}\"")
            lines.append(f"> — _{q['source']}_")
            lines.append("")
    else:
        lines.append("- No suitable quotes extracted.")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    if recs:
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
    else:
        lines.append("1. Add more transcripts to generate meaningful recommendations.")
    lines.append("")

    # Top Keywords (bonus section)
    lines.append("## Top Keywords")
    kw_items = [f"**{w}** ({score:.2f})" for w, score in keywords[:20]]
    lines.append(", ".join(kw_items))

    return "\n".join(lines)
