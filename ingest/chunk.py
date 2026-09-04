import re
import uuid
import warnings
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# That XML warning you saw is harmless (some filings are XML, not HTML).
# This line just silences the noise.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = Path(__file__).resolve().parents[1]

MIN_CHARS = 800        # a real section is long; a shorter span is a pointer / cross-ref

# Rough upper bound on how long each section can plausibly get, in characters -
# a guard against a surviving cross-reference in a header-sparse stretch of the
# document running away to the end of the file.  (JPMorgan's risk factors alone
# run ~110k; Coca-Cola's MD&A ~105k.)
SECTION_MAX = {
    "Business": 140_000,
    "Risk Factors": 260_000,
    "MD&A": 360_000,
    "Market Risk": 45_000,
}

# ---------------------------------------------------------------------------
# Matching a filing's section headers is fiddly, because every issuer prints
# them differently:
#     "Item 1A. Risk Factors"      (Apple  - title case, period)
#     "ITEM 1A. RISK FACTORS"      (Microsoft, Coca-Cola, UnitedHealth - all caps)
#     "ITEM 1A. RIS K FACTORS"     (Microsoft - a styled drop-cap letter gets
#                                   split off by a stray space)
#     "Item 1A: Risk Factors"      (colon instead of a period)
# and a 10-K and a 10-Q number the same logical section differently (MD&A is
# Item 7 in a 10-K but Item 2 in a 10-Q; Market Risk is 7A vs 3).  Several
# issuers (Cisco, Pfizer, Intel, Disney) also lay the header - or the whole
# section body - out inside <table> elements, which clean_text() now keeps when
# they hold prose rather than numbers.
#
# So everything below is case-insensitive, tolerates stray whitespace inside the
# spelled-out title, and is anchored on BOTH the item number and the title -
# matching the title is what keeps us from grabbing a passing cross-reference.

def _loose(phrase: str) -> str:
    """Turn 'risk factors' into a regex that shrugs off caps and stray spaces."""
    return "".join(r"\s+" if c == " " else re.escape(c) + r"\s*" for c in phrase)

_ITEM      = r"(?:part\s+[ivxlc]+[\s,]*)?item\s+"   # optional "Part II " then "Item "
_SEP       = r"\s*[.:)\-—–]?\s*"        # separator when we're FINDING a section
_SEP_BOUND = r"\s*[.:]\s+"              # separator a real header always has (period/colon)

_MGMT_DISC = _loose("management") + r"[’'`\s]*s?\s*" + _loose("discussion and analysis")
_MKT_RISK  = _loose("quantitative and qualitative disclosures about market risk")

# name -> (item-number regex, title regex, [standalone heading regexes])
# A standalone heading (no "Item N" in front) is the fallback for issuers that
# bury the numbered header in a table but still print the spelled-out title as a
# bold heading above the text (Cisco / JPMorgan MD&A, Pfizer MD&A).
SECTION_SPECS = {
    "Business":     (r"1(?![0-9a-z])",       _loose("business"), [
        _loose("business overview"),
    ]),
    "Risk Factors": (r"1\s*a(?![0-9a-z])",   _loose("risk factors"), [
        r"(?<=[.:;] )" + _loose("risk factors") + r"(?=[A-Z])",
    ]),
    "MD&A":         (r"(?:7|2)(?![0-9])",    _MGMT_DISC, [
        _MGMT_DISC + r"\s+" + _loose("of") + r"\s*(?:the\s+)?"
        + _loose("financial condition and results of operations"),
        _loose("analysis of financial condition"),   # Pfizer's wording
        _MGMT_DISC,                                  # P&G 10-Q ("... (MD&A) ...")
    ]),
    "Market Risk":  (r"(?:7\s*a|3)(?![0-9])", _MKT_RISK, [_MKT_RISK]),
}

# The standard Form 10-K / 10-Q item titles.  A real section boundary is an item
# number + a period/colon + one of THESE - not arbitrary body text, which is
# what a running page-header ("PART I Item 1A  Business model competition ...")
# looks like.  Loose-matched so styled drop-caps ("PR OPERTIES") still land.
_STD_TITLES = "|".join(_loose(t) for t in [
    "business", "risk factors", "unresolved staff comments", "cybersecurity",
    "information about our executive officers", "properties", "legal proceedings",
    "mine safety disclosures", "mine safety", "market for", "reserved",
    "selected financial data", "quantitative and qualitative", "quantitative",
    "financial statements", "changes in and disagreements",
    "controls and procedures", "other information",
    "disclosure regarding foreign jurisdictions", "directors",
    "executive compensation", "security ownership", "certain relationships",
    "principal account", "exhibit", "form 10-k summary",
    "defaults upon senior securities", "unregistered sales",
])
_GENERIC = re.compile(
    r"(?i:" + _ITEM + r"\d{1,2}\s*[a-e]?" + _SEP_BOUND
    + r"(?:" + _STD_TITLES + r"|" + _MGMT_DISC + r"))"
)

# The auditor's report is the cleanest "the narrative is over now" marker - used
# only to bound a section we found from a standalone heading (no item numbers).
_FIN_STMTS = re.compile(r"(?<!the )(?<!our )report of independent registered public "
                        r"accounting firm", re.IGNORECASE)

# Right after the title, only a handful of things reliably mean "pointer, not the
# section itself" - most real sections open with prose that happens to contain
# words like "in this report" or "set forth below", so we stay narrow here:
#   "Risk Factors - Global Operations section"   (a named sub-item)
#   "Risk Factors of this Form 10-K on pages 9-31"
#   "Item 1. Business" of this report.  Some ...   (closing a quoted reference)
_XREF_AFTER = re.compile(
    r"^(?:"
    r"\s*[—–-]{1,3}\s*[“\"'A-Z]"
    r"|[\s.,;:)”\"'’]*(?:of (?:this|our|the)\b|on pages?\b|section[s)]?\b)"
    r"|[”\"'][\s.,]*(?:of|in|and|including|for|as)\b"
    r")",
    re.IGNORECASE,
)
# A pointer to a document, or to another item, within a few words of the title.
_XREF_DOC = re.compile(
    r"\b(?:form 10-[kq]|annual report|proxy statement|of this report\b"
    r"|(?:beginning )?on pages?\s+\d)",
    re.IGNORECASE,
)
# Just before a header, a quote mark or an explicit "see / refer to / in ..." cue
# (optionally with a short filler word: "including this", "set forth in our").
_XREF_BEFORE = re.compile(
    r"(?:including|see|refer to|captioned|entitled|pursuant to|described in|"
    r"set forth in|contained in|discussed in|as defined in|under the caption|"
    r"within|conjunction with|found in|in)"
    r"(?:\s+(?:this|the|our|its|such|each|part))?\s*[“\"'(]?\s*$|[“\"]\s*$",
    re.IGNORECASE,
)
# A header that runs on from the middle of a sentence (preceded by a comma or a
# connective, with no sentence break) is a mention, not a real header.
_MID_SENTENCE = re.compile(r"(?:,|\b(?:with|and|from|conjunction with))\s*$", re.IGNORECASE)
# A dash between the number and the title ("Item 1A - Risk Factors") is how some
# issuers (Disney) write the cross-reference - the real header uses a period.
_XREF_DASH = re.compile(r"^" + _ITEM + r"\d{1,2}\s*[a-e]?\s*[-—–]", re.IGNORECASE)


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # A <table> is dropped when it is a number grid (financial statements, data
    # tables) or a table-of-contents, and KEPT when it carries prose - a laid-out
    # section header, or, for issuers like Intel and Disney, the section body.
    for tbl in soup.find_all("table"):
        t = re.sub(r"\s+", " ", tbl.get_text(" ")).strip()
        digits = sum(c.isdigit() for c in t)
        rows = len(tbl.find_all("tr"))
        header = len(t) < 90 and len(re.findall(r"\d", t)) <= 2 and (
            re.match(r"(?:part\s+[ivxlc]+[\s,]*)?item\s+\d{1,2}\s*[a-e]?\s*[.:)]?\s*[A-Za-z]",
                     t, re.IGNORECASE)
            or re.fullmatch(r"(?:management[’'`]s )?discussion and analysis.*|risk factors"
                            r"|quantitative and qualitative disclosures about market risk"
                            r"|properties", t, re.IGNORECASE)
        )
        prose = len(t) > 400 and len(t.split()) > 60 and digits / len(t) < 0.10 and rows < 15
        if header or prose:
            tbl.replace_with(" " + t + " ")
        else:
            tbl.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def _is_xref(text: str, start: int, end: int) -> bool:
    """True if a header match at [start:end] is a cross-reference, not a real header."""
    if _XREF_AFTER.match(text[end:end + 40]):
        return True
    if _XREF_DOC.search(text[end:end + 55]):
        return True
    if _XREF_BEFORE.search(text[max(0, start - 40):start]):
        return True
    if _XREF_DASH.match(text[start:end]):
        return True
    return bool(_MID_SENTENCE.search(text[max(0, start - 25):start].rstrip()))


def _boundaries(text: str) -> list[int]:
    """Positions of every real (non-cross-reference) item header in the document."""
    return sorted({m.start() for m in _GENERIC.finditer(text)
                   if not _is_xref(text, m.start(), m.end())})


_LEAD_ITEM = re.compile(r"(?:part\s+[ivxlc]+[\s,]*)?item\s+\d{1,2}\s*[a-e]?\s*[.:]\s*$", re.IGNORECASE)


def _match_starts(text: str, patterns: list[str]):
    hits = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            start = m.start()
            # a standalone title can sit right after "Part II, Item 7." - judge
            # the cross-reference test from the front of that item number.
            lead = _LEAD_ITEM.search(text[max(0, start - 45):start])
            if lead:
                start -= len(lead.group())
            if not _is_xref(text, start, m.end()):
                hits.add((start, m.end()))
    return sorted(hits)


def _starts(text: str, num_re: str, title_re: str, standalone_res: list[str]):
    """Plausible starting positions for a section: the numbered header
    ("Item 7. Management's Discussion...") and, as a fallback for issuers that
    bury the numbered header in a stripped table, the spelled-out heading."""
    numbered = _match_starts(text, [_ITEM + r"(?:" + num_re + r")" + _SEP + title_re])
    standalone = _match_starts(text, list(standalone_res)) if standalone_res else []
    return numbered, standalone


def _pick(text, starts, bounds, cap):
    """Of the candidate starts for a section, keep the one that spans the most
    text before the next header - a real header is followed by the real
    disclosure; a leftover pointer or a table-of-contents line collapses to
    almost nothing and loses here."""
    best = None
    for start, _ in starts:
        nxt = next((b for b in bounds if b > start + 200), len(text))
        span = text[start:min(start + cap, nxt)]
        if best is None or len(span) > len(best[1]):
            best = (start, span)
    return best


def find_sections(text: str) -> dict[str, str]:
    bounds = _boundaries(text)

    fin = next((m.start() for m in _FIN_STMTS.finditer(text)), len(text))

    # pass 1: a first cut at where each section starts and how far it runs
    picks = {}
    for name, (num_re, title_re, standalone) in SECTION_SPECS.items():
        cap = SECTION_MAX[name]
        numbered, alt = _starts(text, num_re, title_re, standalone)
        got = _pick(text, numbered, bounds, cap)
        if (got is None or len(got[1]) < 5_000) and alt:
            fb = _pick(text, numbered + alt, bounds, cap)
            # a fallback heading (no item numbers to lean on) is only trustworthy
            # if something actually closes the section - a later header, the next
            # section, or the auditor's report.  An open-ended run to the cap is
            # almost always a cross-reference that swallowed the rest of the file.
            if fb and fb[0] not in {s for s, _ in numbered}:
                end = fb[0] + len(fb[1])
                if fin - fb[0] > MIN_CHARS and fin < end:
                    fb = (fb[0], text[fb[0]:fin])
                    end = fin
                if end < fb[0] + cap - 1 and end < len(text) - 1:
                    got = fb
            elif fb:
                got = fb
        if got:
            picks[name] = got  # (start, span)

    # pass 2: a section also ends where the next section begins
    other_starts = sorted(s for s, _ in picks.values())
    result = {}
    for name, (start, span) in picks.items():
        end = start + len(span)
        nxt_sec = next((s for s in other_starts if s > start + MIN_CHARS), end)
        end = min(end, nxt_sec)
        span = text[start:end]
        if len(span) >= MIN_CHARS:
            result[name] = span
    return result


def chunk_section(text: str, size=1200, overlap=200) -> list[str]:
    words, out, i = text.split(), [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        i += size - overlap
    return out


def main():
    manifest = pd.read_parquet(ROOT / "data" / "processed" / "manifest.parquet")
    all_chunks = []
    for _, row in manifest.iterrows():
        html = Path(row["local_path"]).read_text(encoding="utf-8", errors="ignore")
        for section, sec_text in find_sections(clean_text(html)).items():
            for piece in chunk_section(sec_text):
                all_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "ticker": row["ticker"], "form": row["form"],
                    "fiscal_period": row["reportDate"],
                    "section": section, "text": piece,
                })
    df = pd.DataFrame(all_chunks)
    df.to_parquet(ROOT / "data" / "processed" / "chunks.parquet", index=False)
    print(f"{len(df)} chunks. Sections found: {df.section.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
