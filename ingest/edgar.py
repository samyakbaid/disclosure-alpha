import os, time, requests
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
UA = os.environ["SEC_USER_AGENT"]
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}

def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(0.15)          # stay under 10 req/s with margin
    return r

@lru_cache(maxsize=1)
def ticker_to_cik() -> dict[str, str]:
    """Map upper-case ticker -> zero-padded 10-digit CIK."""
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    return {row["ticker"].upper(): str(row["cik_str"]).zfill(10)
            for row in data.values()}

def get_submissions(cik: str) -> dict:
    """Filing history for a CIK (zero-padded)."""
    return _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()

def recent_filings(ticker: str, forms=("10-K", "10-Q"), n=8) -> list[dict]:
    """Return up to n recent filings of the given form types for a ticker."""
    cik = ticker_to_cik()[ticker.upper()]
    recent = get_submissions(cik)["filings"]["recent"]
    out, cols = [], ["accessionNumber", "form", "filingDate",
                     "primaryDocument", "reportDate"]
    for i in range(len(recent["accessionNumber"])):
        if recent["form"][i] in forms:
            row = {c: recent[c][i] for c in cols}
            acc = row["accessionNumber"].replace("-", "")
            row["cik"] = cik
            row["url"] = (f"https://www.sec.gov/Archives/edgar/data/"
                          f"{int(cik)}/{acc}/{row['primaryDocument']}")
            out.append(row)
        if len(out) >= n:
            break
    return out

def get_company_facts(cik: str) -> dict:
    """All XBRL facts (structured fundamentals) for a CIK."""
    return _get(f"https://data.sec.gov/api/xbrl/"
                f"companyfacts/CIK{cik}.json").json()