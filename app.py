import os
import time
import threading
from collections import defaultdict

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from agent.agent import run  # noqa: E402  (after load_dotenv so env vars are set)

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "PG", "UNH", "HD", "CAT",
           "KO", "PFE", "BA", "DIS", "CSCO"]

# ---------- rate limiting ----------
MAX_QUERIES_PER_DAY = int(os.environ.get("MAX_QUERIES_PER_DAY", "20"))
WINDOW = 86_400  # 24 hours in seconds

_usage = defaultdict(list)   # ip -> [timestamp, ...]
_lock = threading.Lock()


def _check_rate_limit(request: gr.Request) -> bool:
    """Return True if allowed, False if rate-limited."""
    ip = request.client.host if request else "unknown"
    now = time.time()
    with _lock:
        # prune old entries
        _usage[ip] = [t for t in _usage[ip] if now - t < WINDOW]
        if len(_usage[ip]) >= MAX_QUERIES_PER_DAY:
            return False
        _usage[ip].append(now)
        return True


def _remaining(request: gr.Request) -> int:
    ip = request.client.host if request else "unknown"
    now = time.time()
    with _lock:
        recent = [t for t in _usage.get(ip, []) if now - t < WINDOW]
    return max(0, MAX_QUERIES_PER_DAY - len(recent))


# ---------- handler ----------
def ask(question, ticker, request: gr.Request):
    if not question.strip():
        return "Enter a question.", "", ""

    if not _check_rate_limit(request):
        return ("⚠️ **Rate limit reached.** You've used all "
                f"{MAX_QUERIES_PER_DAY} queries for today. "
                "Please try again in 24 hours."), "", ""

    remaining = _remaining(request)

    r = run(question, ticker=ticker if ticker != "All" else None, verbose=False)
    srcs = "\n\n".join(
        f"**[{i}]** {s['ticker']} · {s['form']} · {s['fiscal_period']} · "
        f"{s['section']}\n\n> {s['text'][:400]}..."
        for i, s in enumerate(r["sources"], 1))

    status = f"✅ {remaining} queries remaining today"
    return r["answer"], srcs, status


# ---------- UI ----------
with gr.Blocks(
    title="DisclosureAlpha",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css="""
    .rate-info { font-size: 0.85em; opacity: 0.7; }
    footer { display: none !important; }
    """
) as demo:
    gr.Markdown(
        "# 📊 DisclosureAlpha\n"
        "Ask questions about SEC filings for 14 large-cap US companies. "
        "Answers are grounded in retrieved excerpts with inline citations.\n\n"
        f"*Rate limit: {MAX_QUERIES_PER_DAY} queries per day per visitor.*"
    )
    with gr.Row():
        q = gr.Textbox(label="Question", scale=3,
                       placeholder="What risks does Apple cite about suppliers?")
        t = gr.Dropdown(["All"] + TICKERS, value="AAPL", label="Company")
    btn = gr.Button("Ask", variant="primary")
    status = gr.Markdown(elem_classes=["rate-info"])
    out = gr.Markdown(label="Answer")
    src = gr.Markdown(label="Sources")
    btn.click(ask, [q, t], [out, src, status])
    q.submit(ask, [q, t], [out, src, status])
    gr.Examples(
        [["What risks does Apple cite about suppliers?", "AAPL"],
         ["What are Microsoft's operating segments?", "MSFT"],
         ["How is JPMorgan exposed to interest rate risk?", "JPM"]],
        [q, t])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)