import gradio as gr
from agent.agent import run

TICKERS = ["AAPL","MSFT","NVDA","JPM","XOM","PG","UNH","HD","CAT","KO",
           "PFE","BA","DIS","CSCO"]

def ask(question, ticker):
    if not question.strip():
        return "Enter a question.", ""
    r = run(question, ticker=ticker if ticker != "All" else None, verbose=False)
    srcs = "\n\n".join(
        f"**[{i}]** {s['ticker']} · {s['form']} · {s['fiscal_period']} · "
        f"{s['section']}\n\n> {s['text'][:400]}..."
        for i, s in enumerate(r["sources"], 1))
    return r["answer"], srcs

with gr.Blocks(title="DisclosureAlpha") as demo:
    gr.Markdown("# DisclosureAlpha\n"
                "Ask questions about SEC filings. Answers are grounded in "
                "retrieved excerpts with citations.")
    with gr.Row():
        q = gr.Textbox(label="Question", scale=3,
                       placeholder="What risks does Apple cite about suppliers?")
        t = gr.Dropdown(["All"] + TICKERS, value="AAPL", label="Company")
    btn = gr.Button("Ask", variant="primary")
    out = gr.Markdown(label="Answer")
    src = gr.Markdown(label="Sources")
    btn.click(ask, [q, t], [out, src])
    gr.Examples([["What risks does Apple cite about suppliers?", "AAPL"],
                 ["What are Microsoft's operating segments?", "MSFT"],
                 ["How is JPMorgan exposed to interest rate risk?", "JPM"]],
                [q, t])

if __name__ == "__main__":
    demo.launch()