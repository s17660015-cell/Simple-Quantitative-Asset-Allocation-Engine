"""Build the technical report directly from verified run outputs."""

from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PDF_PATH = ROOT / "Quantitative_Asset_Allocation_Paper.pdf"
MD_PATH = ROOT / "Quantitative_Asset_Allocation_Paper.md"
NAVY = colors.HexColor("#17324d")
BLUE = colors.HexColor("#005b91")
LIGHT = colors.HexColor("#edf3f7")
GRAY = colors.HexColor("#536370")
WIDTH = A4[0] - 100


def main():
    summary = json.loads((RESULTS / "summary.json").read_text())
    provenance = json.loads((RESULTS / "data-manifest.json").read_text())
    mvo = summary["mvo"]
    m = mvo["daily_rebalanced_in_sample_metrics"]
    rows = summary["hrp"]["metrics"]
    zero = [x for x in rows if x["cost_bps_per_traded_dollar"] == 0]
    five = [x for x in rows if x["cost_bps_per_traded_dollar"] == 5]
    by_strategy = {x["strategy"]: x for x in five}
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20,
                                leading=23, textColor=NAVY, spaceAfter=9),
        "subtitle": ParagraphStyle("subtitle", fontSize=9, leading=12,
                                   textColor=GRAY, spaceAfter=8),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=12,
                                  leading=15, textColor=BLUE, spaceBefore=9, spaceAfter=6),
        "body": ParagraphStyle("body", fontSize=9.4, leading=13.2, spaceAfter=6),
        "caption": ParagraphStyle("caption", fontSize=8.1, leading=11,
                                  textColor=GRAY, spaceBefore=5, spaceAfter=6),
        "small": ParagraphStyle("small", fontSize=8, leading=11,
                                textColor=GRAY, spaceAfter=5),
        "table": ParagraphStyle("table", fontSize=8.2, leading=10.5),
        "header": ParagraphStyle("header", fontName="Helvetica-Bold", fontSize=8.2,
                                 leading=10.5, textColor=colors.white),
        "formula": ParagraphStyle("formula", fontName="Courier", fontSize=8.4,
                                  leading=12, backColor=LIGHT, borderPadding=7, spaceAfter=9),
    }

    def p(text, style="body"):
        return Paragraph(text, styles[style])

    def table(headers, values, widths):
        content = [[p(h, "header") for h in headers]] + [
            [p(escape(str(v)), "table") for v in row] for row in values
        ]
        t = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, 0), 0.4, NAVY),
        ]))
        return t

    def figure(filename, ratio, caption):
        return KeepTogether([Image(str(RESULTS / filename), width=WIDTH, height=WIDTH / ratio),
                             p(caption, "caption")])

    story = [
        p("Portfolio allocation with<br/>maximum-Sharpe optimization and HRP", "title"),
        p("Spark Hu | Original project: November 2025 | Documentation revised: 18 October 2026", "subtitle"),
        p("A rolling hierarchical risk parity (HRP) portfolio and an equal-weight portfolio are evaluated on the same five assets over 2021-2024. HRP had lower volatility and drawdown; equal weighting achieved higher growth and Sharpe. A separate maximum-Sharpe fit illustrates how estimated means and covariances determine an allocation."),
        p("The maximum-Sharpe fit", "heading"),
        p(f"The first calculation uses MSFT, JPM, TLT, GLD and BTC-USD. Adjusted prices cover {mvo['price_start'][:10]} to {mvo['price_end'][:10]}, giving {m['observations']} simple-return observations. Prices are restricted to dates available for all five assets before returns are calculated. This retains the return from Friday to Monday, including Bitcoin's intervening weekend movement."),
        p("The optimizer maximizes estimated excess return divided by volatility, subject to nonnegative weights that sum to one. A constant effective annual cash return of 4% is converted to a daily rate. SLSQP solves the constrained problem, and the implementation checks convergence, feasibility and asset-label alignment."),
        p("r[p,t] = sum[j] w[j] r[j,t]<br/>Var[r[p]] = w' S w<br/>Sharpe[annual] = sqrt(252) * (w' mu - rf[daily]) / sqrt(w' S w)", "formula"),
        table(["Asset", "Fitted allocation"],
              [[asset, f"{weight:.2%}"]
               for asset, weight in mvo["weights"].items()], [WIDTH / 2, WIDTH / 2]),
        Spacer(1, 10),
        figure("Portfolio_Optimization_Model.png", 12 / 5.1,
               "Figure 1. Each point is a sampled allocation evaluated using the fitted means and covariance. The red point is the numerical maximum-Sharpe fit. The cloud shows feasible portfolios; computing the efficient frontier would require further optimization."),
        p(f"The fitted annualized arithmetic mean is {m['annualized_arithmetic_mean']:.2%}, with volatility {m['annualized_volatility']:.2%} and Sharpe {m['sharpe_daily_excess']:.3f}. The empirical daily fifth percentile is {m['daily_return_quantile_05']:.2%}. These values use the same observations that selected the weights and assume daily rebalancing without costs. They describe the fit rather than unseen-period performance.", "small"),
        PageBreak(),
        p("The rolling HRP test", "title"),
        p("The second calculation invests in MSFT, GOOG, JPM, XLE and GLD. Equal weighting uses these same assets and the same trading schedule, making it a direct allocation comparator. SPY is excluded from estimation and supplies a buy-and-hold market reference."),
        p("PyPortfolioOpt's HRP implementation builds a single-linkage tree from correlation distances and allocates using cluster variances. The method avoids a full covariance inverse but still relies on estimated correlations and covariances."),
        p("Each allocation uses 252 past return observations. In the first window, the last training return ends on 31 December 2020, execution is modeled at the 4 January 2021 close, and the first evaluated return ends on 5 January. Targets are updated every 21 observations. Holdings drift between updates, so the daily returns reflect the stated trading rule."),
        p(f"The evaluation ends on {zero[0]['last_return_date'][:10]} and contains {zero[0]['observations']} returns. The table and chart show the zero-cost case."),
        table(["Strategy", "Final wealth", "Compound growth*", "Annual vol.", "Sharpe", "Max. drawdown"],
              [[x["strategy"], f"{x['terminal_wealth']:.3f}",
                f"{x['annualized_compound_growth_252']:.2%}", f"{x['annualized_volatility']:.2%}",
                f"{x['sharpe_daily_excess']:.3f}", f"{x['maximum_drawdown']:.2%}"] for x in zero],
              [132, 59, 84, 68, 55, WIDTH - 398]),
        p("* Annualized compound growth is final wealth raised to 252/N, minus one. Sharpe uses daily excess returns under the 4% cash assumption. Drawdown includes initial capital.", "caption"),
        Spacer(1, 7),
        figure("Walk_Forward_Backtest.png", 10.5 / 7,
               "Figure 2. Growth and drawdown from the rolling test. HRP's lower measured risk came with lower growth than equal weighting. SPY represents a different asset exposure."),
        PageBreak(),
        p("Costs and interpretation", "title"),
        p("The cost sensitivity charges 5 basis points per dollar traded, counting both purchases and sales. Initial investment from cash is included. At subsequent updates, the charge depends on the difference between drifted holdings and the new target. Costs are deducted before the next holding block."),
        table(["Strategy", "Growth: 0 bps", "Growth: 5 bps", "Sharpe: 0 bps", "Sharpe: 5 bps"],
              [[x["strategy"], f"{x['annualized_compound_growth_252']:.2%}",
                f"{by_strategy[x['strategy']]['annualized_compound_growth_252']:.2%}",
                f"{x['sharpe_daily_excess']:.3f}",
                f"{by_strategy[x['strategy']]['sharpe_daily_excess']:.3f}"] for x in zero],
              [145, 88, 88, 86, WIDTH - 407]),
        p("What the comparison shows", "heading"),
        p(f"Equal weighting achieved annualized compound growth of {zero[1]['annualized_compound_growth_252']:.2%}, against {zero[0]['annualized_compound_growth_252']:.2%} for HRP. Its measured Sharpe was also higher. HRP reduced volatility and drawdown in this sample. This leaves a risk-return trade-off rather than an unqualified advantage for the more elaborate rule."),
        p("The maximum-Sharpe fit cannot resolve that comparison. It uses a different asset universe and selects weights on the full sample; HRP estimates its targets on rolling past windows. A matched comparison would have to align assets, estimation windows, constraints and trading assumptions."),
        p("Limits and corrections", "heading"),
        p("These results come from a small selected universe over one historical period. They have not been evaluated on an untouched holdout or subjected to a statistical superiority test. The cash rate is constant, and the 252-observation annualization does not account for serial dependence."),
        p("Adjusted daily prices approximate total returns and execution values. They do not synchronize the exact timestamps of every market or fully model slippage, impact and taxes. The empirical VaR is a return quantile, not a bound on future losses. HRP's clustering also provides no explicit sector cap."),
        p("The revision corrects mislabelled weights, return aggregation and holding-period accounting, then adds a signal delay, an equal-weight baseline and saved numerical checks. The original files remain in the archive; CHANGES.md records the corrections. A covariance-shrinkage study is planned to separate numerical conditioning from estimation error; it is not included in these historical results."),
        p("Reproducibility and sources", "heading"),
        p(f"Prices were retrieved at {escape(provenance['retrieved_at_utc'])}. The local CSV's SHA-256 is recorded in results/data-manifest.json. The source commit is ce87a9d13131d03699dd1b0834acd945c54bbb6d. Seed {summary['random_seed']}, run settings, dependency versions and numerical results are saved in results/summary.json; notebooks contain executed outputs.", "small"),
        p('<link href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678" color="#005b91">López de Prado (2016), Building Diversified Portfolios that Outperform Out-of-Sample.</link><br/>'
          '<link href="https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/hierarchical_portfolio.html" color="#005b91">PyPortfolioOpt: HRP implementation.</link><br/>'
          '<link href="https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html" color="#005b91">SciPy: SLSQP documentation.</link><br/>'
          '<link href="https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html" color="#005b91">yfinance: download and adjustment conventions.</link>', "small"),
        p("Acknowledgment: The October code corrections and this report were prepared with Codex assistance. The original project dates from November 2025.", "small"),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LIGHT)
        canvas.line(50, 39, A4[0] - 50, 39)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(50, 27, "Portfolio allocation | Technical revision | 18 October 2026")
        canvas.drawRightString(A4[0] - 50, 27, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=50, rightMargin=50,
                           topMargin=42, bottomMargin=49,
                           title="Portfolio allocation with maximum-Sharpe optimization and HRP",
                           author="Spark Hu original project; assisted technical revision")
    # Export matching wording and tables to Markdown from the same content.
    import re
    from html import unescape
    def plain(text):
        text = re.sub(r'<link href="([^"]+)"[^>]*>(.*?)</link>',
                      lambda match: f'[{match[2]}]({match[1]})', text)
        text = text.replace("<br/>", "\n").replace("<b>", "**").replace("</b>", "**")
        return unescape(re.sub(r"<[^>]+>", "", text))

    lines = []
    for item in story:
        if isinstance(item, Paragraph):
            text = plain(item.text)
            if item.style.name == "title":
                text = ("# " if not lines else "## ") + text.replace("\n", " ")
            elif item.style.name == "heading":
                text = "## " + text
            elif item.style.name == "formula":
                text = "```text\n" + text + "\n```"
            lines += [text, ""]
        elif isinstance(item, Table):
            values = [[plain(cell.text) if isinstance(cell, Paragraph) else str(cell)
                       for cell in row] for row in item._cellvalues]
            lines.append("| " + " | ".join(values[0]) + " |")
            lines.append("| " + " | ".join(["---"] * len(values[0])) + " |")
            lines += ["| " + " | ".join(row) + " |" for row in values[1:]]
            lines.append("")
        elif isinstance(item, KeepTogether):
            for part in item._content:
                if isinstance(part, Image):
                    relative = Path(part.filename).relative_to(ROOT).as_posix()
                    lines += [f"![Portfolio analysis figure]({relative})", ""]
                elif isinstance(part, Paragraph):
                    lines += [plain(part.text), ""]
    MD_PATH.write_text("\n".join(lines))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Created", PDF_PATH)
    print("Created", MD_PATH)


if __name__ == "__main__":
    main()
