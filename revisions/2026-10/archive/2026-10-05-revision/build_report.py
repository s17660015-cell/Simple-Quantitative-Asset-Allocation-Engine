"""Build a corrected technical-note draft directly from verified run outputs."""

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
        p("Quantitative Asset Allocation<br/>Corrected Demonstrations", "title"),
        p("Spark Hu | Original project: November 2025 | Assisted technical revision: 5 October 2026", "subtitle"),
        p("<b>Draft for author review.</b> This note documents an in-sample maximum-Sharpe allocation and a separate rolling HRP experiment. The two experiments use different asset universes and do not establish HRP's superiority over MVO. The earlier version is retained in a labelled archive."),
        p("1. In-sample maximum-Sharpe allocation", "heading"),
        p(f"Adjusted common-date prices for MSFT, JPM, TLT, GLD and BTC-USD run from {mvo['price_start'][:10]} to {mvo['price_end'][:10]}. There are {m['observations']} simple-return observations. Complete price rows are aligned before differencing; crypto movement over intervening weekends is included in the next common-date return."),
        p("The fitted long-only allocation maximizes estimated excess return per unit of volatility, with weights summing to one. The cash-return assumption is a constant effective annual 4%, converted consistently to a daily rate. SLSQP convergence and feasibility are checked; asset labels are preserved."),
        p("r[p,t] = sum[j] w[j] r[j,t]<br/>Var[r[p]] = w' S w<br/>Sharpe[annual] = sqrt(252) * (w' mu - rf[daily]) / sqrt(w' S w)", "formula"),
        table(["Asset", "Fitted allocation", "Interpretation"],
              [[asset, f"{weight:.2%}", "Historical in-sample numerical solution"]
               for asset, weight in mvo["weights"].items()], [85, 100, WIDTH - 185]),
        Spacer(1, 10),
        figure("Portfolio_Optimization_Model.png", 12 / 5.1,
               "Figure 1. Seeded random feasible allocations and fitted sample correlations. The cloud evaluates weight vectors under the same estimated moments; it is neither a future-return simulation nor a computed efficient frontier."),
        p(f"The fitted allocation has estimated annualized arithmetic mean {m['annualized_arithmetic_mean']:.2%}, annualized volatility {m['annualized_volatility']:.2%} and daily-excess Sharpe {m['sharpe_daily_excess']:.3f}. Its empirical daily fifth percentile is {m['daily_return_quantile_05']:.2%}. These statistics use daily restored weights in the fitting sample, with no costs; they are not out-of-sample performance or a future loss guarantee.", "small"),
        PageBreak(),
        p("2. Rolling HRP allocation experiment", "title"),
        p("MSFT, GOOG, JPM, XLE and GLD are the investable assets. SPY is excluded from fitting and provides a buy-and-hold market reference. Equal weighting uses the same five assets, dates, holding schedule and cost assumptions as HRP."),
        p("Each HRP target uses 252 past return observations and explicitly specified single-linkage clustering. The latest training return ends one observation before the execution close. The first test return follows that execution close. New targets are set every 21 retained observations, with holdings allowed to drift inside each block."),
        p("HRP uses correlation-based distances and covariance-based cluster variances. It avoids a full covariance inverse but still depends on historical estimates and does not impose a sector allocation cap. The implementation is supplied by PyPortfolioOpt."),
        p(f"Evaluation covers {zero[0]['first_return_date'][:10]} to {zero[0]['last_return_date'][:10]}, with {zero[0]['observations']} return observations. The table and chart below show the declared zero-cost case."),
        table(["Strategy", "Final wealth", "Compound growth*", "Annual vol.", "Sharpe", "Max. drawdown"],
              [[x["strategy"], f"{x['terminal_wealth']:.3f}",
                f"{x['annualized_compound_growth_252']:.2%}", f"{x['annualized_volatility']:.2%}",
                f"{x['sharpe_daily_excess']:.3f}", f"{x['maximum_drawdown']:.2%}"] for x in zero],
              [132, 59, 84, 68, 55, WIDTH - 398]),
        p("* Compound growth uses 252/N annualization, not exact calendar elapsed time. Sharpe uses the same constant cash-return assumption. Drawdown includes initial capital.", "caption"),
        Spacer(1, 7),
        figure("Walk_Forward_Backtest.png", 10.5 / 7,
               "Figure 2. Out-of-window realized return accounting for the rolling rules, with initial wealth included. Equal weighting is a same-asset allocation control; SPY has a different exposure. Observed differences alone do not establish significance or general superiority."),
        PageBreak(),
        p("3. Costs, corrections and limitations", "title"),
        p("A second case charges 5 basis points per dollar traded. Purchases and sales both count; initial investment from cash is included. Later trade sizes are measured against drifted prior weights. The proportional charge is applied before each new holding block. This is an illustrative sensitivity model, rather than a complete execution-cost estimate."),
        table(["Strategy", "Growth: 0 bps", "Growth: 5 bps", "Sharpe: 0 bps", "Sharpe: 5 bps"],
              [[x["strategy"], f"{x['annualized_compound_growth_252']:.2%}",
                f"{by_strategy[x['strategy']]['annualized_compound_growth_252']:.2%}",
                f"{x['sharpe_daily_excess']:.3f}",
                f"{by_strategy[x['strategy']]['sharpe_daily_excess']:.3f}"] for x in zero],
              [145, 88, 88, 86, WIDTH - 407]),
        p("Corrections to the earlier version", "heading"),
        p("The revision fixes asset-label alignment, mixed-calendar differencing, weighted-log-return aggregation and the mismatch between fixed daily weights and a monthly holding claim. It also adds an explicit signal delay, a same-asset baseline, cost sensitivity, numerical checks, saved execution records and narrower report language. The original archive and CHANGES.md retain the correction history."),
        p("Limits of interpretation", "heading"),
        p("1. The maximum-Sharpe fit is in-sample, whereas the HRP rules are estimated on rolling past windows. Different assets and evaluation procedures prevent a matched MVO-versus-HRP conclusion."),
        p("2. A small selected universe and a single historical period do not establish future performance. The original selection chronology is unknown, and no untouched holdout or statistical superiority test is claimed."),
        p("3. Adjusted daily prices approximate total returns and execution values. Cross-market timestamps, corporate-action conventions, market impact, taxes and other execution details are not fully modeled. A signal delay makes chronology clearer but does not remove every possible source of bias."),
        p("4. Mean, covariance, volatility annualization and the constant cash return are modeling choices. Serial dependence, changing regimes and estimation uncertainty can alter their interpretation. The empirical VaR is a historical quantile, not a maximum-loss guarantee."),
        p("5. Avoiding an explicit covariance inverse is not evidence of statistically accurate risk estimates. The proposed covariance-estimation and shrinkage study remains a separate, unfinished extension."),
        p("Reproducibility and sources", "heading"),
        p(f"Prices were retrieved at {escape(provenance['retrieved_at_utc'])}. The local CSV's SHA-256 is recorded in results/data-manifest.json. The source commit is ce87a9d13131d03699dd1b0834acd945c54bbb6d. Seed {summary['random_seed']}, run settings, dependency versions and numerical results are saved in results/summary.json; notebooks contain executed outputs.", "small"),
        p('<link href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678" color="#005b91">López de Prado (2016), Building Diversified Portfolios that Outperform Out-of-Sample.</link><br/>'
          '<link href="https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/hierarchical_portfolio.html" color="#005b91">PyPortfolioOpt: HRP implementation.</link><br/>'
          '<link href="https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html" color="#005b91">SciPy: SLSQP documentation.</link><br/>'
          '<link href="https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html" color="#005b91">yfinance: download and adjustment conventions.</link>', "small"),
        p("Revision status: The October implementation, checks and this explanatory draft were prepared with Codex assistance. They are supplied for the original author's review, reproduction and interpretation. This note does not claim that the author independently produced the assisted revision or completed the proposed mathematical extension.", "small"),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LIGHT)
        canvas.line(50, 39, A4[0] - 50, 39)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(50, 27, "Corrected demonstration | Draft for author review | 5 October 2026")
        canvas.drawRightString(A4[0] - 50, 27, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=50, rightMargin=50,
                           topMargin=42, bottomMargin=49,
                           title="Quantitative Asset Allocation: Corrected Demonstrations",
                           author="Spark Hu original project; assisted technical revision")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    lines = ["# Quantitative Asset Allocation: Corrected Demonstrations", "",
             "Original project: Spark Hu, November 2025. Assisted technical revision: 5 October 2026. Draft for author review.", "",
             "Two separate demonstrations are presented; no matched HRP-versus-MVO superiority claim is made.", "",
             "## In-sample maximum-Sharpe allocation", "",
             f"Common prices: {mvo['price_start'][:10]} to {mvo['price_end'][:10]}; {m['observations']} return observations.", "",
             "| Asset | Weight |", "|---|---:|"]
    lines += [f"| {asset} | {weight:.4%} |" for asset, weight in mvo["weights"].items()]
    lines += ["", f"Estimated annualized arithmetic mean: {m['annualized_arithmetic_mean']:.4%}; volatility: {m['annualized_volatility']:.4%}; daily-excess Sharpe: {m['sharpe_daily_excess']:.4f}. Historical daily fifth percentile: {m['daily_return_quantile_05']:.4%}.", "",
              "These statistics assume daily restoration of the fitted weights in the same sample and no costs. The quantile is not a future loss guarantee.", "",
              "![Random feasible portfolios and correlations](results/Portfolio_Optimization_Model.png)", "",
              "## Rolling HRP experiment", "",
              "Investable assets: MSFT, GOOG, JPM, XLE and GLD. SPY is excluded from fitting. Equal weighting uses the same assets and schedule. Settings: 252-observation lookback, one-observation signal delay, single linkage, and 21-observation holding blocks with drift.", "",
              "| Strategy | Cost bps | Compound growth (252/N) | Annual volatility | Sharpe | Maximum drawdown |",
              "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {x['strategy']} | {x['cost_bps_per_traded_dollar']} | {x['annualized_compound_growth_252']:.4%} | {x['annualized_volatility']:.4%} | {x['sharpe_daily_excess']:.4f} | {x['maximum_drawdown']:.4%} |" for x in rows]
    lines += ["", "![Walk-forward wealth and drawdowns](results/Walk_Forward_Backtest.png)", "",
              "The 5-bps case charges purchases and sales, including initial investment. Costs are illustrative. Growth uses 252/N annualization; daily excess returns use a constant effective annual cash return of 4%. Drawdown includes initial capital.", "",
              "## Interpretation and limits", "",
              "These selected assets and dates are an illustrative case. Historical data are freshly retrieved adjusted-price observations, not an original point-in-time execution feed. MVO's universe and fitting procedure differ from HRP's. No general superiority, future loss bound, statistical significance or completed covariance-shrinkage extension is claimed. See CHANGES.md and WALKTHROUGH.md for corrections and full explanations.", "",
              "## Reproducibility", "",
              f"Retrieval: {provenance['retrieved_at_utc']}. Data hash: `{provenance['sha256']}`. Original commit: `{summary['original_source_commit']}`. Random seed: {summary['random_seed']}. Exact versions and numerical outputs are recorded in results/summary.json and requirements-lock.txt.", "",
              "## Method sources", "",
              "- [López de Prado (2016), HRP](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)",
              "- [PyPortfolioOpt implementation](https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/hierarchical_portfolio.html)",
              "- [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)",
              "- [yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)", "",
              "The October correction and explanatory draft were prepared with Codex assistance for author review. The author should reproduce the calculations and supply their own interpretation before presenting this as evidence of their understanding."]
    MD_PATH.write_text("\n".join(lines) + "\n")
    print("Created", PDF_PATH)
    print("Created", MD_PATH)


if __name__ == "__main__":
    main()
