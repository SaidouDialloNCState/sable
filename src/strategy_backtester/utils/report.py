from __future__ import annotations
import base64, io, datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from ..backtest import metrics as M

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def write_html_report(returns: pd.Series, title: str, out_path: str, params: dict|None=None) -> str:
    # Metrics
    summary = {
        "CAGR": M.annualized_return(returns),
        "Volatility": M.annualized_vol(returns),
        "Sharpe": M.sharpe(returns),
        "MaxDrawdown": M.max_drawdown(returns),
    }

    # Equity
    eq = M.equity_curve(returns)
    fig1, ax1 = plt.subplots(); eq.plot(ax=ax1)
    ax1.set_title("Equity Curve"); ax1.set_xlabel("Date"); ax1.set_ylabel("Equity (normalized)")
    img1 = _fig_to_b64(fig1)

    # Drawdown
    dd = (eq / eq.cummax()) - 1.0
    fig2, ax2 = plt.subplots(); dd.plot(ax=ax2)
    ax2.set_title("Drawdown"); ax2.set_xlabel("Date"); ax2.set_ylabel("Drawdown")
    img2 = _fig_to_b64(fig2)

    # Rolling Sharpe (126d)
    r = returns.fillna(0); rm = r.rolling(126).mean(); rs = r.rolling(126).std(ddof=0).replace(0, pd.NA)
    roll_sharpe = (rm / rs).fillna(0) * (252 ** 0.5)
    fig3, ax3 = plt.subplots(); roll_sharpe.plot(ax=ax3)
    ax3.set_title("Rolling Sharpe (126d)"); ax3.set_xlabel("Date"); ax3.set_ylabel("Sharpe")
    img3 = _fig_to_b64(fig3)

    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in (params or {}).items())
    metrics_rows = "".join(f"<tr><td>{k}</td><td>{summary[k]:.4f}</td></tr>" for k in ["CAGR","Volatility","Sharpe","MaxDrawdown"])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title} - Report</title></head>
<body>
<h1>{title}</h1><p>Generated: {ts}</p>
{'<h3>Parameters</h3><table border="1" cellpadding="4">'+params_rows+'</table>' if params_rows else ''}
<h3>Metrics</h3><table border="1" cellpadding="4">{metrics_rows}</table>
<h3>Equity Curve</h3><img src="data:image/png;base64,{img1}"/>
<h3>Drawdown</h3><img src="data:image/png;base64,{img2}"/>
<h3>Rolling Sharpe (126d)</h3><img src="data:image/png;base64,{img3}"/>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
