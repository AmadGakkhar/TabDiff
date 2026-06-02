"""Generate summary charts for the session report (plotly + kaleido)."""
import plotly.graph_objects as go

OUT = "synthetic_outputs"

# --- Chart 1: fraud class balance across datasets ---
labels = ["Real<br>(original)", "Synthetic #1<br>(dist-match)", "Synthetic #2<br>(enhanced)"]
fraud = [5.98, 4.17, 11.00]
nonfraud = [100 - f for f in fraud]
fig1 = go.Figure()
fig1.add_bar(name="No fraud (0)", x=labels, y=nonfraud, marker_color="#1f9e94",
             text=[f"{v:.2f}%" for v in nonfraud], textposition="inside")
fig1.add_bar(name="Fraud (1)", x=labels, y=fraud, marker_color="#e4572e",
             text=[f"{v:.2f}%" for v in fraud], textposition="outside")
fig1.update_layout(barmode="stack", title="FraudFound_P class balance by dataset",
                   yaxis_title="% of rows", template="plotly_white",
                   width=720, height=460, legend=dict(orientation="h", y=1.08))
fig1.write_image(f"{OUT}/chart_class_balance.png", scale=2)
print("saved chart_class_balance.png")

# --- Chart 2: quality metrics, #1 vs #2 ---
metrics = ["density/Shape", "density/Trend", "density/Overall", "mle (utility)", "c2st (realism)"]
d1 = [0.9824, 0.9703, 0.9764, 0.7614, 0.8607]
d2 = [0.9791, 0.9651, 0.9721, 0.7987, 0.8323]
fig2 = go.Figure()
fig2.add_bar(name="#1 distribution-match", x=metrics, y=d1, marker_color="#1f77b4",
             text=[f"{v:.3f}" for v in d1], textposition="outside")
fig2.add_bar(name="#2 fraud-enhanced", x=metrics, y=d2, marker_color="#e4572e",
             text=[f"{v:.3f}" for v in d2], textposition="outside")
fig2.update_layout(barmode="group", title="Quality metrics (higher = better) — #1 vs #2",
                   yaxis_title="score", yaxis_range=[0, 1.05], template="plotly_white",
                   width=860, height=480, legend=dict(orientation="h", y=1.1))
fig2.write_image(f"{OUT}/chart_quality_metrics.png", scale=2)
print("saved chart_quality_metrics.png")

# --- Chart 3: training loss trajectory ---
import re
last = {}
pat = re.compile(r"Epoch (\d+)/8000:.*?TotalLoss=([0-9.]+)")
for ln in open("logs/train_full.log", errors="ignore"):
    m = pat.search(ln)
    if m:
        last[int(m.group(1))] = float(m.group(2))
es = sorted(last)
xs = [e for e in es if e % 25 == 0]
ys = [last[e] for e in xs]
fig3 = go.Figure()
fig3.add_scatter(x=xs, y=ys, mode="lines", line=dict(color="#1f9e94"), name="TotalLoss")
fig3.add_vline(x=4617, line_dash="dash", line_color="#e4572e",
               annotation_text="best EMA (4617)", annotation_position="top")
fig3.update_layout(title="Training loss (TotalLoss) vs epoch — stopped at 6000",
                   xaxis_title="epoch", yaxis_title="TotalLoss", template="plotly_white",
                   width=860, height=440)
fig3.write_image(f"{OUT}/chart_training_loss.png", scale=2)
print("saved chart_training_loss.png")
