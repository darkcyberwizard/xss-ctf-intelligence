"""
gradio_app.py - Gradio UI for XSS CTF Intelligence Platform

Two tabs:
1. CTF Challenge Generator
2. Learning Analytics Dashboard
"""
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats

import os
from app.generator import generate_challenge, format_challenge_markdown
from app.analytics import load_and_validate, compute_summary_stats, segment_students
from app.config import VULN_TYPES, DIFFICULTIES, CONTEXTS, XSS_SUBTYPES

# Load token from Space secret or .env at startup
HF_TOKEN_LOADED = os.getenv("HF_TOKEN", "")
if HF_TOKEN_LOADED:
    os.environ["HF_TOKEN"] = HF_TOKEN_LOADED


# ── Tab 1: CTF Generator ───────────────────────────────────────────────────────

def run_generator(vuln_type, difficulty, context, xss_subtype):
    if not os.getenv("HF_TOKEN", ""):
        return "❌ HF_TOKEN not configured. Add it as a Space secret in HuggingFace Settings.", "{}"
    subtype = xss_subtype if vuln_type == "XSS" else ""
    challenge = generate_challenge(vuln_type, difficulty, context, subtype)
    markdown = format_challenge_markdown(challenge)
    json_out = json.dumps(challenge, indent=2)
    return markdown, json_out


def update_subtype_visibility(vuln_type):
    return gr.update(visible=(vuln_type == "XSS"))


# ── Statistical Tests ──────────────────────────────────────────────────────────

def run_statistical_tests(df):
    """Run Wilcoxon signed-rank and Mann-Whitney U tests, return markdown summary."""
    results = []

    # 1. Wilcoxon signed-rank: pre vs post (are gains significant?)
    try:
        w_stat, p_val = scipy_stats.wilcoxon(df["post_score"], df["pre_score"])
        n = len(df)
        # Effect size r = Z / sqrt(n)
        z_score = scipy_stats.norm.ppf(p_val / 2)
        r = abs(z_score) / np.sqrt(n)
        r_label = "large" if r >= 0.5 else "medium" if r >= 0.3 else "small"
        sig = "✅ Significant" if p_val < 0.05 else "❌ Not significant"

        results.append(f"""### 📐 Wilcoxon Signed-Rank Test (Pre vs Post)
*Are the learning gains statistically significant?*

| Metric | Value |
|--------|-------|
| W statistic | {round(w_stat, 3)} |
| p-value | {round(p_val, 4)} |
| Effect size r | {round(r, 3)} ({r_label}) |
| Result | {sig} (α = 0.05) |

> **Interpretation:** {"The learning gains are statistically significant — the game had a real effect on student knowledge." if p_val < 0.05 else "The learning gains are not statistically significant at p < 0.05. Consider a larger sample."}
""")
    except Exception as e:
        results.append(f"⚠️ Wilcoxon test failed: {e}\n")

    # 2. Mann-Whitney U: simulator vs non-simulator gains
    sim_gains     = df[df["used_simulator"] == 1]["learning_gain"]
    non_sim_gains = df[df["used_simulator"] == 0]["learning_gain"]

    if len(sim_gains) >= 3 and len(non_sim_gains) >= 3:
        try:
            u_stat, p_val2 = scipy_stats.mannwhitneyu(sim_gains, non_sim_gains, alternative="two-sided")
            n1, n2 = len(sim_gains), len(non_sim_gains)
            r2 = abs(scipy_stats.norm.ppf(p_val2 / 2)) / np.sqrt(n1 + n2)
            r2_label = "large" if r2 >= 0.5 else "medium" if r2 >= 0.3 else "small"
            sig2 = "✅ Significant" if p_val2 < 0.05 else "❌ Not significant"

            results.append(f"""### 📐 Mann-Whitney U Test (Simulator vs No Simulator)
*Does simulator usage significantly affect learning gain?*

| Metric | Value |
|--------|-------|
| U statistic | {round(u_stat, 3)} |
| p-value | {round(p_val2, 4)} |
| Effect size r | {round(r2, 3)} ({r2_label}) |
| Sim group (n) | {n1} |
| No-sim group (n) | {n2} |
| Result | {sig2} (α = 0.05) |

> **Interpretation:** {"Simulator usage is associated with significantly different learning gains." if p_val2 < 0.05 else "No statistically significant difference in learning gains between simulator and non-simulator groups."}
""")
        except Exception as e:
            results.append(f"⚠️ Mann-Whitney test failed: {e}\n")
    else:
        results.append("⚠️ *Mann-Whitney U test skipped — insufficient data in one or both groups (need n ≥ 3 each).*\n")

    # 3. V1 vs V2 comparison (if both present)
    if df["version"].nunique() > 1:
        v1_gains = df[df["version"] == "V1"]["learning_gain"]
        v2_gains = df[df["version"] == "V2"]["learning_gain"]
        if len(v1_gains) >= 3 and len(v2_gains) >= 3:
            try:
                u3, p3 = scipy_stats.mannwhitneyu(v1_gains, v2_gains, alternative="two-sided")
                r3 = abs(scipy_stats.norm.ppf(p3 / 2)) / np.sqrt(len(v1_gains) + len(v2_gains))
                r3_label = "large" if r3 >= 0.5 else "medium" if r3 >= 0.3 else "small"
                sig3 = "✅ Significant" if p3 < 0.05 else "❌ Not significant"

                results.append(f"""### 📐 Mann-Whitney U Test (V1 vs V2)
*Did the redesigned game version produce significantly different learning gains?*

| Metric | Value |
|--------|-------|
| U statistic | {round(u3, 3)} |
| p-value | {round(p3, 4)} |
| Effect size r | {round(r3, 3)} ({r3_label}) |
| V1 (n) | {len(v1_gains)} |
| V2 (n) | {len(v2_gains)} |
| Result | {sig3} (α = 0.05) |

> **Interpretation:** {"V2 produced significantly different learning gains compared to V1." if p3 < 0.05 else "No statistically significant difference in learning gains between V1 and V2."}
""")
            except Exception as e:
                results.append(f"⚠️ V1 vs V2 test failed: {e}\n")

    return "\n---\n".join(results)


# ── Tab 2: Learning Analytics ──────────────────────────────────────────────────

def run_analytics(file, version_filter):
    if file is None:
        return "Please upload a CSV file.", "", None, None, None, None

    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"❌ Could not read CSV: {e}", "", None, None, None, None

    try:
        df, warnings = load_and_validate(df)
    except ValueError as e:
        return f"❌ {e}\n\nExpected columns: pre_score, post_score, used_simulator (0/1), time_in_simulator (minutes), version (V1/V2)", "", None, None, None, None

    if version_filter != "All":
        df = df[df["version"] == version_filter]
        if len(df) == 0:
            return f"No data found for version {version_filter}.", "", None, None, None, None

    df = segment_students(df)
    stats = compute_summary_stats(df)

    # Summary text
    warn_text = "\n".join([f"⚠️ {w}" for w in warnings]) + "\n\n" if warnings else ""

    # Hake's g interpretation
    g = stats['avg_normalised_gain']
    g_label = "High" if g >= 0.7 else "Medium" if g >= 0.3 else "Low"

    summary = f"""{warn_text}## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| Students | {stats['n_students']} |
| Avg Pre-Score | {stats['pre_mean']} |
| Avg Post-Score | {stats['post_mean']} |
| Avg Learning Gain | +{stats['avg_learning_gain']} |
| Normalised Gain (Hake's g) | {g} → **{g_label} gain** |
| % Students Improved | {stats['pct_improved']}% |
| % Used Simulator | {stats['pct_used_simulator']}% |

## 🎮 Simulator Impact
| Group | n | Avg Gain | Hake's g |
|-------|---|----------|---------|
| Used Simulator | {stats.get('simulator_users', 0)} | {stats.get('sim_avg_gain', 'N/A')} | {stats.get('sim_avg_norm_gain', 'N/A')} |
| No Simulator | {stats.get('non_simulator_users', 0)} | {stats.get('non_sim_avg_gain', 'N/A')} | {stats.get('non_sim_avg_norm_gain', 'N/A')} |
"""

    if "version_comparison" in stats:
        summary += "\n## 📋 Version Comparison\n"
        vc = stats["version_comparison"]
        for version in vc.get("n", {}).keys():
            summary += f"\n**{version}:** n={vc['n'][version]}, Pre={round(vc['pre_mean'][version],1)}, Post={round(vc['post_mean'][version],1)}, Gain={round(vc['avg_gain'][version],2)}, g={round(vc['avg_norm_gain'][version],3)}\n"

    # Statistical tests
    stats_text = run_statistical_tests(df)

    # Plot 1: Pre vs Post score distribution — use violin+strip for reliability
    fig1 = go.Figure()
    fig1.add_trace(go.Box(
        y=df["pre_score"].tolist(), x=["Pre-Score"] * len(df),
        name="Pre-Score", marker_color="#636EFA",
        boxpoints="all", jitter=0.4, pointpos=0
    ))
    fig1.add_trace(go.Box(
        y=df["post_score"].tolist(), x=["Post-Score"] * len(df),
        name="Post-Score", marker_color="#00CC96",
        boxpoints="all", jitter=0.4, pointpos=0
    ))
    fig1.update_layout(
        title="Pre vs Post Score Distribution",
        yaxis_title="Score",
        xaxis_title="",
        template="plotly_white",
        yaxis=dict(range=[0, 110]),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Plot 2: Learning gain by simulator usage
    plot_df2 = df.copy()
    plot_df2["Group"] = plot_df2["used_simulator"].map({1: "Used Simulator", 0: "No Simulator"})
    fig2 = px.box(
        plot_df2, x="Group", y="learning_gain", color="Group",
        title="Learning Gain: Simulator vs Non-Simulator",
        labels={"learning_gain": "Learning Gain", "Group": ""},
        template="plotly_white",
        color_discrete_map={"Used Simulator": "#00CC96", "No Simulator": "#EF553B"},
        points="all"
    )
    fig2.update_layout(showlegend=False)

    # Plot 3: Student segments — ensure all 4 segments always shown
    all_segments = [
        "High Prior / High Gain",
        "High Prior / Low Gain",
        "Low Prior / High Gain ⭐",
        "Low Prior / Low Gain ⚠️",
    ]
    seg_series = df["segment"].value_counts()
    segment_counts = pd.DataFrame({
        "Segment": all_segments,
        "Count": [int(seg_series.get(s, 0)) for s in all_segments]
    })
    colors = ["#00CC96", "#636EFA", "#FFA15A", "#EF553B"]
    fig3 = go.Figure()
    for i, row in segment_counts.iterrows():
        fig3.add_trace(go.Bar(
            x=[row["Segment"]], y=[row["Count"]],
            name=row["Segment"],
            marker_color=colors[i],
            text=[str(row["Count"])],
            textposition="outside",
        ))
    fig3.update_layout(
        title="Student Learning Profiles",
        xaxis_title="Segment",
        yaxis_title="Count",
        template="plotly_white",
        showlegend=False,
        xaxis_tickangle=-15,
        yaxis=dict(range=[0, segment_counts["Count"].max() * 1.3 + 1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        barmode="group",
    )

    # Plot 4: V1 vs V2 or pre/post scatter
    if df["version"].nunique() > 1:
        version_df = df.groupby("version").agg(
            pre_mean=("pre_score", "mean"),
            post_mean=("post_score", "mean"),
            avg_gain=("learning_gain", "mean"),
            hakes_g=("normalised_gain", "mean")
        ).reset_index()
        melted = version_df.melt(id_vars="version", value_vars=["pre_mean", "post_mean", "avg_gain"])
        melted["label"] = melted["value"].round(1).astype(str)
        fig4 = go.Figure()
        metric_colors = {"pre_mean": "#636EFA", "post_mean": "#00CC96", "avg_gain": "#FFA15A"}
        metric_names = {"pre_mean": "Pre Mean", "post_mean": "Post Mean", "avg_gain": "Avg Gain"}
        for metric in ["pre_mean", "post_mean", "avg_gain"]:
            sub = melted[melted["variable"] == metric]
            fig4.add_trace(go.Bar(
                x=sub["version"].tolist(),
                y=sub["value"].tolist(),
                name=metric_names[metric],
                marker_color=metric_colors[metric],
                text=[f"{v:.1f}" for v in sub["value"].tolist()],
                textposition="outside",
            ))
        fig4.update_layout(
            title="V1 vs V2 Comparison",
            xaxis_title="Version",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(range=[0, version_df[["pre_mean","post_mean","avg_gain"]].max().max() * 1.2]),
        )
    else:
        plot_df4 = df.copy()
        plot_df4["Group"] = plot_df4["used_simulator"].map({1: "Used Simulator", 0: "No Simulator"})
        fig4 = px.scatter(
            plot_df4, x="pre_score", y="post_score", color="Group",
            title="Pre vs Post by Simulator Usage",
            labels={"pre_score": "Pre-Score", "post_score": "Post-Score"},
            template="plotly_white",
        )
        # Add diagonal reference line (no gain)
        max_val = max(df["pre_score"].max(), df["post_score"].max()) + 5
        fig4.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                       line=dict(color="gray", dash="dash"))

    return summary, stats_text, fig1, fig2, fig3, fig4


# ── Build UI ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="XSS CTF Intelligence Platform", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🔐 XSS CTF Intelligence Platform
**Two tools in one:** Generate CTF security challenges with AI · Analyse student learning patterns

> Built by [Nipuna Weeratunge](https://www.linkedin.com/in/nipunaw) |
> [GitHub](https://github.com/darkcyberwizard/xss-ctf-intelligence)
""")

    with gr.Tabs():

        # ── Tab 1: Generator ───────────────────────────────────────────────
        with gr.TabItem("🚩 CTF Challenge Generator"):
            gr.Markdown("Generate structured CTF challenges for cybersecurity education using LLM.")
            with gr.Row():
                with gr.Column(scale=1):
                    vuln_type   = gr.Dropdown(VULN_TYPES,   label="Vulnerability Type", value="XSS")
                    xss_subtype = gr.Dropdown(XSS_SUBTYPES, label="XSS Subtype", value="Reflected XSS", visible=True)
                    difficulty  = gr.Dropdown(DIFFICULTIES, label="Difficulty", value="Medium")
                    context     = gr.Dropdown(CONTEXTS,     label="Context",    value="Login page")
                    gen_btn     = gr.Button("⚡ Generate Challenge", variant="primary")
                with gr.Column(scale=2):
                    challenge_md   = gr.Markdown(label="Generated Challenge")
                    challenge_json = gr.Textbox(label="Raw JSON", lines=10)

            vuln_type.change(fn=update_subtype_visibility, inputs=vuln_type, outputs=xss_subtype)
            gen_btn.click(
                fn=run_generator,
                inputs=[vuln_type, difficulty, context, xss_subtype],
                outputs=[challenge_md, challenge_json]
            )

        # ── Tab 2: Analytics ───────────────────────────────────────────────
        with gr.TabItem("📊 Learning Analytics"):
            gr.Markdown("""
Analyse student learning outcomes from pre/post test scores and simulator engagement data.

**Upload a CSV with columns:** `pre_score, post_score, used_simulator (0/1), time_in_simulator (minutes), version (V1/V2)`
""")
            with gr.Row():
                file_upload    = gr.File(label="Upload Student Data CSV", file_types=[".csv"])
                version_filter = gr.Dropdown(["All", "V1", "V2"], label="Filter by Version", value="All")
                analyse_btn    = gr.Button("📊 Analyse", variant="primary")

            summary_out = gr.Markdown(label="Summary Statistics")
            stats_out   = gr.Markdown(label="Statistical Tests")

            with gr.Row():
                plot1 = gr.Plot(label="Score Distribution")
                plot2 = gr.Plot(label="Simulator Impact")
            with gr.Row():
                plot3 = gr.Plot(label="Student Segments")
                plot4 = gr.Plot(label="Version Comparison / Scatter")

            analyse_btn.click(
                fn=run_analytics,
                inputs=[file_upload, version_filter],
                outputs=[summary_out, stats_out, plot1, plot2, plot3, plot4]
            )

            gr.Markdown("""
---
### Metrics explained
- **Learning Gain** = Post − Pre score
- **Hake's g** = (Post − Pre) / (Max − Pre) — controls for prior knowledge ceiling
- **Wilcoxon signed-rank** — tests whether pre/post gains are statistically significant (non-parametric)
- **Mann-Whitney U** — compares two independent groups (simulator vs none, V1 vs V2)
- **Effect size r** = Z / √n — practical magnitude of the effect (small < 0.3, medium 0.3–0.5, large ≥ 0.5)
""")

    gr.Markdown("---\n*XSS CTF Intelligence Platform — Cybersecurity Education Research*")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
