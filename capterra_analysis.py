# =============================================================================
# B2B Ticketing Software: NPS, Sentiment & Churn Risk Analysis
# Dataset: Capterra Reviews (Zoho Desk, Freshdesk, Zendesk,
#          ServiceNow, Jira Service Management, OTRS)
# Author: Atharva Mishra
# =============================================================================

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

# ── PLOT STYLE ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "axes.edgecolor":   "#2e2e3a",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#a0a0b0",
    "ytick.color":      "#a0a0b0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2e2e3a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "figure.dpi":       130,
})

PALETTE  = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ce93d8", "#80cbc4"]
TOOLS    = ["Zoho Desk", "Freshdesk", "Zendesk",
            "ServiceNow", "Jira Service Management", "otrs"]
COLOR_MAP = dict(zip(TOOLS, PALETTE))


# =============================================================================
# 1. LOAD & CLEAN
# =============================================================================
print("=" * 60)
print("  B2B Ticketing Software — Research Analysis")
print("=" * 60)

df = pd.read_csv("capterra_reviews.csv")
print(f"\n[INFO] Raw records loaded : {len(df):,}")

# Drop rows with no text at all
df = df.dropna(subset=["overall_text", "overall_rating"])
df["overall_rating"]         = pd.to_numeric(df["overall_rating"],         errors="coerce")
df["likelihood_to_recommend"]= pd.to_numeric(df["likelihood_to_recommend"],errors="coerce")
df["ease_of_use"]            = pd.to_numeric(df["ease_of_use"],            errors="coerce")
df["customer_service"]       = pd.to_numeric(df["customer_service"],       errors="coerce")
df["features"]               = pd.to_numeric(df["features"],               errors="coerce")
df["value_for_money"]        = pd.to_numeric(df["value_for_money"],        errors="coerce")

df = df.dropna(subset=["overall_rating"])
print(f"[INFO] Records after cleaning : {len(df):,}")
print(f"[INFO] Tools covered          : {df['ticket_system'].nunique()}")
print(f"[INFO] Columns                : {df.shape[1]}\n")


# =============================================================================
# 2. NPS ANALYSIS
# =============================================================================
print("─" * 60)
print("  SECTION 1 — NET PROMOTER SCORE (NPS)")
print("─" * 60)

nps_df = df.dropna(subset=["likelihood_to_recommend"]).copy()

def nps_segment(score):
    if score >= 9:  return "Promoter"
    if score >= 7:  return "Passive"
    return "Detractor"

nps_df["nps_segment"] = nps_df["likelihood_to_recommend"].apply(nps_segment)

# Overall NPS
counts   = nps_df["nps_segment"].value_counts()
total    = len(nps_df)
pct_p    = counts.get("Promoter",  0) / total * 100
pct_d    = counts.get("Detractor", 0) / total * 100
overall_nps = round(pct_p - pct_d, 1)
print(f"\n  Overall NPS Score : {overall_nps}")
print(f"  Promoters  : {pct_p:.1f}%")
print(f"  Passives   : {counts.get('Passive',0)/total*100:.1f}%")
print(f"  Detractors : {pct_d:.1f}%\n")

# Per-tool NPS
tool_nps = []
for tool in TOOLS:
    sub = nps_df[nps_df["ticket_system"] == tool]
    if len(sub) == 0:
        continue
    p  = (sub["nps_segment"] == "Promoter").sum()  / len(sub) * 100
    d  = (sub["nps_segment"] == "Detractor").sum() / len(sub) * 100
    tool_nps.append({"Tool": tool, "NPS": round(p - d, 1),
                     "Promoters%": round(p, 1), "Detractors%": round(d, 1),
                     "Count": len(sub)})
nps_summary = pd.DataFrame(tool_nps).sort_values("NPS", ascending=False)
print(nps_summary.to_string(index=False))


# =============================================================================
# 3. SENTIMENT ANALYSIS
# =============================================================================
print("\n" + "─" * 60)
print("  SECTION 2 — SENTIMENT ANALYSIS (VADER)")
print("─" * 60)

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if pd.isna(text) or str(text).strip() == "":
        return np.nan
    return analyzer.polarity_scores(str(text))["compound"]

df["sentiment_overall"] = df["overall_text"].apply(get_sentiment)
df["sentiment_pros"]    = df["pros_text"].apply(get_sentiment)
df["sentiment_cons"]    = df["cons_text"].apply(get_sentiment)

def label_sentiment(score):
    if pd.isna(score): return "Neutral"
    if score >= 0.05:  return "Positive"
    if score <= -0.05: return "Negative"
    return "Neutral"

df["sentiment_label"] = df["sentiment_overall"].apply(label_sentiment)

sent_summary = df.groupby("ticket_system")["sentiment_overall"].agg(["mean","std","count"])
sent_summary.columns = ["Mean Sentiment", "Std Dev", "Reviews"]
sent_summary = sent_summary.sort_values("Mean Sentiment", ascending=False).round(3)
print(f"\n  Sentiment Score: -1 (very negative) → +1 (very positive)\n")
print(sent_summary.to_string())

overall_sent = df.groupby(["ticket_system","sentiment_label"]).size().unstack(fill_value=0)
print("\n  Sentiment Breakdown by Tool:")
print(overall_sent.to_string())


# =============================================================================
# 4. CHURN RISK ANALYSIS
# =============================================================================
print("\n" + "─" * 60)
print("  SECTION 3 — CHURN RISK SCORING")
print("─" * 60)

# Churn risk = composite of low rating + low recommendation + negative sentiment
# Each component normalized 0-1, then averaged → churn_risk (0=safe, 1=high risk)

churn_df = df.copy()

# Normalize overall_rating (1–5) → risk: rating=1 means high risk
churn_df["rating_risk"] = (5 - churn_df["overall_rating"]) / 4

# Normalize likelihood_to_recommend (1–10) → risk
churn_df["recommend_risk"] = churn_df["likelihood_to_recommend"].apply(
    lambda x: (10 - x) / 9 if not pd.isna(x) else np.nan
)

# Sentiment risk: compound score in [-1,1] → risk in [0,1]
churn_df["sentiment_risk"] = churn_df["sentiment_overall"].apply(
    lambda x: (1 - x) / 2 if not pd.isna(x) else np.nan
)

# Composite (weighted average, drop NaN columns per row)
churn_df["churn_risk"] = churn_df[
    ["rating_risk", "recommend_risk", "sentiment_risk"]
].mean(axis=1, skipna=True)

# Segment
def churn_segment(score):
    if score >= 0.65: return "High Risk"
    if score >= 0.40: return "Medium Risk"
    return "Low Risk"

churn_df["churn_segment"] = churn_df["churn_risk"].apply(churn_segment)

churn_summary = (
    churn_df.groupby("ticket_system")
    .agg(
        Avg_Churn_Risk=("churn_risk","mean"),
        High_Risk_Pct=("churn_segment", lambda x: (x=="High Risk").sum()/len(x)*100),
        Medium_Risk_Pct=("churn_segment", lambda x: (x=="Medium Risk").sum()/len(x)*100),
        Low_Risk_Pct=("churn_segment", lambda x: (x=="Low Risk").sum()/len(x)*100),
    )
    .sort_values("Avg_Churn_Risk", ascending=False)
    .round(2)
)
print("\n  Churn Risk by Tool (Higher = More Likely to Switch):\n")
print(churn_summary.to_string())


# =============================================================================
# 5. FEATURE-LEVEL SATISFACTION
# =============================================================================
print("\n" + "─" * 60)
print("  SECTION 4 — FEATURE-LEVEL SATISFACTION")
print("─" * 60)

feature_cols = [
    "Ticket Creation and Assignment", "Automated Ticket Routing",
    "Status Tracking and Updates",    "Priority and SLA Management",
    "Customer and Agent Portals",     "Knowledge Base Integration",
    "Email Notifications and Alerts", "Reporting and Analytics",
    "Customizable Workflows",         "Multi-Channel Support (Email, Chat, Phone)"
]
# These cols are binary: 1=mentioned as strength, -1=mentioned as weakness, 0=not mentioned
feat_df = df[["ticket_system"] + feature_cols].copy()
feat_means = feat_df.groupby("ticket_system")[feature_cols].mean().round(3)
print("\n  Feature Strength Score (1=Strength, -1=Weakness, 0=Neutral):\n")
print(feat_means.to_string())


# =============================================================================
# 6. VISUALISATIONS
# =============================================================================
print("\n" + "─" * 60)
print("  SECTION 5 — GENERATING CHARTS")
print("─" * 60)

# ── FIG 1: NPS Bar Chart ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors_nps = [COLOR_MAP.get(t, "#888") for t in nps_summary["Tool"]]
bars = ax.barh(nps_summary["Tool"], nps_summary["NPS"],
               color=colors_nps, edgecolor="none", height=0.55)
ax.axvline(0, color="#555", linewidth=1)
for bar, val in zip(bars, nps_summary["NPS"]):
    ax.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
            f"{val}", va="center", ha="left" if val >= 0 else "right",
            fontsize=10, color="#e0e0e0")
ax.set_xlabel("Net Promoter Score", fontsize=11)
ax.set_title("NPS by Ticketing Tool — Capterra Reviews", fontsize=13, pad=12)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("fig1_nps.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  [SAVED] fig1_nps.png")

# ── FIG 2: Sentiment Distribution (violin) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
tool_order = df.groupby("ticket_system")["sentiment_overall"].median().sort_values(ascending=False).index.tolist()
parts = ax.violinplot(
    [df[df["ticket_system"]==t]["sentiment_overall"].dropna().values for t in tool_order],
    positions=range(len(tool_order)), showmedians=True, showextrema=False
)
for i, (pc, tool) in enumerate(zip(parts["bodies"], tool_order)):
    pc.set_facecolor(COLOR_MAP.get(tool, "#888"))
    pc.set_alpha(0.75)
parts["cmedians"].set_color("#ffffff")
parts["cmedians"].set_linewidth(2)
ax.set_xticks(range(len(tool_order)))
ax.set_xticklabels(tool_order, rotation=15, ha="right", fontsize=10)
ax.set_ylabel("VADER Compound Score", fontsize=11)
ax.set_title("Sentiment Score Distribution by Tool", fontsize=13, pad=12)
ax.axhline(0, color="#555", linewidth=1, linestyle="--")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_sentiment_violin.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  [SAVED] fig2_sentiment_violin.png")

# ── FIG 3: Churn Risk Stacked Bar ─────────────────────────────────────────────
churn_pcts = (
    churn_df.groupby(["ticket_system","churn_segment"])
    .size().unstack(fill_value=0)
)
churn_pcts = churn_pcts.div(churn_pcts.sum(axis=1), axis=0) * 100
churn_pcts = churn_pcts.reindex(columns=["Low Risk","Medium Risk","High Risk"])
churn_pcts = churn_pcts.loc[churn_pcts["High Risk"].sort_values(ascending=False).index]

seg_colors = {"Low Risk": "#81c784", "Medium Risk": "#ffb74d", "High Risk": "#e57373"}
fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(churn_pcts))
for seg in ["Low Risk", "Medium Risk", "High Risk"]:
    vals = churn_pcts[seg].values
    ax.bar(churn_pcts.index, vals, bottom=bottom,
           label=seg, color=seg_colors[seg], edgecolor="#0f1117", linewidth=0.5)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 5:
            ax.text(i, b + v/2, f"{v:.0f}%", ha="center", va="center",
                    fontsize=9, color="#0f1117", fontweight="bold")
    bottom += vals
ax.set_ylabel("Percentage of Reviews (%)", fontsize=11)
ax.set_title("Churn Risk Segmentation by Tool", fontsize=13, pad=12)
ax.legend(loc="upper right", framealpha=0.2)
ax.set_xticklabels(churn_pcts.index, rotation=15, ha="right", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig3_churn_risk.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  [SAVED] fig3_churn_risk.png")

# ── FIG 4: Feature Heatmap ────────────────────────────────────────────────────
short_names = {
    "Ticket Creation and Assignment":           "Ticket Creation",
    "Automated Ticket Routing":                 "Auto Routing",
    "Status Tracking and Updates":              "Status Tracking",
    "Priority and SLA Management":              "SLA Mgmt",
    "Customer and Agent Portals":               "Portals",
    "Knowledge Base Integration":               "Knowledge Base",
    "Email Notifications and Alerts":           "Email Alerts",
    "Reporting and Analytics":                  "Reporting",
    "Customizable Workflows":                   "Workflows",
    "Multi-Channel Support (Email, Chat, Phone)":"Multi-Channel",
}
heatmap_data = feat_means.rename(columns=short_names)
fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, linecolor="#1a1a2e",
            ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Feature Strength Heatmap by Tool\n(+1 = Strength  |  0 = Neutral  |  -1 = Weakness)",
             fontsize=12, pad=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("fig4_feature_heatmap.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  [SAVED] fig4_feature_heatmap.png")

# ── FIG 5: NPS vs Sentiment vs Churn Risk — Bubble Chart ────────────────────
merged = nps_summary.merge(
    sent_summary.reset_index().rename(columns={"ticket_system":"Tool","Mean Sentiment":"Sentiment"}),
    on="Tool"
).merge(
    churn_summary.reset_index().rename(columns={"ticket_system":"Tool","Avg_Churn_Risk":"ChurnRisk"}),
    on="Tool"
)

fig, ax = plt.subplots(figsize=(9, 6))
for _, row in merged.iterrows():
    color = COLOR_MAP.get(row["Tool"], "#888")
    ax.scatter(row["NPS"], row["Sentiment"],
               s=row["ChurnRisk"] * 3000,
               color=color, alpha=0.7, edgecolors="#fff", linewidths=0.8)
    ax.annotate(row["Tool"], (row["NPS"], row["Sentiment"]),
                textcoords="offset points", xytext=(8, 4), fontsize=9)

ax.set_xlabel("NPS Score →  Higher is Better", fontsize=11)
ax.set_ylabel("Avg Sentiment Score →  Higher is Better", fontsize=11)
ax.set_title("NPS vs Sentiment vs Churn Risk\n(Bubble Size = Churn Risk — Larger = More At-Risk)",
             fontsize=12, pad=12)
ax.axvline(0, color="#555", linewidth=1, linestyle="--")
ax.axhline(0, color="#555", linewidth=1, linestyle="--")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig5_bubble.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  [SAVED] fig5_bubble.png")

# ── FIG 6: Word Clouds (Pros vs Cons) ────────────────────────────────────────
pros_text = " ".join(df["pros_text"].dropna().astype(str).tolist())
cons_text = " ".join(df["cons_text"].dropna().astype(str).tolist())

stop_extras = {"use","easy","also","one","get","well","can","need",
               "make","good","really","much","many","great","very",
               "system","software","tool","product","solution","using",
               "used","would","like","time","way","even","lot","best"}

wc_pros = WordCloud(width=800, height=400, background_color="#0f1117",
                    colormap="YlGn", stopwords=stop_extras,
                    max_words=80, collocations=False).generate(pros_text)

wc_cons = WordCloud(width=800, height=400, background_color="#0f1117",
                    colormap="OrRd", stopwords=stop_extras,
                    max_words=80, collocations=False).generate(cons_text)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.imshow(wc_pros, interpolation="bilinear"); ax1.axis("off")
ax1.set_title("What Users Love (Pros)", fontsize=13, pad=10, color="#81c784")
ax2.imshow(wc_cons, interpolation="bilinear"); ax2.axis("off")
ax2.set_title("What Users Dislike (Cons)", fontsize=13, pad=10, color="#e57373")
fig.suptitle("Voice of Customer — Word Clouds", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("fig6_wordclouds.png", bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("  [SAVED] fig6_wordclouds.png")


# =============================================================================
# 7. FINAL SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 60)
print("  FINAL SUMMARY — KEY METRICS BY TOOL")
print("=" * 60)

final = (
    nps_summary[["Tool","NPS"]]
    .merge(sent_summary.reset_index().rename(columns={"ticket_system":"Tool",
                                                       "Mean Sentiment":"Avg Sentiment"})[["Tool","Avg Sentiment"]], on="Tool")
    .merge(churn_summary.reset_index().rename(columns={"ticket_system":"Tool"})[["Tool","Avg_Churn_Risk","High_Risk_Pct"]], on="Tool")
    .merge(df.groupby("ticket_system")["overall_rating"].mean().reset_index().rename(
           columns={"ticket_system":"Tool","overall_rating":"Avg Rating"}), on="Tool")
)
final = final.sort_values("NPS", ascending=False).round(2)
print(f"\n{final.to_string(index=False)}")

print("\n" + "=" * 60)
print("  KEY INSIGHTS")
print("=" * 60)

best_nps   = final.iloc[0]["Tool"]
worst_nps  = final.iloc[-1]["Tool"]
best_sent  = final.loc[final["Avg Sentiment"].idxmax(), "Tool"]
riskiest   = final.loc[final["High_Risk_Pct"].idxmax(), "Tool"]
safest     = final.loc[final["High_Risk_Pct"].idxmin(), "Tool"]

print(f"""
  1. NPS LEADER      : {best_nps} has the highest NPS — strongest customer advocacy.
  2. NPS LAGGARD     : {worst_nps} has the lowest NPS — users least likely to recommend.
  3. SENTIMENT LEADER: {best_sent} has the most positive review language overall.
  4. HIGHEST CHURN   : {riskiest} carries the highest churn risk — customers most
                       likely to switch to a competitor.
  5. LOWEST CHURN    : {safest} retains customers most effectively.

  METHODOLOGY NOTES:
  ─────────────────
  • NPS computed from 'likelihood_to_recommend' (1–10 scale):
      Promoters ≥ 9 | Passives 7–8 | Detractors ≤ 6
      NPS = %Promoters − %Detractors

  • Sentiment scored using VADER compound score (−1 to +1)
      on 'overall_text' column.

  • Churn Risk = weighted composite of:
      (a) Low overall rating
      (b) Low likelihood to recommend
      (c) Negative sentiment score
    Normalized 0–1 per component; averaged across available columns.

  • Feature scores are binary mentions in reviews:
      +1 = mentioned as strength | −1 = mentioned as weakness | 0 = neutral

  Dataset: Capterra Reviews | N = {len(df):,} reviews | 6 B2B Ticketing Tools
""")

print("  Analysis complete. All figures saved as PNG files.")
print("=" * 60)
