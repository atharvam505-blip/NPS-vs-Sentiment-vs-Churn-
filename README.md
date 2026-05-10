# NPS-vs-Sentiment-vs-Churn-# B2B Ticketing Software — Customer Satisfaction Research
### NPS · Sentiment Analysis · Churn Risk Scoring

> An independent market research study analyzing **4,852 Capterra reviews** across 6 enterprise ticketing tools —
> applying NPS methodology, VADER sentiment analysis, and a composite churn risk framework
> to surface actionable insights for B2B software vendors.

---

## Tools Covered

| Tool | Reviews |
|---|---|
| Zoho Desk | 1,393 |
| Freshdesk | 1,025 |
| Zendesk | 1,475 |
| ServiceNow | 221 |
| Jira Service Management | 649 |
| OTRS | 89 |

---

## Project Structure

```
├── capterra_analysis.py     # Main analysis script
├── capterra_reviews.csv     # Dataset (Capterra via Kaggle)
├── fig1_nps.png             # NPS by tool
├── fig2_sentiment_violin.png# Sentiment distribution
├── fig3_churn_risk.png      # Churn risk segmentation
├── fig4_feature_heatmap.png # Feature strength heatmap
├── fig5_bubble.png          # NPS vs Sentiment vs Churn bubble chart
├── fig6_wordclouds.png      # Voice of Customer word clouds
└── README.md
```

---

## Methodology

### 1. Net Promoter Score (NPS)
Derived from the `likelihood_to_recommend` field (1–10 scale):

```
Promoters  → score ≥ 9
Passives   → score 7–8
Detractors → score ≤ 6

NPS = %Promoters − %Detractors
```

NPS is the industry-standard metric used by market research firms to measure customer loyalty and brand advocacy.

---

### 2. Sentiment Analysis (VADER)
Applied **VADER (Valence Aware Dictionary and sEntiment Reasoner)** to the `overall_text` review column.

- Compound score range: **−1 (very negative) → +1 (very positive)**
- Threshold: score ≥ 0.05 = Positive | ≤ −0.05 = Negative | else = Neutral
- VADER is chosen over ML models because it is lexicon-based, requires no training data, and performs well on short user-generated text

---

### 3. Churn Risk Scoring
A composite index built from three normalized signals:

| Component | Source Column | Direction |
|---|---|---|
| Rating Risk | `overall_rating` (1–5) | Low rating → High risk |
| Recommend Risk | `likelihood_to_recommend` (1–10) | Low score → High risk |
| Sentiment Risk | VADER compound score | Negative → High risk |

Each component normalized to [0, 1], then averaged:

```
Churn Risk = mean(rating_risk, recommend_risk, sentiment_risk)

High Risk   → score ≥ 0.65
Medium Risk → score 0.40–0.64
Low Risk    → score < 0.40
```

---

### 4. Feature-Level Satisfaction
Binary feature mention scores extracted from dataset columns:

```
+1 = Feature mentioned as a strength
 0 = Not mentioned
−1 = Feature mentioned as a weakness
```

Features analyzed: Ticket Creation, Automated Routing, SLA Management,
Knowledge Base, Reporting & Analytics, Multi-Channel Support, and more.

---

## Key Findings

### NPS Rankings
| Tool | NPS | Promoters% | Detractors% |
|---|---|---|---|
| **Freshdesk** | **59.0** | 65.2% | 6.1% |
| ServiceNow | 51.4 | 58.7% | 7.3% |
| Zoho Desk | 49.9 | 57.5% | 7.6% |
| Jira Service Mgmt | 47.8 | 56.5% | 8.6% |
| Zendesk | 47.1 | 56.2% | 9.1% |
| OTRS | 45.2 | 59.5% | 14.3% |

**Overall NPS across all tools: 50.8**

---

### Sentiment Leaders
- **Freshdesk** scores highest on sentiment (0.627) — reviews use the most positive language
- **OTRS** and **Jira** trail with scores below 0.50
- Across all tools, sentiment is predominantly positive — suggesting general satisfaction but with notable pain points

---

### Churn Risk Insights
- **OTRS** carries the highest churn risk (2.25% High Risk segment)
- **Zoho Desk** retains customers most effectively (0.22% High Risk)
- **Freshdesk** combines low churn risk with highest NPS — strongest overall position

---

### Feature Insights
- **Automated Ticket Routing** is a weakness across ALL 6 tools (negative scores universally)
- **Ticket Creation & Assignment** is the strongest feature across the board
- **Reporting & Analytics** is a consistent gap — an opportunity for vendors to differentiate

---

## Visualizations

### Fig 1 — NPS by Tool
![NPS Chart](fig1_nps.png)

### Fig 2 — Sentiment Distribution
![Sentiment Violin](fig2_sentiment_violin.png)

### Fig 3 — Churn Risk Segmentation
![Churn Risk](fig3_churn_risk.png)

### Fig 4 — Feature Strength Heatmap
![Feature Heatmap](fig4_feature_heatmap.png)

### Fig 5 — NPS vs Sentiment vs Churn (Bubble Chart)
![Bubble Chart](fig5_bubble.png)

### Fig 6 — Voice of Customer Word Clouds
![Word Clouds](fig6_wordclouds.png)

---

## How to Run

**Requirements**
```bash
pip install pandas numpy matplotlib seaborn vaderSentiment wordcloud
```

**Run**
```bash
python capterra_analysis.py
```

Place `capterra_reviews.csv` in the same directory. All 6 charts are saved automatically as PNG files.

**Dataset source:** [Capterra Reviews — Kaggle](https://www.kaggle.com/datasets/tobiasbueck/capterra-reviews)

---

## Skills Demonstrated

- Market research methodology (NPS framework)
- Natural Language Processing (VADER sentiment analysis)
- Composite index design (churn risk scoring)
- Voice of Customer (VoC) research
- Data visualization and insight communication
- B2B SaaS competitive analysis

---

