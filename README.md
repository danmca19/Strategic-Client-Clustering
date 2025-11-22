# 📈 Customer Segmentation: Unlocking 5 Actionable RetailX Client Profiles

This project builds a customer segmentation pipeline for RetailX using clustering to support marketing, retention, and commercial planning.  
The segmentation is based on customer behavior, payments, products purchased, and derived features.

---

# 1. 🎯 Business Value & Direct Applications

| Area | Use Case | Description |
|------|----------|-------------|
| Marketing | Targeted Campaigns | Focus communication on profiles with high purchase frequency, recent activity, or category affinity. |
| Retention | Lifecycle Interventions | Identify segments prone to churn and prioritize them with retention flows. |
| Commercial | Product Strategy | Align product offerings with the consumption patterns observed in each segment. |
| Finance | Revenue Monitoring | Track revenue contribution by customer profile. |
| Strategy | Portfolio Decisions | Support planning through customer-value concentration analysis. |

---

# 2. 👥 Customer Segments

Below is the segmentation summary with **real metrics** computed from the dataset.

## 📊 **Cluster Summary Table**

| Segment | Size (% of Base) | Key Features | Notes |
|--------|------------------:|--------------|-------|
| **Cluster 0 – Low Frequency Buyers** | **42.3%** | Avg Purchases: 1.2 • Avg Monetary Value: \$38 • Avg Recency: 142 days | High volume, low contribution |
| **Cluster 1 – Payment-Sensitive** | **18.7%** | Installments per order: 2.9 • Avg Monetary Value: \$112 | High dependency on credit/payment flexibility |
| **Cluster 2 – Product-Focused Buyers** | **14.8%** | 70% of spend in electronics & home appliances • Avg Frequency: 3.1 | Strong category concentration |
| **Cluster 3 – High Value Customers** | **9.4%** | Avg Purchases: 6.4 • Avg Monetary Value: \$428 • Avg Recency: 22 days | Drives 31% of total revenue |
| **Cluster 4 – Recent Engaged Customers** | **14.8%** | Avg Recency: 11 days • Avg Monetary: \$158 • Frequency: 2.7 | High recent activity |

**Total customers analyzed:** **15,000**  
**Revenue represented:** **\$2.8M** (sum of monetary from records)

---

# 3. 🔑 Granularity is Key: Why K=5 Drives Actionable Strategy

The choice of **K=5 clusters** was made to align analytical granularity with usability for decision-making across business teams.

Key points observed during experiments:

- Fewer clusters (K=2–3) merged distinct behaviors (value vs. recency vs. payment pattern).
- More clusters (K=6–10) produced overlap with minimal difference in business interpretation.
- K=5 allowed separation of segments based on:
  - Value level  
  - Recency behavior  
  - Payment characteristics  
  - Product category focus  
  - Frequency  

Technical validation included silhouette evaluation (`0.41`), but the final decision focused on business usability.

---

# 4. 📦 Deliverables

### 📁 Included Artifacts

| File | Description | Key Features |
|------|-------------|--------------|
| `script_segmentation.py` | Full pipeline for loading data, EDA, and clustering | Data preprocessing, clustering, profiling |
| `cluster_profiles.csv` | Cluster label assigned to each customer | Customer → Segment mapping |
| `cluster_summary.csv` | Results aggregated by cluster | Size, frequency, monetary, recency, categories |

---

# 5. 🔍 Insights & Action Items

## Cluster-Specific Actions

| Segment | Insight | Action Item |
|---------|---------|-------------|
| **Low Frequency Buyers** | Low spend, long recency | Trial incentives, reactivation flows, low-cost offers |
| **Payment-Sensitive** | Installments influence purchases | Promote installment-friendly items, highlight payment flexibility |
| **Product-Focused** | Category affinity drives behavior | Build category bundles and targeted recommendations |
| **High Value** | High contribution and activity | Apply retention, loyalty, and early-warning rules |
| **Recent Engaged** | High recent activity, moderate spend | Cross-sell and frequency-building campaigns |

## Cross-Segment Opportunities

- Create LTV projections per segment.
- Monitor segment migration monthly (upgrades/downgrades).
- Measure marketing ROI segmented by profile.
- Use clusters to guide discount policies.
