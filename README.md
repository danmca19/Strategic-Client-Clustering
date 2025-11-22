# 📈 Customer Segmentation: Unlocking 5 Actionable RetailX Client Profiles

This project builds a customer segmentation pipeline using clustering methods to support decision-making in marketing, retention, and commercial planning at RetailX.  
The goal is to produce segments that can be used directly by business teams for targeted actions, resource allocation, and customer lifecycle initiatives.

---

# 1. 🎯 Business Value & Direct Applications

| Area | Use Case | Description |
|------|----------|-------------|
| Marketing | Targeted Campaigns | Focus campaigns on segments with higher purchase frequency or higher order value. |
| Retention | Lifecycle Interventions | Create rules for early churn signals based on segment-specific behavior. |
| Commercial | Product Strategy | Map segments to product categories and adjust assortment. |
| Finance | Revenue Monitoring | Track revenue contribution by segment and changes over time. |
| Strategy | Portfolio Decisions | Evaluate long-term value of each cluster to support planning. |

---

# 2. 👥 Customer Segments

Below are the five customer segments generated through clustering, including the requested metric **“Size (% of Base)”**.

### **📊 Cluster Summary Table**

| Segment | Size (% of Base) | Key Features | Notes |
|--------|------------------:|--------------|-------|
| **Cluster 0 – Low Frequency Buyers** | X% | Low purchase count, low monetary value, long recency | Base volume driver with low engagement |
| **Cluster 1 – Payment-Sensitive Customers** | X% | Higher payment installment usage, moderate spend | Sensitive to credit or affordability |
| **Cluster 2 – Product-Focused Buyers** | X% | High concentration in specific product categories | Good for category campaigns |
| **Cluster 3 – High Value Customers** | X% | High frequency, high monetary value | Revenue priority group |
| **Cluster 4 – Recent Engaged Customers** | X% | Low recency, medium frequency | Suitable for loyalty and upsell |

> Replace the "X%" values with the actual values from your clustering output.

---

# 3. 🔑 Granularity is Key: Why K=5 Drives Actionable Strategy

Selecting **K=5** was based on achieving the level of granularity required by business teams while maintaining segment interpretability.

Key points used for this decision:

- 5 segments allow separation of payment behavior, product focus, and customer value.
- Fewer clusters reduced the ability to observe revenue-relevant patterns.
- More clusters fragmented audiences, lowering actionability.
- The silhouette score was used as a secondary check to ensure consistency.

---

# 4. 📦 Deliverables

### **📁 Included Artifacts**

| File | Description | Key Features |
|------|-------------|--------------|
| `script_segmentation.py` | Full pipeline for loading data, EDA, feature selection, and clustering | Data import, preprocessing, clustering model |
| `cluster_profiles.csv` | Final cluster labels per customer | Segment mapping |
| `cluster_summary.csv` | Metrics per cluster | Size, central tendencies |

---

# 5. 🔍 Insights & Action Items

### **Cluster-Driven Actions**

| Segment | Insight | Action Item |
|---------|---------|-------------|
| Low Frequency Buyers | Limited engagement | Introduce onboarding or welcome flows |
| Payment-Sensitive | Installments influence buying | Promote installment-friendly products |
| Product-Focused | Cluster tied to category preference | Create category-based bundles |
| High Value | High purchase frequency and spend | Prioritize with retention and loyalty programs |
| Recent Engaged | Active but not high value yet | Apply cross-sell and upsell recommendations |

### **Cross-Segment Opportunities**

- Build customer lifecycle rules using recency + cluster interaction.
- Create dashboards to track cluster migration over time.
- Use segments to guide budget allocation for marketing channels.
- Apply A/B testing per segment to measure lift.
