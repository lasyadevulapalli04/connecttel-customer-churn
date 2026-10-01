# ConnectTel Churn Analysis: Strategic Business Insights & Recommendations

This document translates technical model findings and exploratory data analysis into actionable strategic recommendations for ConnectTel business decision-makers.

---

## 1. Executive Summary

Customer churn is a critical operational metric for ConnectTel. Retaining existing subscribers requires proactive identification of churn indicators. Our predictive machine learning pipeline identifies high-risk customers with **80.62% test accuracy**, **65.83% precision**, and **56.15% recall** (F1-score: **0.6061**, ROC-AUC: **0.8422**), enabling targeted retention strategies before cancellation occurs.

---

## 2. Primary Churn Risk Drivers

Based on Exploratory Data Analysis (EDA) and SHAP feature importance analysis:

### A. Contract Commitment Type
- **Finding:** Customers on Month-to-month contracts exhibit the highest churn probability (~42.7%). Long-term contract holders (1-Year at ~11.3% and 2-Year at ~2.8%) demonstrate significantly higher loyalty.
- **Root Cause:** Low switching costs for Month-to-month subscribers make them susceptible to competitor promotions.

### B. High Monthly Charges & Fiber Optic Subscriptions
- **Finding:** Subscribers with higher monthly bills (especially Fiber Optic internet users at ~41.9% churn rate) show elevated churn rates compared to DSL users (~19.0%).
- **Root Cause:** Perceived value gap if pricing is high ($70–$110/month) without premium customer support.

### C. Absence of Support & Security Add-ons
- **Finding:** Customers who do NOT subscribe to Tech Support, Online Security, or Device Protection add-ons churn at ~41.6% vs ~15.2% for subscribers with support.
- **Root Cause:** Lack of product ecosystem integration reduces customer sticky factor.

### D. Electronic Check Payment Method
- **Finding:** Customers paying via Electronic Check churn at ~45.3% vs ~16.0% for automated credit card or bank transfer payments.

---

## 3. Targeted Retention Action Plan

| Customer Segment | Identified Risk Driver | Recommended Retention Action | Estimated Impact |
|---|---|---|---|
| Month-to-Month Users | High flexibility, easy cancellation | Offer 15% discount for 1-Year contract commitment | High churn reduction |
| Fiber Optic Users | High monthly bill friction | Bundle 3 months of free Premium Tech Support | Improved customer satisfaction |
| Electronic Check Payees | Manual billing friction | Provide $5 monthly credit for switching to Auto-Pay | Reduced billing drop-offs |
| New Tenure (< 6 Months) | Onboarding friction | Dispatch automated satisfaction check-ins at Month 1 & 3 | Higher 1-year survival |

---

## 4. Key Metrics for Business Success
- **Churn Reduction Rate:** Target 15-20% reduction in annual churn rate.
- **Retention Campaign ROI:** Measure revenue retained vs cost of promotional discounts.
- **Model Precision & Recall Balance:** Ensure retention offers target genuine high-risk customers without wasting discounts on low-risk customers.

---

## 5. SHAP Empirical Ranking Summary
- Top Feature 1: `Contract_Month-to-month` (+0.85 SHAP impact)
- Top Feature 2: `tenure` (-0.72 SHAP impact)
- Top Feature 3: `InternetService_Fiber optic` (+0.54 SHAP impact)
- Top Feature 4: `PaymentMethod_Electronic check` (+0.38 SHAP impact)
