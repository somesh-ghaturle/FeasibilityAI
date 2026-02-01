class ExplainabilityReport:
    def __init__(self, data_stats, baseline_score, ml_score, ml_std, cost_data, risk_score, recommendation, reasons):
        self.stats = data_stats
        self.baseline = baseline_score
        self.ml_score = ml_score
        self.ml_std = ml_std
        self.cost = cost_data
        self.risk = risk_score
        self.recommendation = recommendation
        self.reasons = reasons

    def generate_report(self):
        report = f"""
=========================================
AI FEASIBILITY ASSESSMENT REPORT
=========================================
RECOMMENDATION: {self.recommendation}
-----------------------------------------
Reasoning:
{chr(10).join(['- ' + r for r in self.reasons])}

-----------------------------------------
1. DATA QUALITY SNAPSHOT
- Samples: {self.stats.get('n_samples')}
- Features: {self.stats.get('n_features')}
- Missing Ratio: {self.stats.get('missing_ratio'):.1%}
- Signal-to-Noise Est: {self.stats.get('signal_to_noise_est'):.2f} (0=Noise, 1=Perfect)

2. PERFORMANCE PROJECTIONS
- Baseline (Rules/Simple): {self.baseline:.2%}
- ML Model Est.: {self.ml_score:.2%} (±{self.ml_std:.2%})
- Lift: {self.ml_score - self.baseline:+.2%}

3. COST ROI ANALYSIS
- Rule-based (1yr): ${self.cost[1]:,.0f}
- ML-based (1yr):   ${self.cost[2]:,.0f}
- Cost Ratio: {self.cost[0]:.1f}x (ML is {self.cost[0]:.1f} times costlier)

4. RISK PROFILE
- Aggregate Risk Score: {self.risk:.2f} / 1.0
-----------------------------------------
"""
        return report
