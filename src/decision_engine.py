from .risk_engine import RiskEngine
from .cost_model import CostModel

class DecisionEngine:
    def __init__(self, ml_score, baseline_score, cost_ratio, risk_score, thresholds=None):
        self.ml_score = ml_score
        self.baseline_score = baseline_score
        self.cost_ratio = cost_ratio # ML Cost / Rule Cost. (>1 means ML is more expensive)
        self.risk_score = risk_score
        self.thresholds = thresholds or {
            'min_improvement': 0.05, # ML must be 5% better
            'max_risk': 0.7,
            'max_cost_ratio': 5.0 # ML shouldn't be more than 5x the cost of rules unless improvement is huge
        }

    def make_decision(self):
        improvement = self.ml_score - self.baseline_score
        
        reasons = []
        recommendation = ""
        
        # Logic Flow
        
        # 1. Performance Check
        if improvement < self.thresholds['min_improvement']:
            recommendation = "USE RULES / HEURISTICS"
            reasons.append(f"ML improvement ({improvement:.2%}) is negligible over baseline.")
            return recommendation, reasons

        # 2. Risk Check
        if self.risk_score > self.thresholds['max_risk']:
            recommendation = "HYBRID / HUMAN-IN-THE-LOOP"
            reasons.append(f"Risk score ({self.risk_score:.2f}) is too high for pure automation.")
            return recommendation, reasons

        # 3. Cost Check
        # If ML is way more expensive but improvement is marginal relative to cost?
        # Simple heuristic: if cost is 2x, we want significant improvement.
        if self.cost_ratio > self.thresholds['max_cost_ratio']:
            recommendation = "USE RULES (DUE TO COST)"
            reasons.append(f"ML cost is {self.cost_ratio:.1f}x higher than rules, not justified by {improvement:.2%} gain.")
            return recommendation, reasons

        # Otherwise
        recommendation = "USE AI / ML"
        reasons.append(f"ML shows significant lift ({improvement:.2%}) with acceptable risk and cost.")
        
        return recommendation, reasons
