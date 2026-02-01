class CostModel:
    def __init__(self, context_data: dict):
        """
        context_data expected keys:
        - training_cost_est (USD)
        - inference_cost_monthly (USD)
        - maintenance_cost_monthly (USD)
        - rule_dev_time_hours (int)
        - ml_dev_time_hours (int)
        - hourly_rate (USD)
        """
        self.data = context_data

    def compute_roi_score(self):
        """
        Returns a cost comparison score. 
        Higher score = ML is more expensive relative to Rules.
        1.0 means cost parity. < 1.0 means ML is cheaper (rare). > 1.0 means ML is more expensive.
        """
        # Defaults
        rate = self.data.get('hourly_rate', 100)
        
        # Rule Based Cost (1 year)
        rule_dev_cost = self.data.get('rule_dev_time_hours', 20) * rate
        rule_maint_cost = self.data.get('rule_maintenance_monthly', 100) * 12
        total_rule_cost_1y = rule_dev_cost + rule_maint_cost

        # ML Cost (1 year)
        ml_dev_cost = self.data.get('ml_dev_time_hours', 100) * rate
        training_cost = self.data.get('training_cost_est', 500)
        inference_cost = self.data.get('inference_cost_monthly', 200) * 12
        ml_maint_cost = self.data.get('maintenance_cost_monthly', 500) * 12 # Retraining etc
        total_ml_cost_1y = ml_dev_cost + training_cost + inference_cost + ml_maint_cost

        if total_rule_cost_1y == 0:
            return 10.0 # Avoid div/0, rules are infinitely cheaper if free

        ratio = total_ml_cost_1y / total_rule_cost_1y
        return ratio, total_rule_cost_1y, total_ml_cost_1y
