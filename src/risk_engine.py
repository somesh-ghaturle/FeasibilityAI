class RiskEngine:
    def __init__(self, data_stats: dict, context_data: dict, ml_accuracy_std: float):
        self.stats = data_stats
        self.context = context_data
        self.ml_std = ml_accuracy_std

    def calculate_risk(self):
        """
        Calculates aggregate risk score 0.0 to 1.0 (1.0 = Max Risk)
        """
        # 1. Data Risk
        # - High missing values
        # - Low sample count
        # - High signal-to-noise (wait, high signal is low risk. Low signal is high risk)
        
        r_data = 0.0
        if self.stats.get('n_samples', 0) < 500: r_data += 0.4
        if self.stats.get('missing_ratio', 0) > 0.2: r_data += 0.3
        if self.stats.get('signal_to_noise_est', 1) < 0.6: r_data += 0.3
        
        # 2. Model Risk
        # - High variance (std) in CV
        r_model = 0.0
        if self.ml_std > 0.1: r_model += 0.5
        if self.ml_std > 0.05: r_model += 0.2

        # 3. Business Risk
        # - "High" criticality
        r_business = 0.0
        if self.context.get('decision_criticality') == 'high':
            r_business = 0.8
        elif self.context.get('decision_criticality') == 'medium':
            r_business = 0.4

        # Weighted Sum
        # Data risk is foundational, Business risk is a multiplier
        total_risk = (r_data * 0.4) + (r_model * 0.3) + (r_business * 0.3)
        return min(total_risk, 1.0)
