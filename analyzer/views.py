from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pandas as pd
import json
import os
import sys

# Ensure src is in path if needed, though being at root it should be fine if running from root
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.feature_extractor import FeatureExtractor
from src.baseline_models import BaselineEstimator
from src.ml_models import MLEstimator
from src.cost_model import CostModel
from src.risk_engine import RiskEngine
from src.decision_engine import DecisionEngine
from src.explainability import ExplainabilityReport

def home(request):
    if request.method == 'POST' and request.FILES.get('dataset'):
        try:
            # 1. Handle File Upload
            myfile = request.FILES['dataset']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            file_path = fs.path(filename)
            
            # Load Data
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif filename.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    return render(request, 'analyzer/home.html', {'error': 'Unsupported file format'})
            except Exception as e:
                return render(request, 'analyzer/home.html', {'error': f"Error reading file: {e}"})

            # 2. Get Form Data
            task_type = request.POST.get('task_type', 'classification')
            target_col = request.POST.get('target_col')
            
            # Cost/Context Inputs
            context_data = {
                'task_type': task_type,
                'decision_criticality': request.POST.get('criticality', 'medium'),
                'rule_dev_time_hours': int(request.POST.get('rule_dev_hours', 20)),
                'ml_dev_time_hours': int(request.POST.get('ml_dev_hours', 100)),
                'training_cost_est': float(request.POST.get('training_cost', 500)),
                'inference_cost_monthly': float(request.POST.get('inference_cost', 50)),
                'maintenance_cost_monthly': float(request.POST.get('maint_cost', 100)),
                'hourly_rate': float(request.POST.get('hourly_rate', 100)),
            }

            if target_col not in df.columns:
                 # Clean up details if we fail early
                return render(request, 'analyzer/home.html', {'error': f"Target column '{target_col}' not found. Columns: {', '.join(df.columns)}"})

            # 3. PIPELINE EXECUTION
            
            # A. Feature Extraction
            extractor = FeatureExtractor(df, target_col, task_type)
            stats = extractor.extract_features()

            # B. Baseline Stats
            baseline = BaselineEstimator(df, target_col, task_type)
            base_score = baseline.get_baseline_performance()

            # C. ML Performance Est
            ml_est = MLEstimator(df, target_col, task_type)
            ml_score, ml_std, best_model_name = ml_est.estimate_performance()

            # D. Cost Model
            cost_model = CostModel(context_data)
            cost_res = cost_model.compute_roi_score() # (ratio, rule_cost, ml_cost)

            # E. Risk Engine
            risk_eng = RiskEngine(stats, context_data, ml_std)
            risk_score = risk_eng.calculate_risk()

            # F. Decision Engine
            decider = DecisionEngine(ml_score, base_score, cost_res[0], risk_score)
            recommendation, reasons = decider.make_decision()

            # G. Explanation, passing best_model_name
            explainer = ExplainabilityReport(stats, base_score, ml_score, ml_std, cost_res, risk_score, recommendation, reasons, best_model_name)
            report_text = explainer.generate_report()

            # Clean up file?
            # os.remove(file_path)

            context = {
                'report': report_text,
                'recommendation': recommendation,
                'stats': stats,
                'base_score': base_score,
                'ml_score': ml_score,
                'cost_res': cost_res,
                'risk_score': risk_score,
                'reasons': reasons,
                'best_model': best_model_name 
            }
            return render(request, 'analyzer/results.html', context)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return render(request, 'analyzer/home.html', {'error': f"Pipeline failed: {e}"})

    return render(request, 'analyzer/home.html')
