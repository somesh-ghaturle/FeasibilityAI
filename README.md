# FeasibilityAI: AI Feasibility & ROI Decision System

## 🧠 Project Goal
A meta-AI system that evaluates a task + data and recommends:
- ✅ Use ML / LLM
- ❌ Use rules / SQL / heuristics
- ⚠ Hybrid approach
...and explains **WHY**, with cost, risk, and reliability estimates.

## 🧩 Core Problem It Solves
Companies jump into AI without asking:
- Do we have enough data?
- Will ML outperform rules?
- Is the cost justified?
- Is the error risk acceptable?

**FeasibilityAI answers these before AI is built.**

## 🏗️ Architecture
- **Feature Extraction Engine**: Analyzes data entropy, missingness, and signal-to-noise.
- **Baseline Estimator**: Checks how well simple rules or heuristics perform.
- **ML Estimator**: Trains lightweight models (XGBoost/LogReg) to guess potential lift.
- **Cost Model**: Calculates ROI based on dev time, training cost, and inference.
- **Risk Engine**: Scores data, model, and business risks.
- **Decision Engine**: The logic core that outputs the final Go/No-Go.

## 🚀 How to Run locally

1. **Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Django Server**
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

3. **Use the Tool**
   - Go to `http://127.0.0.1:8000`
   - Upload a CSV dataset (must have a target column).
   - Fill in cost/time estimates.
   - Click **Evaluate**.

## 🧠 Logic & Decision Thresholds
The system triggers "DO NOT USE AI" if:
- `ml_score - baseline_score < 5%`: The lift is negligible.
- `cost_ratio > 5.0`: ML is 5x more expensive than rules without massive gain.
- `risk_score > 0.7`: Too risky for automation (recommends Hybrid).

## License
MIT
