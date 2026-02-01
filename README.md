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

## � Installation Guide

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd FeasibilityAI
   ```

2. **Create a Virtual Environment**
   It's recommended to work within a virtual environment to keep dependencies clean.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Database**
   Apply migrations to set up the internal Django database (SQLite).
   ```bash
   python manage.py migrate
   ```

## 🛠️ How to Use This Tool

1. **Start the Application**
   ```bash
   python manage.py runserver
   ```
   You will see an output indicating the server is running at `http://127.0.0.1:8000/`.

2. **Generate Test Data (Optional)**
   We have included a script to generate 3 test datasets for you to verify the system logic:
   ```bash
   python generate_data.py
   ```
   This will create a `data/` folder with:
   - `customer_churn_feasible.csv` (AI Recommended)
   - `rule_based_loan.csv` (Rules Recommended)
   - `random_noise_infeasible.csv` (No AI / Noise)

3. **Run a Feasibility Audit**
   - Open your browser to `http://127.0.0.1:8000`.
   - **Upload Dataset**: Select one of the CSV files generated above (e.g., `data/customer_churn_feasible.csv`).
   - **Target Column**: Enter the exact name of the column you want to predict (e.g., `churn` for the churn dataset).
   - **Set Context**: Adjust the Development Hours and Cost estimates to reflect your real-world constraints.
   - Click **Evaluate Feasibility**.

4. **Interpret Results**
   The system will generate a report showing:
   - **Recommendation**: Green (Use AI), Yellow (Hybrid), or Red (Use Rules).
   - **Why?**: A breakdown of Lift, Cost, and Risk.
   - **Details**: Signal-to-noise ratio, estimated ROI, and baseline comparisons.

## 🧠 Logic & Decision Thresholds
The system triggers "DO NOT USE AI" if:
- `ml_score - baseline_score < 5%`: The lift is negligible.
- `cost_ratio > 5.0`: ML is 5x more expensive than rules without massive gain.
- `risk_score > 0.7`: Too risky for automation (recommends Hybrid).

## License
MIT
