# FairTune
End-to-end **LLM fine-tuning + eval-as-code** platform with automated **safety & fairness (incl. ethnicity-parity)** checks.

## ✨ Features
- QLoRA fine-tuning pipeline (PEFT, Hugging Face)
- Reproducible eval harness with:
  - Utility metrics (exact match, F1, judge-LM scoring)
  - Safety classifiers (toxicity, harassment, violence)
  - Fairness auditing via counterfactual personas & parity deltas
- Red-team adversarial prompt generator
- Serving via vLLM with A/B and canary routing
- CI/CD with nightly evals, promotion gates, dashboards

## 🚀 Quickstart
```bash
git clone https://github.com/<you>/fairtune
cd fairtune

# Install deps
pip install -r requirements.txt

# Run baseline eval on base model
python src/eval_runner/run_eval.py --model meta-llama/Llama-3-8B-Instruct

# Fine-tune with QLoRA
python src/train_lora.py --config configs/train.yaml

# Evaluate candidate vs baseline
python src/eval_runner/run_eval.py --baseline baseline.json --candidate candidate.json

# Launch dashboard
streamlit run dash/app.py