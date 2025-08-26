import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def eval_utility(model_id: str):
    """
    Simple utility evaluation:
    - Loads a small subset of SQuAD (Q&A dataset).
    - Runs model with greedy decoding.
    - Scores with exact_match and F1 (from HuggingFace `evaluate`).
    """
    # Load small subset of SQuAD v1 for speed
    dataset = load_dataset("squad", split="validation[:50]")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    qa_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64)

    preds, refs = [], []

    for sample in dataset:
        question = sample["question"]
        context = sample["context"]
        prompt = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"

        output = qa_pipe(prompt)[0]["generated_text"]
        # Get answer after "Answer:" (very naive parsing)
        pred = output.split("Answer:")[-1].strip()

        preds.append(pred)
        refs.append(sample["answers"]["text"][0])  # use first ground-truth answer

    # Use HuggingFace evaluate metrics
    squad_metric = evaluate.load("squad")
    # Format predictions correctly for squad metric
    formatted_preds = [
        {"id": str(i), "prediction_text": preds[i]} for i in range(len(preds))
    ]
    formatted_refs = [
        {"id": str(i), "answers": {"text": [refs[i]], "answer_start": [0]}}
        for i in range(len(refs))
    ]

    results = squad_metric.compute(predictions=formatted_preds, references=formatted_refs)

    return {
        "exact_match": results["exact_match"],
        "f1": results["f1"],
        "notes": f"Evaluated on {len(dataset)} samples from SQuAD."
    }