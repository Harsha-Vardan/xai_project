import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DATASET = """
The patient arrived with a history of repeated anxiety episodes.
Family members reported that he was beaten during childhood.
He now avoids crowded spaces and social situations.
Clinical notes mention unresolved stress from early years.
A follow-up session was scheduled for supportive therapy.
"""

REFERENCE_OUTPUT = "Man suffered abuse"


def load_dataset_sample(name: str = "cnn") -> Tuple[str, str]:
    """Lightweight hardcoded samples in CNN/SQuAD style for quick demos."""
    key = name.strip().lower()

    if key == "cnn":
        text = """
        A man was beaten repeatedly for years.
        He was forced to work without pay.
        The court called it modern slavery.
        The accused were jailed.
        """
        reference = "Man suffered abuse"
        return text.strip(), reference

    if key == "squad":
        text = """
        The Supreme Court ruled on the case.
        The decision was announced in 2020.
        It changed legal interpretation.
        """
        reference = "The decision was announced in 2020"
        return text.strip(), reference

    return DATASET.strip(), REFERENCE_OUTPUT


def split_sentences(text: str) -> List[str]:
    """Split multiline text into clean sentence units."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        return lines

    parts = [part.strip() for part in text.split(".") if part.strip()]
    return [f"{part}." for part in parts]


def model_fn(context: str) -> str:
    """Simple deterministic rule-based model that simulates an LLM output."""
    lowered = context.lower()
    keywords = ("abuse", "trauma", "beaten")
    if any(keyword in lowered for keyword in keywords):
        return "Man suffered abuse"

    if "decision" in lowered and "2020" in lowered:
        return "The decision was announced in 2020"

    return "General case"


def _call_model(model, context: str, counter: Optional[Dict[str, int]] = None) -> str:
    if counter is not None:
        counter["count"] = counter.get("count", 0) + 1
    return model(context)


def loo_importance(
    context: str,
    model=model_fn,
    counter: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[float], str]:
    """Leave-One-Out sentence importance: 1 if output changes, else 0."""
    sentences = split_sentences(context)
    original_output = _call_model(model, context, counter)
    scores: List[float] = []

    for i in range(len(sentences)):
        reduced_context = " ".join(sent for j, sent in enumerate(sentences) if j != i)
        reduced_output = _call_model(model, reduced_context, counter)
        score = 1.0 if reduced_output != original_output else 0.0
        scores.append(score)

    return sentences, scores, original_output


def lime_importance(
    context: str,
    num_samples: int = 20,
    seed: int = 42,
    candidate_indices: Optional[List[int]] = None,
    model=model_fn,
    counter: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[float], str]:
    """Fast simplified LIME using random sentence masking."""
    sentences = split_sentences(context)
    original_output = _call_model(model, context, counter)

    n = len(sentences)
    if n == 0:
        return [], [], original_output

    candidates = candidate_indices if candidate_indices is not None else list(range(n))
    candidates = [idx for idx in candidates if 0 <= idx < n]

    rng = random.Random(seed)
    importance = [0.0] * n

    for _ in range(num_samples):
        mask = [1] * n
        for idx in candidates:
            mask[idx] = rng.randint(0, 1)

        perturbed_context = " ".join(sentences[idx] for idx in range(n) if mask[idx] == 1)
        perturbed_output = _call_model(model, perturbed_context, counter)

        if perturbed_output != original_output:
            for idx in candidates:
                if mask[idx] == 0:
                    importance[idx] += 1.0

    if num_samples > 0:
        importance = [round(score / num_samples, 3) for score in importance]

    return sentences, importance, original_output


def clime_importance(
    context: str,
    num_samples: int = 20,
    seed: int = 42,
    model=model_fn,
    counter: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[float], str]:
    """Contextual LIME: random masking that always preserves at least one sentence."""
    sentences = split_sentences(context)
    original_output = _call_model(model, context, counter)
    n = len(sentences)
    if n == 0:
        return [], [], original_output

    rng = random.Random(seed)
    importance = [0.0] * n

    for _ in range(num_samples):
        mask = [rng.randint(0, 1) for _ in range(n)]
        if sum(mask) == 0:
            mask[rng.randint(0, n - 1)] = 1

        perturbed_context = " ".join(sent for sent, keep in zip(sentences, mask) if keep == 1)
        perturbed_output = _call_model(model, perturbed_context, counter)

        if perturbed_output != original_output:
            for idx in range(n):
                if mask[idx] == 0:
                    importance[idx] += 1.0

    max_val = max(importance) if max(importance) > 0 else 1.0
    normalized = [round(score / max_val, 3) for score in importance]
    return sentences, normalized, original_output


def lshap_importance(
    context: str,
    num_samples: int = 20,
    seed: int = 42,
    model=model_fn,
    counter: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[float], str]:
    """Fast approximate SHAP via random subset marginal comparisons."""
    sentences = split_sentences(context)
    original_output = _call_model(model, context, counter)
    n = len(sentences)
    if n == 0:
        return [], [], original_output

    rng = random.Random(seed)
    importance = [0.0] * n

    for _ in range(num_samples):
        subset = [rng.randint(0, 1) for _ in range(n)]

        for idx in range(n):
            with_idx = subset.copy()
            without_idx = subset.copy()

            with_idx[idx] = 1
            without_idx[idx] = 0

            text_with = " ".join(sentences[j] for j in range(n) if with_idx[j] == 1)
            text_without = " ".join(sentences[j] for j in range(n) if without_idx[j] == 1)

            out_with = _call_model(model, text_with, counter)
            out_without = _call_model(model, text_without, counter)

            if out_with != out_without:
                importance[idx] += 1.0

    max_val = max(importance) if max(importance) > 0 else 1.0
    normalized = [round(score / max_val, 3) for score in importance]
    return sentences, normalized, original_output


def faithfulness(
    context: str,
    importance_scores: List[float],
    top_k: int = 2,
    model=model_fn,
    counter: Optional[Dict[str, int]] = None,
) -> int:
    """Faithfulness: 1 if removing top-k important sentences flips the output."""
    sentences = split_sentences(context)
    if not sentences:
        return 0

    top_k = max(1, min(top_k, len(sentences)))
    original_output = _call_model(model, context, counter)

    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda idx: importance_scores[idx] if idx < len(importance_scores) else 0.0,
        reverse=True,
    )
    removed = set(ranked_indices[:top_k])

    reduced_context = " ".join(sent for idx, sent in enumerate(sentences) if idx not in removed)
    reduced_output = _call_model(model, reduced_context, counter)

    return 1 if reduced_output != original_output else 0


def hybrid_importance_dynamic(
    context: str,
    threshold: float = 0.1,
    num_samples: int = 20,
    seed: int = 42,
    model=model_fn,
    counter: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[float], List[float], List[int], str]:
    """Hybrid explainer: thresholded LOO selection, then LIME on selected sentences."""
    sentences, loo_scores, original_output = loo_importance(context, model=model, counter=counter)
    n = len(sentences)
    if n == 0:
        return [], [], [], [], original_output

    selected_indices = [idx for idx, score in enumerate(loo_scores) if score > threshold]

    if not selected_indices:
        best_idx = max(range(n), key=lambda idx: loo_scores[idx])
        selected_indices = [best_idx]

    selected_text = " ".join(sentences[idx] for idx in selected_indices)

    _, selected_lime_scores, _ = lime_importance(
        selected_text,
        num_samples=num_samples,
        seed=seed,
        model=model,
        counter=counter,
    )

    final_scores = [0.0] * n
    for local_idx, original_idx in enumerate(selected_indices):
        if local_idx < len(selected_lime_scores):
            final_scores[original_idx] = selected_lime_scores[local_idx]

    return sentences, final_scores, loo_scores, selected_indices, original_output


def _print_scores(title: str, sentences: List[str], scores: List[float]) -> None:
    print(f"\n{title}")
    for idx, (sentence, score) in enumerate(zip(sentences, scores), start=1):
        print(f"  {idx}. score={score:<5} | {sentence}")


def _plot_results(results: List[Dict[str, object]], output_dir: str = "plots") -> None:
    """Generate and save per-dataset and summary plots for easier interpretation."""
    if plt is None:
        print("\nPlotting skipped: matplotlib is not installed.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for result in results:
        dataset = str(result["dataset"])
        sentences = result["sentences"]
        scores_map = result["scores"]
        faithfulness_map = result["faithfulness"]
        model_calls_map = result["model_calls"]

        methods = list(scores_map.keys())
        x = list(range(1, len(sentences) + 1))

        fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)

        width = 0.14
        center = (len(methods) - 1) / 2
        for idx, method in enumerate(methods):
            shifted = [val + (idx - center) * width for val in x]
            axes[0].bar(shifted, scores_map[method], width=width, label=method)
        axes[0].set_title(f"{dataset.upper()} - Sentence Importance by Method")
        axes[0].set_xlabel("Sentence index")
        axes[0].set_ylabel("Importance score")
        axes[0].set_xticks(x)
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(loc="upper right")

        faith_methods = list(faithfulness_map.keys())
        faith_values = [faithfulness_map[m] for m in faith_methods]
        axes[1].bar(faith_methods, faith_values)
        axes[1].set_title(f"{dataset.upper()} - Faithfulness (Top-k removal)")
        axes[1].set_ylabel("Faithfulness")
        axes[1].set_ylim(0, 1.2)

        call_methods = list(model_calls_map.keys())
        call_values = [model_calls_map[m] for m in call_methods]
        axes[2].bar(call_methods, call_values)
        axes[2].set_title(f"{dataset.upper()} - Model Calls per Method")
        axes[2].set_ylabel("Number of model calls")
        axes[2].tick_params(axis="x", rotation=20)

        figure_file = output_path / f"{dataset}_results.png"
        fig.savefig(figure_file, dpi=150)
        plt.close(fig)
        print(f"Plot saved: {figure_file}")

    summary_fig, summary_ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    dataset_labels = [str(r["dataset"]) for r in results]
    hybrid_faith = [int(r["faithfulness"]["Hybrid"]) for r in results]
    summary_ax.bar(dataset_labels, hybrid_faith)
    summary_ax.set_title("Hybrid Faithfulness Across Datasets")
    summary_ax.set_ylabel("Faithfulness")
    summary_ax.set_ylim(0, 1.2)
    summary_file = output_path / "hybrid_consistency.png"
    summary_fig.savefig(summary_file, dpi=150)
    plt.close(summary_fig)
    print(f"Plot saved: {summary_file}")


def _run_dataset(dataset_name: str) -> Dict[str, object]:
    context, reference_output = load_dataset_sample(dataset_name)

    print("\n==================================================")
    print(f"Dataset: {dataset_name}")
    print("Dataset sentences:")
    sentences = split_sentences(context)
    for idx, sentence in enumerate(sentences, start=1):
        print(f"  {idx}. {sentence}")

    print(f"\nReference output: {reference_output}")
    print(f"Model output on full context: {model_fn(context)}")

    loo_counter = {"count": 0}
    loo_sentences, loo_scores, _ = loo_importance(context, counter=loo_counter)
    loo_faith = faithfulness(context, loo_scores, top_k=2, counter=loo_counter)

    lime_counter = {"count": 0}
    lime_sentences, lime_scores, _ = lime_importance(context, num_samples=20, seed=42, counter=lime_counter)
    lime_faith = faithfulness(context, lime_scores, top_k=2, counter=lime_counter)

    clime_counter = {"count": 0}
    clime_sentences, clime_scores, _ = clime_importance(context, num_samples=20, seed=42, counter=clime_counter)
    clime_faith = faithfulness(context, clime_scores, top_k=2, counter=clime_counter)

    lshap_counter = {"count": 0}
    lshap_sentences, lshap_scores, _ = lshap_importance(context, num_samples=20, seed=42, counter=lshap_counter)
    lshap_faith = faithfulness(context, lshap_scores, top_k=2, counter=lshap_counter)

    hybrid_counter = {"count": 0}
    (
        hybrid_sentences,
        hybrid_scores,
        hybrid_loo_scores,
        hybrid_selected_indices,
        _,
    ) = hybrid_importance_dynamic(
        context,
        threshold=0.1,
        num_samples=20,
        seed=17,
        counter=hybrid_counter,
    )
    hybrid_faith = faithfulness(context, hybrid_scores, top_k=2, counter=hybrid_counter)

    _print_scores("LOO Importance Scores:", loo_sentences, loo_scores)
    _print_scores("LIME Importance Scores:", lime_sentences, lime_scores)
    _print_scores("C-LIME Importance Scores:", clime_sentences, clime_scores)
    _print_scores("L-SHAP Importance Scores:", lshap_sentences, lshap_scores)
    _print_scores("Dynamic Hybrid Importance Scores:", hybrid_sentences, hybrid_scores)

    print("\n--- RESULTS ---")
    print("LOO:", loo_scores)
    print("LIME:", lime_scores)
    print("C-LIME:", clime_scores)
    print("L-SHAP:", lshap_scores)
    print("Hybrid:", hybrid_scores)

    print("\nDynamic Hybrid Scores:", hybrid_scores)
    print("Hybrid selected indices from LOO (threshold > 0.1):", [idx + 1 for idx in hybrid_selected_indices])
    print("Hybrid internal LOO scores:", hybrid_loo_scores)

    print("\nFaithfulness")
    print(f"  LOO: {loo_faith}")
    print(f"  LIME: {lime_faith}")
    print(f"  C-LIME: {clime_faith}")
    print(f"  L-SHAP: {lshap_faith}")
    print(f"  Hybrid: {hybrid_faith}")

    print("\nModel Calls")
    print(f"  LOO pipeline calls: {loo_counter['count']}")
    print(f"  LIME pipeline calls: {lime_counter['count']}")
    print(f"  C-LIME pipeline calls: {clime_counter['count']}")
    print(f"  L-SHAP pipeline calls: {lshap_counter['count']}")
    print(f"  Hybrid pipeline calls: {hybrid_counter['count']}")

    scores_map = {
        "LOO": loo_scores,
        "LIME": lime_scores,
        "C-LIME": clime_scores,
        "L-SHAP": lshap_scores,
        "Hybrid": hybrid_scores,
    }
    faithfulness_map = {
        "LOO": loo_faith,
        "LIME": lime_faith,
        "C-LIME": clime_faith,
        "L-SHAP": lshap_faith,
        "Hybrid": hybrid_faith,
    }
    model_calls_map = {
        "LOO": loo_counter["count"],
        "LIME": lime_counter["count"],
        "C-LIME": clime_counter["count"],
        "L-SHAP": lshap_counter["count"],
        "Hybrid": hybrid_counter["count"],
    }

    return {
        "dataset": dataset_name,
        "sentences": sentences,
        "scores": scores_map,
        "faithfulness": faithfulness_map,
        "model_calls": model_calls_map,
        "hybrid_scores": hybrid_scores,
        "hybrid_faith": hybrid_faith,
    }


def main() -> None:
    print("=== Explainable AI Demo (LOO, LIME, C-LIME, L-SHAP, Dynamic Hybrid) ===")

    results = []
    for dataset_name in ("cnn", "squad"):
        results.append(_run_dataset(dataset_name))

    consistent = all(result["hybrid_faith"] == 1 for result in results)

    print("\n==================================================")
    print("Hybrid Consistency Check")
    for result in results:
        print(
            f"  {result['dataset']}: faithfulness={result['hybrid_faith']}, "
            f"scores={result['hybrid_scores']}"
        )
    print(f"Hybrid works consistently: {'YES' if consistent else 'NO'}")

    _plot_results(results, output_dir="plots")


if __name__ == "__main__":
    main()
