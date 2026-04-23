from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Lazy imports for the evaluate library to avoid startup overhead 
# and give user time to install dependencies
def _get_evaluate():
    try:
        import evaluate
        return evaluate
    except (ImportError, TypeError, AttributeError) as e:
        logger.error(f"The 'evaluate' library failed to import due to a system/dependency error: {e}")
        logger.error("Tip: This is often caused by a bug in the 'datasets' library regarding 'jax'. Try running: pip install jax")
        return None

def bleu_score(predictions: List[str], references: List[str], max_order: int = 1) -> float:
    """
    Computes BLEU score (precision of generated text). 
    Defaults to BLEU-1 (max_order=1) as per user request.
    """
    eval_lib = _get_evaluate()
    if not eval_lib: return 0.0
    
    # BLEU expects references to be a list of lists of strings
    formatted_refs = [[r] for r in references]
    
    bleu = eval_lib.load("bleu")
    results = bleu.compute(predictions=predictions, references=formatted_refs, max_order=max_order)
    return float(results["bleu"])

def rouge_l_score(predictions: List[str], references: List[str]) -> float:
    """
    Computes ROUGE-L (Longest Common Subsequence) score.
    Measures recall and structural flow.
    """
    eval_lib = _get_evaluate()
    if not eval_lib: return 0.0
    
    rouge = eval_lib.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return float(results["rougeL"])

def wer_score(predictions: List[str], references: List[str]) -> float:
    """
    Computes Word Error Rate (WER).
    Measures edit distance (Lower is better).
    """
    eval_lib = _get_evaluate()
    if not eval_lib: return 1.0 # Max error if lib missing
    
    wer = eval_lib.load("wer")
    results = wer.compute(predictions=predictions, references=references)
    return float(results)
    
def calculate_text_report(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Helper to get a full text evaluation report."""
    return {
        "bleu1": bleu_score(predictions, references, max_order=1),
        "rougeL": rouge_l_score(predictions, references),
        "wer": wer_score(predictions, references)
    }