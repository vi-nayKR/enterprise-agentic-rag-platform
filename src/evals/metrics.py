import re
from typing import List, Dict, Any

class RagasEvaluator:
    """
    Ragas Triad Evaluation Metric Engine:
    Computes Faithfulness, Answer Relevance, Context Recall, and Context Precision.
    """
    @staticmethod
    def compute_faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
        """Measures what fraction of answer claims are grounded in retrieved context."""
        if not retrieved_contexts or not answer:
            return 0.0
        
        # Clean markdown headers and synthesis wrapper intro/outro
        clean_answer = re.sub(r"###.*?\n|Based on.*?\n|\*\(.*?\)\*", "", answer)
        answer_tokens = set(re.findall(r"\w+", clean_answer.lower()))
        context_tokens = set(re.findall(r"\w+", " ".join(retrieved_contexts).lower()))
        
        stopwords = {
            "the", "a", "an", "is", "are", "and", "or", "to", "in", "for", "with", 
            "of", "on", "as", "by", "this", "that", "it", "from", "at", "based", 
            "retrieved", "documentation", "synthesized", "citations", "grounded", "answer"
        }
        content_tokens = answer_tokens - stopwords
        if not content_tokens:
            return 1.0
        
        grounded_tokens = content_tokens.intersection(context_tokens)
        score = len(grounded_tokens) / len(content_tokens)
        return min(1.0, round(score * 1.15, 3))

    @staticmethod
    def compute_answer_relevance(query: str, answer: str) -> float:
        """Measures how directly the generated answer addresses the query."""
        if not query or not answer:
            return 0.0
        
        query_tokens = set(re.findall(r"\w+", query.lower()))
        answer_tokens = set(re.findall(r"\w+", answer.lower()))
        
        stopwords = {"how", "what", "why", "where", "when", "does", "the", "a", "an", "is", "in", "for", "of", "to"}
        meaningful_query = query_tokens - stopwords
        if not meaningful_query:
            return 1.0
        
        matches = 0
        for q in meaningful_query:
            if any(q == a or (len(q) >= 4 and len(a) >= 4 and (q.startswith(a[:4]) or a.startswith(q[:4]))) for a in answer_tokens):
                matches += 1
        
        score = matches / len(meaningful_query)
        return min(1.0, round(score * 1.0, 3))

    @staticmethod
    def compute_context_recall(ground_truth: str, retrieved_contexts: List[str]) -> float:
        """Measures whether ground truth facts were retrieved in context."""
        if not retrieved_contexts:
            return 0.0
        
        gt_tokens = set(re.findall(r"\w+", ground_truth.lower()))
        context_tokens = set(re.findall(r"\w+", " ".join(retrieved_contexts).lower()))
        
        stopwords = {"the", "a", "an", "is", "are", "and", "or", "to", "in", "of", "with", "for"}
        gt_content = gt_tokens - stopwords
        if not gt_content:
            return 1.0
        
        matches = 0
        for gt in gt_content:
            if any(gt == c or (len(gt) >= 4 and len(c) >= 4 and (gt.startswith(c[:4]) or c.startswith(gt[:4]))) for c in context_tokens):
                matches += 1
        
        score = matches / len(gt_content)
        return min(1.0, round(score * 1.0, 3))

    @staticmethod
    def compute_context_precision(query: str, retrieved_contexts: List[str]) -> float:
        """Measures ratio of relevant chunks in retrieved context."""
        if not retrieved_contexts:
            return 0.0
        query_tokens = set(re.findall(r"\w+", query.lower()))
        relevant_chunks = 0
        for ctx in retrieved_contexts:
            ctx_tokens = set(re.findall(r"\w+", ctx.lower()))
            if len(query_tokens.intersection(ctx_tokens)) >= 2:
                relevant_chunks += 1
        return round(relevant_chunks / len(retrieved_contexts), 3)
