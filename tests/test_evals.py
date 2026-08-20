import pytest
from src.evals.metrics import RagasEvaluator
from src.evals.ragas_pipeline import run_ragas_evaluation

@pytest.mark.asyncio
async def test_metric_calculators():
    query = "How does LangGraph supervisor routing work?"
    context = ["LangGraph supervisor routes queries to specialist agents based on stateful cyclic graphs."]
    answer = "LangGraph supervisor routes queries to specialist agents."
    ground_truth = "LangGraph supervisor classifies intent and routes queries to specialist agents."

    faithfulness = RagasEvaluator.compute_faithfulness(answer, context)
    relevance = RagasEvaluator.compute_answer_relevance(query, answer)
    recall = RagasEvaluator.compute_context_recall(ground_truth, context)

    assert faithfulness >= 0.75
    assert relevance >= 0.70
    assert recall >= 0.75

@pytest.mark.asyncio
async def test_ragas_evaluation_pipeline():
    report = await run_ragas_evaluation()
    assert report["mean_faithfulness"] >= 0.70
    assert report["mean_answer_relevance"] >= 0.70
    assert report["mean_context_recall"] >= 0.70
    assert len(report["samples"]) == 4
