# src/pipeline.py

from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional

from api_client import LLMClient
from prompts.strategies import build_prompt
from utils import parse_verdict, parse_confidence, normalize_text
from config import SETTINGS

logger = logging.getLogger(__name__)


class LLMJudgePipeline:

    def __init__(
        self,
        strategy:    str   = None,
        provider:    str   = None,
        model:       str   = None,
        threshold:   float = 0.5,
        few_shot_k:  int   = None,
        consistency_votes: int = 1,
    ):
        self.strategy    = strategy   or SETTINGS.prompt.default_strategy
        self.threshold   = threshold
        self.few_shot_k  = few_shot_k or SETTINGS.prompt.few_shot_k
        self.consistency_votes = consistency_votes or SETTINGS.prompt.consistency_votes

        self.client = LLMClient(provider=provider, model=model)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        text:      Optional[str] = None,
        answer:    Optional[str] = None,
        question:  Optional[str] = None,
        knowledge: Optional[str] = None,
    ) -> Dict[str, Any]:
        
        if answer is None:
            answer = text or ""

        answer    = normalize_text(str(answer))
        question  = normalize_text(str(question  or ""))
        knowledge = normalize_text(str(knowledge or ""))

        if self.consistency_votes > 1:
            return self._predict_with_consistency(answer, question, knowledge)

        return self._predict_single(answer, question, knowledge)

    def _predict_single(
        self,
        answer:    str,
        question:  str,
        knowledge: str,
    ) -> Dict[str, Any]:
        system_prompt, user_prompt = build_prompt(
            strategy   = self.strategy,
            answer     = answer,
            question   = question,
            knowledge  = knowledge,
            few_shot_k = self.few_shot_k,
        )

        t0 = time.time()
        try:
            response_text, usage = self.client.call(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"API call failed: {e}")
            response_text = ""
            usage = {"prompt_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        verdict    = parse_verdict(response_text)
        confidence = parse_confidence(response_text)

        # Fallback: if verdict unparseable, default to confidence threshold
        if verdict is None:
            verdict = 1 if confidence > self.threshold else 0
            logger.warning(f"Verdict unparseable from: {response_text[:80]!r} — defaulting to {verdict}")

        label = verdict
        prob  = confidence if label == 1 else (1.0 - confidence)
        # Ensure prob reflects hallucination probability
        if label == 1 and confidence < 0.5:
            prob = 0.6   # low-confidence hallucination
        elif label == 0 and confidence < 0.5:
            prob = 0.3   # low-confidence factual

        explanation = (
            f"LLM-as-Judge ({self.client.model}, {self.strategy}) "
            f"verdict: {'HALLUCINATED' if label == 1 else 'FACTUAL'} "
            f"(confidence={confidence:.2f})"
        )

        return {
            "input_text":  f"Q: {question} | A: {answer}",
            "answer":      answer,
            "question":    question,
            "knowledge":   knowledge,
            "raw_response": response_text,
            "final_result": {
                "label":          label,
                "probability":    round(prob, 4),
                "final_score":    round(prob, 4),
                "confidence":     round(confidence, 4),
                "explanation":    explanation,
                "strategy":       self.strategy,
                "model":          self.client.model,
                "threshold":      self.threshold,
                "inference_ms":   elapsed_ms,
                "usage":          usage,
            },
        }

    def _predict_with_consistency(
        self,
        answer:    str,
        question:  str,
        knowledge: str,
    ) -> Dict[str, Any]:
        
        votes = []
        confidences = []

        for _ in range(self.consistency_votes):
            result = self._predict_single(answer, question, knowledge)
            fr     = result["final_result"]
            votes.append(fr["label"])
            confidences.append(fr["confidence"])
            time.sleep(0.5)   # small delay between calls

        final_label = 1 if sum(votes) > len(votes) / 2 else 0
        avg_confidence = sum(confidences) / len(confidences)

        result = self._predict_single(answer, question, knowledge)
        result["final_result"]["label"]       = final_label
        result["final_result"]["probability"] = round(avg_confidence if final_label == 1 else 1 - avg_confidence, 4)
        result["final_result"]["final_score"] = result["final_result"]["probability"]
        result["final_result"]["votes"]       = votes
        result["final_result"]["consistency_votes"] = self.consistency_votes
        return result

    def predict_row(self, row: dict) -> Dict[str, Any]:
        answer    = normalize_text(str(row.get("text",      "")))
        question  = normalize_text(str(row.get("question",  "")))
        knowledge = normalize_text(str(row.get("knowledge", "")))
        result    = self.predict(answer=answer, question=question, knowledge=knowledge)
        return result["final_result"]

    def usage_summary(self) -> Dict[str, Any]:
        return self.client.usage_summary()

    def print_usage(self) -> None:
        self.client.print_usage()
