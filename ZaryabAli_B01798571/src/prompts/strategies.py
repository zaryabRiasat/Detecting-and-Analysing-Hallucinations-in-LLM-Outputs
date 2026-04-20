# src/prompts/strategies.py

from typing import Tuple, List, Dict
from utils import truncate


# ── Few-shot example bank ──────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES: List[Dict] = [
    {
        "question":  "What is the capital of France?",
        "knowledge": "France is a country in Western Europe. Its capital and largest city is Paris.",
        "answer":    "The capital of France is Paris.",
        "verdict":   "FACTUAL",
        "reason":    "The answer correctly states Paris as the capital, which is confirmed by the knowledge passage.",
    },
    {
        "question":  "What is the capital of Germany?",
        "knowledge": "Germany is a country in central Europe. Its capital city is Berlin.",
        "answer":    "The capital of Germany is Madrid.",
        "verdict":   "HALLUCINATED",
        "reason":    "The answer says Madrid but the knowledge passage clearly states Berlin is Germany's capital.",
    },
    {
        "question":  "Who was Albert Einstein?",
        "knowledge": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
        "answer":    "Albert Einstein was a professional footballer who played for Germany.",
        "verdict":   "HALLUCINATED",
        "reason":    "The answer contradicts the knowledge passage — Einstein was a physicist, not a footballer.",
    },
    {
        "question":  "Who created Python?",
        "knowledge": "Python is a programming language created by Guido van Rossum, first released in 1991.",
        "answer":    "Python was created by Guido van Rossum and first released in 1991.",
        "verdict":   "FACTUAL",
        "reason":    "The answer matches the knowledge passage exactly.",
    },
    {
        "question":  "What is the moon made of?",
        "knowledge": "The Moon is a rocky body composed primarily of silicate minerals and metals.",
        "answer":    "The moon is made of cheese.",
        "verdict":   "HALLUCINATED",
        "reason":    "The answer directly contradicts the knowledge that the Moon is composed of silicate minerals.",
    },
]


# ── System prompt (shared across strategies) ──────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert hallucination detection system. "
    "Your task is to determine whether a given answer is factually correct and supported "
    "by the provided knowledge context, or whether it contains hallucinations — "
    "statements that are incorrect, fabricated, or unsupported by evidence. "
    "Be precise, consistent, and unbiased."
)


# ── Strategy 1: Zero-shot ─────────────────────────────────────────────────────
def zero_shot_prompt(
    answer: str,
    question: str = "",
    knowledge: str = "",
) -> Tuple[str, str]:

    parts = []
    if knowledge:
        parts.append(f"Knowledge Context:\n{truncate(knowledge, 600)}")
    if question:
        parts.append(f"Question:\n{question}")
    parts.append(f"Answer to evaluate:\n{truncate(answer, 400)}")

    parts.append(
        "\nTask: Is the above answer factually correct and supported by the knowledge context?\n"
        "Respond with exactly one word: HALLUCINATED or FACTUAL"
    )

    user_prompt = "\n\n".join(parts)
    return SYSTEM_PROMPT, user_prompt


# ── Strategy 2: Few-shot ──────────────────────────────────────────────────────
def few_shot_prompt(
    answer:    str,
    question:  str = "",
    knowledge: str = "",
    k:         int = 3,
) -> Tuple[str, str]:
    
    examples_text = []
    for ex in FEW_SHOT_EXAMPLES[:k]:
        block = []
        if ex["knowledge"]:
            block.append(f"Knowledge: {ex['knowledge']}")
        if ex["question"]:
            block.append(f"Question: {ex['question']}")
        block.append(f"Answer: {ex['answer']}")
        block.append(f"Verdict: {ex['verdict']}")
        examples_text.append("\n".join(block))

    examples_section = "\n\n---\n\n".join(examples_text)

    query_parts = []
    if knowledge:
        query_parts.append(f"Knowledge: {truncate(knowledge, 500)}")
    if question:
        query_parts.append(f"Question: {question}")
    query_parts.append(f"Answer: {truncate(answer, 400)}")
    query_parts.append("Verdict:")

    user_prompt = (
        "Here are some labelled examples:\n\n"
        f"{examples_section}\n\n"
        "---\n\nNow evaluate this new example:\n\n"
        + "\n".join(query_parts)
    )

    return SYSTEM_PROMPT, user_prompt


# ── Strategy 3: Chain-of-Thought ──────────────────────────────────────────────
def chain_of_thought_prompt(
    answer:    str,
    question:  str = "",
    knowledge: str = "",
) -> Tuple[str, str]:
    
    parts = []
    if knowledge:
        parts.append(f"Knowledge Context:\n{truncate(knowledge, 600)}")
    if question:
        parts.append(f"Question:\n{question}")
    parts.append(f"Answer to evaluate:\n{truncate(answer, 400)}")

    parts.append(
        "\nPlease reason through this step by step:\n"
        "Step 1: What factual claims does the answer make?\n"
        "Step 2: Are these claims supported or contradicted by the knowledge context?\n"
        "Step 3: Are there any statements that are implausible, anachronistic, or fabricated?\n"
        "Step 4: Based on your reasoning above, give your final verdict.\n\n"
        "End your response with exactly: VERDICT: HALLUCINATED or VERDICT: FACTUAL"
    )

    user_prompt = "\n\n".join(parts)
    return SYSTEM_PROMPT, user_prompt


# ── Strategy 4: Structured output ─────────────────────────────────────────────
def structured_prompt(
    answer:    str,
    question:  str = "",
    knowledge: str = "",
) -> Tuple[str, str]:

    parts = []
    if knowledge:
        parts.append(f"Knowledge Context:\n{truncate(knowledge, 600)}")
    if question:
        parts.append(f"Question:\n{question}")
    parts.append(f"Answer to evaluate:\n{truncate(answer, 400)}")

    parts.append(
        '\nEvaluate whether the answer is factually correct given the knowledge context.\n'
        'Respond ONLY with a JSON object in this exact format (no other text):\n'
        '{\n'
        '  "verdict": "HALLUCINATED" or "FACTUAL",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "reasoning": "<one sentence explanation>"\n'
        '}'
    )

    user_prompt = "\n\n".join(parts)
    return SYSTEM_PROMPT, user_prompt


# ── Strategy dispatcher ────────────────────────────────────────────────────────
def build_prompt(
    strategy:  str,
    answer:    str,
    question:  str = "",
    knowledge: str = "",
    few_shot_k: int = 3,
) -> Tuple[str, str]:
    """
    Return (system_prompt, user_prompt) for the requested strategy.
    """
    answer    = str(answer).strip()
    question  = str(question).strip()
    knowledge = str(knowledge).strip()

    if strategy == "zero_shot":
        return zero_shot_prompt(answer, question, knowledge)
    elif strategy == "few_shot":
        return few_shot_prompt(answer, question, knowledge, k=few_shot_k)
    elif strategy == "chain_of_thought":
        return chain_of_thought_prompt(answer, question, knowledge)
    elif strategy == "structured":
        return structured_prompt(answer, question, knowledge)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                         f"Choose from: zero_shot, few_shot, chain_of_thought, structured")
