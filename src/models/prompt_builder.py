"""
Prompt construction: seed texts, context tiers, and tokenisation helpers.
"""


# ── Default seed prompts ─────────────────────────────────────────────────────

SEEDS = {
    "bharatgen": (
        "The BharatGen initiative is a government-backed research programme aimed at developing "
        "large-scale foundational AI models rooted in Indian languages, culture, and knowledge systems. "
        "It seeks to make AI accessible to over a billion people by training models on diverse Indic "
        "language corpora spanning Hindi, Tamil, Telugu, Bengali, Kannada, Malayalam, Marathi, Gujarati, "
        "and many other regional languages. The programme emphasises data sovereignty, ethical AI, and "
        "the democratisation of technology for underserved communities across rural and urban India. "
    ),
    "moe": (
        "Mixture-of-Experts (MoE) architectures improve the efficiency of large language models by "
        "activating only a subset of parameters for each input token. A gating network selects the "
        "top-k expert feed-forward networks to process each token, allowing total parameter count to "
        "scale without a proportional increase in compute per forward pass. Load balancing losses are "
        "added during training to prevent expert collapse. "
    ),
    "history": (
        "The history of artificial intelligence spans decades of research, beginning with the Dartmouth "
        "Conference of 1956. Early symbolic AI systems used hand-crafted rules. The field experienced "
        "AI winters before the rise of neural networks in the 1980s. The deep learning revolution of "
        "the 2010s, enabled by GPUs and large datasets, led to breakthroughs in vision, speech, and "
        "natural language processing. "
    ),
}

DEFAULT_TIERS = [
    {
        "label": "short (~256 tok)",
        "target_input": 220,
        "max_new_tokens": 128,
        "seed_key": "bharatgen",
        "question": "Summarise the BharatGen mission in 3 bullet points.",
    },
    {
        "label": "medium (~1024 tok)",
        "target_input": 980,
        "max_new_tokens": 256,
        "seed_key": "moe",
        "question": "Explain how MoE models balance efficiency and scale.",
    },
    {
        "label": "long (~2048 tok)",
        "target_input": 1980,
        "max_new_tokens": 256,
        "seed_key": "history",
        "question": "Write a numbered timeline of major AI milestones from 1956 to the deep learning era.",
    },
    {
        "label": "near-limit (~3584 tok)",
        "target_input": 3500,
        "max_new_tokens": 128,
        "seed_key": "bharatgen+moe",
        "question": "Write two sentences connecting BharatGen and MoE architectures.",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _repeat_to_tokens(tokenizer, seed: str, target: int) -> str:
    """Repeat seed text until it reaches the target token count."""
    chunk = seed.strip() + " "
    text = chunk
    while len(tokenizer.encode(text)) < target:
        text += chunk
    return tokenizer.decode(tokenizer.encode(text)[:target], skip_special_tokens=True)


def _resolve_seed(seed_key: str) -> str:
    """Resolve a seed key like 'bharatgen+moe' to actual text."""
    parts = seed_key.split("+")
    return " ".join(SEEDS[p.strip()] for p in parts if p.strip() in SEEDS)


# ── Public API ───────────────────────────────────────────────────────────────

def build_context_tiers(tokenizer, tiers: list[dict] | None = None) -> list[dict]:
    """
    Build prompts for each context tier.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        tiers: Optional list of tier dicts. Uses DEFAULT_TIERS if None.

    Returns:
        List of tier dicts with 'prompt' key added.
    """
    if tiers is None:
        tiers = [t.copy() for t in DEFAULT_TIERS]

    for t in tiers:
        seed_text = _resolve_seed(t.get("seed_key", "bharatgen"))
        body = _repeat_to_tokens(tokenizer, seed_text, t["target_input"])
        t["prompt"] = f"{body}\n\n{t['question']}"

    return tiers
