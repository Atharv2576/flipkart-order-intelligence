"""Pre-loads every model the agent can call (the embedding model, Part 1's
return-risk model, Part 2's image classifier) once, up front, instead of
paying that cost inside the first user-facing request. Purely a latency
optimisation -- every model here is already lazily loaded on first use, so
skipping this call is always safe, just slower on the first turn.
"""
from part3.embeddings import embed
from part3.tools import _get_return_model, classify_product_image

_warmed_up = False


def warmup() -> None:
    global _warmed_up
    if _warmed_up:
        return

    embed(["warmup"])
    _get_return_model()

    from part2.config import SAMPLE_IMAGES_DIR

    sample = next(SAMPLE_IMAGES_DIR.glob("*.png"), None)
    if sample is not None:
        classify_product_image(str(sample))

    _warmed_up = True
