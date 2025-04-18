import logging
from typing import Dict, Tuple, Union
from enum import Enum

logger = logging.getLogger(__name__)

class Category(Enum):
    """Enumeration for comparison categories."""
    MATH = "math"
    CODING = "coding"
    LONGER_QUERY = "longer_query"
    # Add more categories as needed

    @classmethod
    def from_string(cls, value: str) -> 'Category':
        """Converts a string to a Category enum member."""
        for member in cls:
            if member.value == value.lower():
                return member
        raise ValueError(f"'{value}' is not a valid Category")

# Note: This structure assumes ratings are stored like:
# {
#   "model_name": {
#     "categorie": {  <-- Potential typo: Should this be "category"?
#       "math": 1500.0,
#       "coding": 1450.0
#     }
#   }, ...
# }
# If "categorie" is indeed the intended key, keep it. Otherwise, rename it
# here and ensure consistency in file_operations.py and results.json.
# For now, we proceed assuming "categorie" is the intended key based on input files.
CATEGORY_KEY = "categorie" # Use a constant for the key

ModelRatingsType = Dict[str, Dict[str, Dict[str, float]]]

def update_elo(
    model_ratings: ModelRatingsType,
    model_a: str,
    model_b: str,
    category: Union[Category, str],
    score_a: float, # Score should be 1.0 (A wins), 0.0 (B wins), or 0.5 (Tie)
    k_factor: int = 32,
    initial_rating: float = 1500.0
) -> Tuple[float, float]:
    """
    Updates ELO ratings for two models based on a comparison result.

    Args:
        model_ratings: The dictionary holding all model ratings. Modified in place.
        model_a: Name of the first model.
        model_b: Name of the second model.
        category: The category (enum member or string) of the comparison.
        score_a: The score for model A (1.0 for win, 0.5 for tie, 0.0 for loss).
        k_factor: The K-factor determining rating change sensitivity.
        initial_rating: The rating assigned to a model if not already present.

    Returns:
        A tuple containing the new ELO ratings (new_rating_a, new_rating_b).
    """
    if isinstance(category, Category):
        category_str = category.value
    elif isinstance(category, str):
        category_str = category.lower()
        # Optional: Validate if the string corresponds to a known enum member
        try:
            Category.from_string(category_str)
        except ValueError:
            logger.warning(f"Updating ELO for an unknown category string: '{category_str}'")
    else:
        raise TypeError("category must be a Category enum member or a string")

    # --- Get current ratings, using defaults if models/category don't exist ---
    # Use .setdefault() to ensure the nested structure exists before accessing
    ratings_a_all = model_ratings.setdefault(model_a, {})
    ratings_a_cat = ratings_a_all.setdefault(CATEGORY_KEY, {})
    rating_a = ratings_a_cat.get(category_str, initial_rating)

    ratings_b_all = model_ratings.setdefault(model_b, {})
    ratings_b_cat = ratings_b_all.setdefault(CATEGORY_KEY, {})
    rating_b = ratings_b_cat.get(category_str, initial_rating)

    # --- Calculate expected scores ---
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a # Or 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

    # --- Calculate score for B ---
    # Ensure score_a is valid (0, 0.5, or 1)
    if score_a not in [0.0, 0.5, 1.0]:
        logger.error(f"Invalid score_a received: {score_a}. Aborting ELO update.")
        # Return current ratings as no update can be made
        return rating_a, rating_b
    score_b = 1.0 - score_a

    # --- Calculate new ratings ---
    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * (score_b - expected_b)

    # --- Update the ratings dictionary ---
    # The setdefault calls above already ensure the structure exists
    ratings_a_cat[category_str] = new_rating_a
    ratings_b_cat[category_str] = new_rating_b

    logger.info(f"ELO Update ({category_str}): {model_a} ({rating_a:.1f} -> {new_rating_a:.1f}), {model_b} ({rating_b:.1f} -> {new_rating_b:.1f}), Score A: {score_a}")

    return new_rating_a, new_rating_b

