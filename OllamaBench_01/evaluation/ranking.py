import json
import logging
from typing import Dict, Tuple, Union, TypeAlias, Optional
import math

logger = logging.getLogger(__name__)

# --- Type Alias for Model Ratings Structure ---
# Defines the expected nested dictionary structure for clarity.
ModelRatingsType: TypeAlias = Dict[str, # Model Name (e.g., "llama3:latest")
                                  Dict[str, # Should always be the key "categorie"
                                       Dict[str, float] # Category Name (e.g., "math") -> ELO Rating (float)
                                      ]
                                 ]

# --- Constants ---
CATEGORY_DICT_KEY = "categorie" # The fixed key used in the dictionary structure
DEFAULT_INITIAL_RATING = 1000.0
DEFAULT_K_FACTOR = 32

# --- ELO Calculation Logic ---
def _calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """Calculates the expected score for player A against player B."""
    return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

def update_elo(
    model_ratings: ModelRatingsType,
    model_a: str,
    model_b: str,
    category: str, # Category should always be a string now
    score_a: float, # 1.0 (A wins), 0.0 (B wins), 0.5 (Tie)
    k_factor: int = DEFAULT_K_FACTOR,
    initial_rating: float = DEFAULT_INITIAL_RATING
) -> Tuple[Optional[float], Optional[float]]:
    """
    Updates the ELO ratings for two models within a specific category based on a comparison result.
    Modifies the `model_ratings` dictionary *in place*.

    Args:
        model_ratings: The dictionary containing all current model ratings.
                       Expected structure: {model: {"categorie": {cat: rating}}}.
                       This dictionary will be modified directly.
        model_a: Name of the first model.
        model_b: Name of the second model.
        category: The category string (e.g., "math", "coding") for the comparison.
                  Will be converted to lowercase.
        score_a: Outcome score for model A (1.0 for win, 0.5 for tie, 0.0 for loss).
        k_factor: The K-factor determining rating change sensitivity.
        initial_rating: The rating assigned if a model/category doesn't exist yet.

    Returns:
        A tuple containing the new ELO ratings (new_rating_a, new_rating_b) after the update.
        Returns (None, None) if the input score_a is invalid.
    """
    if not model_a or not model_b:
        logger.error("ELO update failed: Model names cannot be empty.")
        return None, None
    if not category:
        logger.error("ELO update failed: Category string cannot be empty.")
        return None, None
    if score_a not in [0.0, 0.5, 1.0]:
        logger.error(f"ELO update failed: Invalid score_a value '{score_a}'. Must be 0.0, 0.5, or 1.0.")
        return None, None # Return None for both ratings on invalid score

    category_str = category.lower() # Ensure category is lowercase

    # --- Get current ratings, initializing if necessary ---
    # Use setdefault to create nested dictionaries if they don't exist
    ratings_a_all = model_ratings.setdefault(model_a, {CATEGORY_DICT_KEY: {}})
    ratings_a_cat = ratings_a_all.setdefault(CATEGORY_DICT_KEY, {}) # Ensure "categorie" key exists
    rating_a = ratings_a_cat.setdefault(category_str, initial_rating) # Get or set initial rating

    ratings_b_all = model_ratings.setdefault(model_b, {CATEGORY_DICT_KEY: {}})
    ratings_b_cat = ratings_b_all.setdefault(CATEGORY_DICT_KEY, {})
    rating_b = ratings_b_cat.setdefault(category_str, initial_rating)

    logger.debug(f"ELO Pre-Update ({category_str}): {model_a}={rating_a:.1f}, {model_b}={rating_b:.1f}")

    # --- Calculate ELO update ---
    expected_a = _calculate_expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a # Or _calculate_expected_score(rating_b, rating_a)

    score_b = 1.0 - score_a # Calculate score for B

    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * (score_b - expected_b)

    # --- Update the ratings dictionary IN PLACE ---
    ratings_a_cat[category_str] = new_rating_a
    ratings_b_cat[category_str] = new_rating_b

    logger.info(f"ELO Update ({category_str}): {model_a} ({rating_a:.1f} -> {new_rating_a:.1f}), {model_b} ({rating_b:.1f} -> {new_rating_b:.1f}), Score A: {score_a}, K: {k_factor}")

    return new_rating_a, new_rating_b


# --- Example Usage (for understanding) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example ratings structure
    current_ratings: ModelRatingsType = {
        "model_x": {
            "categorie": {
                "math": 1050.0,
                "coding": 1100.0
            }
        },
        "model_y": {
            "categorie": {
                "math": 950.0
                # 'coding' category doesn't exist for model_y yet
            }
        },
         "model_z": {
             "categorie": {} # No categories yet
         }
    }

    print("Initial Ratings:")
    print(json.dumps(current_ratings, indent=2))
    print("-" * 30)

    # --- Scenario 1: Model X beats Model Y in Math ---
    print("Scenario 1: Model X beats Model Y in Math (score_a=1.0)")
    new_x_math, new_y_math = update_elo(current_ratings, "model_x", "model_y", "math", 1.0)
    print(f"Updated Math Ratings: X={new_x_math:.1f}, Y={new_y_math:.1f}")
    print("Current Ratings Structure:")
    print(json.dumps(current_ratings, indent=2))
    print("-" * 30)

    # --- Scenario 2: Model Y ties Model X in Coding (Y starts at initial) ---
    print("Scenario 2: Model Y ties Model X in Coding (score_a=0.5)")
    # model_y starts at initial_rating for coding
    new_x_code, new_y_code = update_elo(current_ratings, "model_x", "model_y", "coding", 0.5)
    print(f"Updated Coding Ratings: X={new_x_code:.1f}, Y={new_y_code:.1f}")
    print("Current Ratings Structure:")
    print(json.dumps(current_ratings, indent=2))
    print("-" * 30)

     # --- Scenario 3: New Model Z loses to Model Y in Math ---
    print("Scenario 3: New Model Z loses to Model Y in Math (score_a=0.0 for Z vs Y)")
    # model_z starts at initial_rating for math
    new_z_math, new_y_math_2 = update_elo(current_ratings, "model_z", "model_y", "math", 0.0)
    print(f"Updated Math Ratings: Z={new_z_math:.1f}, Y={new_y_math_2:.1f}")
    print("Current Ratings Structure:")
    print(json.dumps(current_ratings, indent=2))
    print("-" * 30)

     # --- Scenario 4: Invalid Score ---
    print("Scenario 4: Invalid score")
    res_a, res_b = update_elo(current_ratings, "model_x", "model_y", "math", 0.7)
    print(f"Result with invalid score: {res_a}, {res_b}")
    print("Ratings Structure (should be unchanged from last valid update):")
    print(json.dumps(current_ratings, indent=2))
    print("-" * 30)
