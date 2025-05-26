import json
import logging
from typing import Dict, Tuple, TypeAlias, Optional, List, DefaultDict
import math
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Type Alias for Model Ratings Structure ---
# Defines the expected nested dictionary structure for clarity.
ModelRatingsType: TypeAlias = Dict[str, # Model Name (e.g., "llama3:latest")
                                  Dict[str, # Should always be the key "categorie"
                                       Dict[str, float] # Category Name (e.g., "math") -> ELO Rating (float)
                                      ]
                                 ]

# --- Constants ---
CATEGORY_DICT_KEY = "categorie" # The fixed key used in the dictionary structure
C_ELO_CONSTANT = math.log(10) / 400 # Constant C from the paper's probability formula, derived from standard Elo

# --- ELO Calculation Logic (Shared and Traditional) ---
def _calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculates the expected score for player A against player B.
    This is P(A wins) using the standard Elo formula.
    P(A wins) = 1 / (1 + 10^((rating_b - rating_a) / 400))
    This is equivalent to P(R_A, R_B) = 1 / (1 + exp(-C_ELO_CONSTANT * (rating_a - rating_b)))
    """
    return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

def update_elo(
    model_ratings: ModelRatingsType,
    model_a: str,
    model_b: str,
    category: str,
    score_a: float, # 1.0 (A wins), 0.0 (B wins), 0.5 (Tie)
    k_factor: int,
    initial_rating: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    Updates the ELO ratings for two models within a specific category based on a single comparison result (Traditional Iterative Elo).
    Modifies the `model_ratings` dictionary *in place*.

    Args:
        model_ratings: The dictionary containing all current model ratings.
        model_a: Name of the first model.
        model_b: Name of the second model.
        category: The category string (e.g., "math", "coding").
        score_a: Outcome score for model A (1.0 for win, 0.5 for tie, 0.0 for loss).
        k_factor: The K-factor determining rating change sensitivity.
        initial_rating: The rating assigned if a model/category doesn't exist yet.

    Returns:
        A tuple containing the new ELO ratings (new_rating_a, new_rating_b) after the update.
        Returns (None, None) if the input is invalid.
    """
    if not model_a or not model_b:
        logger.error("Traditional ELO update failed: Model names cannot be empty.")
        return None, None
    if not category:
        logger.error("Traditional ELO update failed: Category string cannot be empty.")
        return None, None
    if score_a not in [0.0, 0.5, 1.0]:
        logger.error(f"Traditional ELO update failed: Invalid score_a value '{score_a}'. Must be 0.0, 0.5, or 1.0.")
        return None, None

    category_str = category.lower()

    ratings_a_all = model_ratings.setdefault(model_a, {CATEGORY_DICT_KEY: {}})
    ratings_a_cat = ratings_a_all.setdefault(CATEGORY_DICT_KEY, {})
    rating_a = ratings_a_cat.setdefault(category_str, initial_rating)

    ratings_b_all = model_ratings.setdefault(model_b, {CATEGORY_DICT_KEY: {}})
    ratings_b_cat = ratings_b_all.setdefault(CATEGORY_DICT_KEY, {})
    rating_b = ratings_b_cat.setdefault(category_str, initial_rating)

    logger.debug(f"Traditional ELO Pre-Update ({category_str}): {model_a}={rating_a:.1f}, {model_b}={rating_b:.1f}")

    expected_a = _calculate_expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a # Or _calculate_expected_score(rating_b, rating_a)

    score_b = 1.0 - score_a

    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * (score_b - expected_b)

    ratings_a_cat[category_str] = new_rating_a
    ratings_b_cat[category_str] = new_rating_b

    logger.info(f"Traditional ELO Update ({category_str}): {model_a} ({rating_a:.1f} -> {new_rating_a:.1f}), {model_b} ({rating_b:.1f} -> {new_rating_b:.1f}), Score A: {score_a}, K: {k_factor}")

    return new_rating_a, new_rating_b

# --- m-ELO Calculation Logic (Based on the paper's MLE approach) ---

# Match data type for m-ELO
MatchType: TypeAlias = Tuple[str, str, str, float] # (model_a, model_b, category, score_a)

def _initialize_ratings_for_mELO(
    model_ratings: ModelRatingsType,
    all_matches: List[MatchType],
    initial_rating: float
):
    """
    Ensures all models and categories appearing in matches are initialized in model_ratings.
    Modifies model_ratings in place.
    """
    for model_a, model_b, category, _ in all_matches:
        cat_str = category.lower()
        for model_name in [model_a, model_b]:
            model_entry = model_ratings.setdefault(model_name, {CATEGORY_DICT_KEY: {}})
            category_entry = model_entry.setdefault(CATEGORY_DICT_KEY, {})
            category_entry.setdefault(cat_str, initial_rating)

def calculate_log_likelihood_mELO(
    model_ratings: ModelRatingsType,
    all_matches: List[MatchType]
) -> float:
    """
    Calculates the total log-likelihood of the observed matches given the current model ratings.
    Formula based on Equation 2 from the paper:
    ln L = sum [ W_ij * ln P(R_i, R_j) + W_ji * ln P(R_j, R_i) ]
    where W_ij is score_a, W_ji is 1 - score_a.
    P(R_i, R_j) is the expected score of i against j.
    """
    total_log_likelihood = 0.0
    epsilon = 1e-9 # To prevent log(0)

    for model_a_name, model_b_name, category, score_a in all_matches:
        cat_str = category.lower()
        try:
            rating_a = model_ratings[model_a_name][CATEGORY_DICT_KEY][cat_str]
            rating_b = model_ratings[model_b_name][CATEGORY_DICT_KEY][cat_str]
        except KeyError:
            logger.error(f"m-ELO LogLikelihood: Rating not found for {model_a_name} or {model_b_name} in category {cat_str}. Skipping match.")
            continue

        prob_a_wins = _calculate_expected_score(rating_a, rating_b)
        prob_b_wins = 1.0 - prob_a_wins # = _calculate_expected_score(rating_b, rating_a)

        score_b = 1.0 - score_a

        # Handle ties (score_a = 0.5) carefully for log-likelihood.
        # The paper's formula W_ij ln P(R_i,R_j) + W_ji ln P(R_j,R_i) assumes W_ij is 0 or 1.
        # For ties, a common approach in Bradley-Terry-Elo models is to treat it as half a win for each.
        # If score_a is 0.5, then W_ij = 0.5, W_ji = 0.5.
        # term = 0.5 * log(P_a) + 0.5 * log(P_b)
        # However, the paper's gradient (Eq 3) C(W_nj - P(R_n, R_j)) works directly with W_nj = 0.5.
        # For simplicity in log-likelihood, we'll use the direct interpretation:
        # If A wins (score_a=1): log(P(A wins))
        # If B wins (score_a=0): log(P(B wins))
        # If Tie (score_a=0.5): 0.5 * log(P(A wins)) + 0.5 * log(P(B wins))
        # This is equivalent to: score_a * log(P(A wins)) + score_b * log(P(B wins))

        term_a = score_a * math.log(prob_a_wins + epsilon)
        term_b = score_b * math.log(prob_b_wins + epsilon)
        total_log_likelihood += term_a + term_b
        
    return total_log_likelihood

def calculate_mELO_ratings_by_gradient_descent(
    model_ratings: ModelRatingsType,
    all_matches: List[MatchType],
    initial_rating: float,
    learning_rate: float,
    epochs: int
) -> None:
    """
    Calculates m-ELO ratings by optimizing the log-likelihood function using gradient descent.
    Modifies `model_ratings` in place.
    Based on Section 4.1 of the paper (m-ELO).
    """
    _initialize_ratings_for_mELO(model_ratings, all_matches, initial_rating)

    logger.info(f"Starting m-ELO calculation: {epochs} epochs, learning rate {learning_rate}")

    for epoch in range(epochs):
        # Initialize gradients for this epoch
        # Structure: {model_name: {CATEGORY_DICT_KEY: {category: gradient_value}}}
        gradients: DefaultDict[str, DefaultDict[str, DefaultDict[str, float]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # Calculate gradients based on all matches
        for model_a_name, model_b_name, category, score_a in all_matches:
            cat_str = category.lower()
            
            # Ensure ratings exist (should be guaranteed by _initialize_ratings_for_mELO)
            try:
                rating_a = model_ratings[model_a_name][CATEGORY_DICT_KEY][cat_str]
                rating_b = model_ratings[model_b_name][CATEGORY_DICT_KEY][cat_str]
            except KeyError:
                # This should not happen if initialization is correct
                logger.error(f"m-ELO Gradient: Rating not found for {model_a_name} or {model_b_name} in cat {cat_str} during epoch {epoch}. This is unexpected.")
                continue


            prob_a_wins = _calculate_expected_score(rating_a, rating_b)
            # prob_b_wins = _calculate_expected_score(rating_b, rating_a) # which is 1.0 - prob_a_wins

            score_b = 1.0 - score_a # Score for model B

            # Gradient contribution from this match (Equation 3: C * (W_ij - P(R_i, R_j)))
            # For model A:
            grad_contrib_a = C_ELO_CONSTANT * (score_a - prob_a_wins)
            gradients[model_a_name][CATEGORY_DICT_KEY][cat_str] += grad_contrib_a

            # For model B:
            # Here, W_ji is score_b, and P(R_j, R_i) is prob_b_wins
            # prob_b_wins = 1 - prob_a_wins
            grad_contrib_b = C_ELO_CONSTANT * (score_b - (1.0 - prob_a_wins)) # score_b - prob_b_wins
            gradients[model_b_name][CATEGORY_DICT_KEY][cat_str] += grad_contrib_b

        # Update ratings using the accumulated gradients
        for model_name, categories_data in gradients.items():
            for cat_dict_key, category_grads in categories_data.items(): # cat_dict_key is CATEGORY_DICT_KEY
                for cat_str, grad_value in category_grads.items():
                    model_ratings[model_name][cat_dict_key][cat_str] += learning_rate * grad_value
        
        if (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0:
            current_log_likelihood = calculate_log_likelihood_mELO(model_ratings, all_matches)
            logger.info(f"m-ELO Epoch {epoch+1}/{epochs} completed. Log-Likelihood: {current_log_likelihood:.4f}")

    logger.info("m-ELO calculation finished.")


# --- Example Usage ---
if __name__ == "__main__":
    model_ratings_store: ModelRatingsType = {}
    initial_elo = 1000.0
    k = 32 # For traditional Elo

    # --- Traditional Elo Example ---
    logger.info("\n--- Traditional Iterative ELO Example ---")
    update_elo(model_ratings_store, "model_A", "model_B", "coding", 1.0, k, initial_elo) # A wins
    update_elo(model_ratings_store, "model_A", "model_C", "coding", 0.5, k, initial_elo) # A ties C
    update_elo(model_ratings_store, "model_B", "model_C", "coding", 0.0, k, initial_elo) # B loses to C (C wins)
    update_elo(model_ratings_store, "model_A", "model_B", "math", 0.0, k, initial_elo)   # A loses to B

    logger.info(f"Final Traditional ELO Ratings: {json.dumps(model_ratings_store, indent=2)}")

    # --- m-ELO Example ---
    logger.info("\n--- m-ELO (MLE Batch) Example ---")
    # Reset ratings for a clean m-ELO calculation or use a new store
    mELO_model_ratings_store: ModelRatingsType = {}
    
    # Sample match data: (model_a, model_b, category, score_a)
    # score_a = 1.0 if A wins, 0.5 if Tie, 0.0 if A loses (B wins)
    sample_matches: List[MatchType] = [
        ("model_X", "model_Y", "general", 1.0), # X beats Y
        ("model_X", "model_Y", "general", 1.0), # X beats Y again
        ("model_X", "model_Z", "general", 0.5), # X ties Z
        ("model_Y", "model_Z", "general", 0.0), # Y loses to Z
        ("model_X", "model_Y", "special", 0.0), # X loses to Y in special
        ("model_Y", "model_Z", "special", 1.0), # Y beats Z in special
        # Add more matches for better results
        ("model_X", "model_Y", "general", 1.0),
        ("model_X", "model_Z", "general", 1.0),
        ("model_Y", "model_Z", "general", 0.5),
        ("model_Z", "model_W", "general", 1.0), # Z beats W
        ("model_X", "model_W", "general", 1.0), # X beats W
    ]
    
    # Add more diverse matches for a more meaningful m-ELO run
    for _ in range(5): # X is generally stronger
        sample_matches.append(("model_X", "model_Y", "general", 1.0))
        sample_matches.append(("model_X", "model_Z", "general", 1.0))
        sample_matches.append(("model_X", "model_W", "general", 1.0))
    for _ in range(2): # Z is decent
        sample_matches.append(("model_Z", "model_Y", "general", 1.0))
        sample_matches.append(("model_Z", "model_W", "general", 0.5))
    for _ in range(1): # Y sometimes beats W
        sample_matches.append(("model_Y", "model_W", "general", 1.0))


    mELO_learning_rate = 0.5 
    mELO_epochs = 2000       # Number of iterations for gradient descent


    calculate_mELO_ratings_by_gradient_descent(
        mELO_model_ratings_store,
        sample_matches,
        initial_rating=initial_elo,
        learning_rate=10.0, # This needs careful tuning based on data scale and number of matches
        epochs=mELO_epochs
    )

    logger.info(f"Final m-ELO Ratings: {json.dumps(mELO_model_ratings_store, indent=2)}")

    # To verify, one could check if model_X > model_Z > model_Y > model_W in general category
    # And model_Y > model_X in special category
    if "model_X" in mELO_model_ratings_store and "general" in mELO_model_ratings_store["model_X"][CATEGORY_DICT_KEY]:
        logger.info(f"Example m-ELO rating for model_X (general): {mELO_model_ratings_store['model_X'][CATEGORY_DICT_KEY]['general']:.2f}")
    if "model_Y" in mELO_model_ratings_store and "general" in mELO_model_ratings_store["model_Y"][CATEGORY_DICT_KEY]:
        logger.info(f"Example m-ELO rating for model_Y (general): {mELO_model_ratings_store['model_Y'][CATEGORY_DICT_KEY]['general']:.2f}")
    if "model_Z" in mELO_model_ratings_store and "general" in mELO_model_ratings_store["model_Z"][CATEGORY_DICT_KEY]:
        logger.info(f"Example m-ELO rating for model_Z (general): {mELO_model_ratings_store['model_Z'][CATEGORY_DICT_KEY]['general']:.2f}")
    if "model_W" in mELO_model_ratings_store and "general" in mELO_model_ratings_store["model_W"][CATEGORY_DICT_KEY]:
        logger.info(f"Example m-ELO rating for model_W (general): {mELO_model_ratings_store['model_W'][CATEGORY_DICT_KEY]['general']:.2f}")

    if "model_X" in mELO_model_ratings_store and "special" in mELO_model_ratings_store["model_X"][CATEGORY_DICT_KEY]:
        logger.info(f"Example m-ELO rating for model_X (special): {mELO_model_ratings_store['model_X'][CATEGORY_DICT_KEY]['special']:.2f}")
    if "model_Y" in mELO_model_ratings_store and "special" in mELO_model_ratings_store["model_Y"][CATEGORY_DICT_KEY]:
        logger.info(f"Example m-ELO rating for model_Y (special): {mELO_model_ratings_store['model_Y'][CATEGORY_DICT_KEY]['special']:.2f}")

