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
            model_entry = model_ratings.setdefault(model_name, {
            'elo_rating_by_category': {},  
            'num_comparisons': 0,          
            'wins': 0,                     
            'losses': 0,                   
            'draws': 0,                    
            'comparison_counts_by_category': {} 
            })
            model_entry['elo_rating_by_category'].setdefault(cat_str, initial_rating)
            model_entry['comparison_counts_by_category'].setdefault(cat_str, {})

def _update_match_statistics( # Renamed for clarity as it updates both overall and category-specific
    model_ratings: ModelRatingsType,
    all_matches: List[MatchType]
) -> None:
    """
    Updates both overall and category-specific match statistics 
    (num_comparisons, wins, losses, draws) for each model based on all_matches.
    Assumes model_ratings has been initialized. Modifies `model_ratings` in place.
    """
    logger.debug("Updating all match statistics (overall and per category)...")
    # Reset all stats first to ensure accurate recounting
    

    for model_a_name, model_b_name, category, score_a in all_matches:
        cat_str = category.lower()

        if model_a_name not in model_ratings or model_b_name not in model_ratings:
            logger.warning(f"Statistics update: Model {model_a_name} or {model_b_name} not found. Skipping stats for this match.")
            continue
        
        # Ensure category exists in comparison_counts_by_category (should be by initialization)
        # This is an extra safeguard.
        for mn in [model_a_name, model_b_name]:
            if cat_str not in model_ratings[mn]['comparison_counts_by_category']:
                 model_ratings[mn]['comparison_counts_by_category'][cat_str] = {
                    'wins': 0, 'losses': 0, 'draws': 0, 'num_comparisons': 0
                }


        # Update overall stats
        model_ratings[model_a_name]['num_comparisons'] += 1
        model_ratings[model_b_name]['num_comparisons'] += 1

        # Update category-specific stats
        model_ratings[model_a_name]['comparison_counts_by_category'][cat_str]['num_comparisons'] += 1
        model_ratings[model_b_name]['comparison_counts_by_category'][cat_str]['num_comparisons'] += 1

        if score_a == 1.0:  # Model A wins
            model_ratings[model_a_name]['wins'] += 1
            model_ratings[model_b_name]['losses'] += 1
            model_ratings[model_a_name]['comparison_counts_by_category'][cat_str]['wins'] += 1
            model_ratings[model_b_name]['comparison_counts_by_category'][cat_str]['losses'] += 1
        elif score_a == 0.0:  # Model B wins
            model_ratings[model_a_name]['losses'] += 1
            model_ratings[model_b_name]['wins'] += 1
            model_ratings[model_a_name]['comparison_counts_by_category'][cat_str]['losses'] += 1
            model_ratings[model_b_name]['comparison_counts_by_category'][cat_str]['wins'] += 1
        elif score_a == 0.5:  # Draw
            model_ratings[model_a_name]['draws'] += 1
            model_ratings[model_b_name]['draws'] += 1
            model_ratings[model_a_name]['comparison_counts_by_category'][cat_str]['draws'] += 1
            model_ratings[model_b_name]['comparison_counts_by_category'][cat_str]['draws'] += 1
        else:
            logger.warning(f"Statistics update: Unknown score {score_a} for match. Not updating win/loss/draw stats for this outcome.")
    logger.debug("All match statistics update complete.")

# --- Core m-ELO Functions (Adapted) ---
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
    Adapted to use model_ratings[model_name]['elo_rating_by_category'][cat_str].
    """
    total_log_likelihood = 0.0
    epsilon = 1e-9 # To prevent log(0)

    for model_a_name, model_b_name, category, score_a in all_matches:
        cat_str = category.lower()
        try:
            # Access ELO ratings using the adapted structure
            rating_a = model_ratings[model_a_name]['elo_rating_by_category'][cat_str]
            rating_b = model_ratings[model_b_name]['elo_rating_by_category'][cat_str]
        except KeyError:
            logger.error(f"m-ELO LogLikelihood: Rating not found for {model_a_name} or {model_b_name} in category '{cat_str}'. Skipping match.")
            continue

        prob_a_wins = _calculate_expected_score(rating_a, rating_b)
        prob_b_wins = 1.0 - prob_a_wins 

        score_b = 1.0 - score_a

        term_a = score_a * math.log(prob_a_wins + epsilon)
        term_b = score_b * math.log(prob_b_wins + epsilon)
        total_log_likelihood += term_a + term_b
        
    return total_log_likelihood

def calculate_mELO_ratings_by_gradient_descent(
    model_ratings: ModelRatingsType,
    all_matches: List[MatchType],
    initial_rating: float, # Used by _initialize_ratings_for_mELO_structure
    learning_rate: float,
    epochs: int
) -> None:
    """
    Calculates m-ELO ratings by optimizing the log-likelihood function using gradient descent.
    Modifies `model_ratings` in place.
    Based on Section 4.1 of the paper (m-ELO).
    Adapted to use model_ratings[model_name]['elo_rating_by_category'][cat_str].
    """
    # Initialize or verify the structure of model_ratings
    _initialize_ratings_for_mELO(model_ratings, all_matches, initial_rating)
    _update_match_statistics(model_ratings, all_matches)


    logger.info(f"Starting m-ELO calculation: {epochs} epochs, learning rate {learning_rate}")

    for epoch in range(epochs):
        # Gradients structure: {model_name: {category_str: gradient_value}}
        gradients: DefaultDict[str, DefaultDict[str, float]] = \
            defaultdict(lambda: defaultdict(float))

        # Calculate gradients based on all matches
        for model_a_name, model_b_name, category, score_a in all_matches:
            cat_str = category.lower()
            
            try:
                # Access ELO ratings using the adapted structure
                rating_a = model_ratings[model_a_name]['elo_rating_by_category'][cat_str]
                rating_b = model_ratings[model_b_name]['elo_rating_by_category'][cat_str]
            except KeyError:
                logger.error(f"m-ELO Gradient: Rating not found for '{model_a_name}' or '{model_b_name}' in category '{cat_str}' during epoch {epoch}. This is unexpected if initialization was correct.")
                continue

            prob_a_wins = _calculate_expected_score(rating_a, rating_b)
            score_b = 1.0 - score_a 

            # Gradient contribution from this match (Equation 3: C * (W_ij - P(R_i, R_j)))
            # For model A:
            grad_contrib_a = C_ELO_CONSTANT * (score_a - prob_a_wins)
            gradients[model_a_name][cat_str] += grad_contrib_a

            # For model B:
            # W_ji is score_b, P(R_j, R_i) is (1.0 - prob_a_wins)
            grad_contrib_b = C_ELO_CONSTANT * (score_b - (1.0 - prob_a_wins))
            gradients[model_b_name][cat_str] += grad_contrib_b

        # Update ratings using the accumulated gradients
        for model_name, category_grads in gradients.items():
            for cat_str, grad_value in category_grads.items():
                # Update ELO ratings in the adapted structure
                model_ratings[model_name]['elo_rating_by_category'][cat_str] += learning_rate * grad_value
        
        if (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0 or epoch == epochs -1 :
            current_log_likelihood = calculate_log_likelihood_mELO(model_ratings, all_matches)
            logger.info(f"m-ELO Epoch {epoch+1}/{epochs} completed. Log-Likelihood: {current_log_likelihood:.4f}")