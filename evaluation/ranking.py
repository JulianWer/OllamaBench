import json
import logging
from typing import Any, Dict, Tuple, TypeAlias, Optional, List, DefaultDict
import math
from collections import defaultdict
import numpy as np 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Type Alias for Model Ratings Structure ---
ModelRatingsType: TypeAlias = Dict[str, Dict[str, Any]]
MatchType: TypeAlias = Tuple[str, str, str, float] # (model_a, model_b, category, score_a)

# --- Constants ---
C_ELO_CONSTANT = math.log(10) / 400

# --- Optimized ELO Calculation Logic ---

def _initialize_ratings_for_mELO(
    model_ratings: ModelRatingsType,
    all_matches: List[MatchType],
    initial_rating: float
):
    """Initializes the data structure for all models and categories found in the matches."""
    logger.debug("Initializing ratings structure for mELO...")
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
            model_entry['comparison_counts_by_category'].setdefault(cat_str, {
                'wins': 0, 'losses': 0, 'draws': 0, 'num_comparisons': 0
            })
    logger.debug("Rating structure initialization complete.")


def _update_match_statistics(
    model_ratings: ModelRatingsType,
    current_matches_being_processed: List[MatchType]
) -> None:
    """
    Update (incremental) game statistics based on the list of matches. 
    This function does not reset any statistics and prevents data loss.
    """
    logger.debug(f"Incremental statistics with {len(current_matches_being_processed)} new matches...")

    for model_a, model_b, category, score_a in current_matches_being_processed:
        cat_str = category.lower()

        model_a_data = model_ratings[model_a]
        model_b_data = model_ratings[model_b]

        stats_a = model_a_data['comparison_counts_by_category'][cat_str]
        stats_b = model_b_data['comparison_counts_by_category'][cat_str]

        model_a_data['num_comparisons'] += 1
        model_b_data['num_comparisons'] += 1
        stats_a['num_comparisons'] += 1
        stats_b['num_comparisons'] += 1

        if score_a == 1.0:  # Model A wins
            model_a_data['wins'] += 1
            stats_a['wins'] += 1
            model_b_data['losses'] += 1
            stats_b['losses'] += 1
        elif score_a == 0.0:  # Model B wins
            model_b_data['wins'] += 1
            stats_b['wins'] += 1
            model_a_data['losses'] += 1
            stats_a['losses'] += 1
        else:  # Tie
            model_a_data['draws'] += 1
            stats_a['draws'] += 1
            model_b_data['draws'] += 1
            stats_b['draws'] += 1

    logger.debug("Update statistics finished.")

def calculate_mELO_ratings(
    model_ratings: ModelRatingsType,
    all_matches: List[MatchType],
    initial_rating: float,
    learning_rate: float,
    epochs: int
) -> None:
    """
    Calculates m-ELO ratings using NumPy for vectorized gradient descent.
    This is significantly faster for large numbers of matches and epochs.
    """
    _initialize_ratings_for_mELO(model_ratings, all_matches, initial_rating)

    logger.info(f"Starting m-ELO calculation with NumPy: {epochs} epochs, learning rate {learning_rate}")

    # Create mappings from model/category names to integer indices for NumPy arrays
    models = sorted(list(model_ratings.keys()))
    model_to_idx = {name: i for i, name in enumerate(models)}
    
    categories_from_matches = set(m[2].lower() for m in all_matches)
    categories_from_ratings = set()
    for model_data in model_ratings.values():
        categories_from_ratings.update(model_data.get('elo_rating_by_category', {}).keys())
    all_categories = sorted(list(categories_from_matches.union(categories_from_ratings)))

    category_to_idx = {name: i for i, name in enumerate(all_categories)}
    
    num_models = len(models)
    num_categories = len(all_categories)

    # Initialize ratings array
    ratings_arr = np.full((num_models, num_categories), initial_rating, dtype=np.float64)
    for model_name, data in model_ratings.items():
        for cat_name, rating in data['elo_rating_by_category'].items():
            ratings_arr[model_to_idx[model_name], category_to_idx[cat_name]] = rating

    # Prepare match data as NumPy arrays
    model_a_indices = np.array([model_to_idx[m[0]] for m in all_matches], dtype=np.int32)
    model_b_indices = np.array([model_to_idx[m[1]] for m in all_matches], dtype=np.int32)
    category_indices = np.array([category_to_idx[m[2].lower()] for m in all_matches], dtype=np.int32)
    scores = np.array([m[3] for m in all_matches], dtype=np.float64)

    # --- Main Gradient Descent Loop ---
    for epoch in range(epochs):
        # 1. Get current ratings for all matches at once
        ratings_a = ratings_arr[model_a_indices, category_indices]
        ratings_b = ratings_arr[model_b_indices, category_indices]

        # 2. Vectorized calculation of expected scores (sigmoid function)
        expected_scores = 1.0 / (1.0 + np.exp(C_ELO_CONSTANT * (ratings_b - ratings_a)))

        # 3. Vectorized calculation of gradient contributions
        grad_contributions = C_ELO_CONSTANT * (scores - expected_scores)

        # 4. Update gradients for all models and categories
        gradients = np.zeros_like(ratings_arr)
        np.add.at(gradients, (model_a_indices, category_indices), grad_contributions)
        np.add.at(gradients, (model_b_indices, category_indices), -grad_contributions) # Gradient for B is opposite of A

        # 5. Update ratings
        ratings_arr += learning_rate * gradients
        
        if (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0 or epoch == epochs - 1:
            logger.info(f"NumPy m-ELO Epoch {epoch+1}/{epochs} completed.")

    # Write the optimized ratings back to the original dictionary structure
    for model_name, model_idx in model_to_idx.items():
        for cat_name, cat_idx in category_to_idx.items():
            model_ratings[model_name]['elo_rating_by_category'][cat_name] = ratings_arr[model_idx, cat_idx]

    _update_match_statistics(model_ratings, all_matches)
