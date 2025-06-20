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


MatchType: TypeAlias = Tuple[str, str, str, float] # (model_a, model_b, category, score_a)

def _initialize_ratings_for_mELO(
    model_ratings: ModelRatingsType,
    all_matches: List[MatchType],
    initial_rating: float
):
    """
    Stellt sicher, dass alle Modelle und Kategorien, die in den Matches vorkommen,
    in model_ratings initialisiert sind.
    Modifiziert model_ratings direkt (in place).
    Die Kategoriestatistiken werden jetzt mit der vollen Zählerstruktur initialisiert.
    """
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
    Update Match statistics (Total and for each category).
    """
    logger.debug(f"Updating match statistics based on {len(current_matches_being_processed)} matches...")


    models_and_categories_in_current_batch = defaultdict(set)
    for m_a, m_b, cat, _ in current_matches_being_processed:
        cat_l = cat.lower()
        models_and_categories_in_current_batch[m_a].add(cat_l)
        models_and_categories_in_current_batch[m_b].add(cat_l)

    for model_name, categories_in_batch in models_and_categories_in_current_batch.items():
        if model_name in model_ratings:
            for cat_l in categories_in_batch:
                if cat_l in model_ratings[model_name]['comparison_counts_by_category']:
                    model_ratings[model_name]['comparison_counts_by_category'][cat_l] = {
                        'wins': 0, 'losses': 0, 'draws': 0, 'num_comparisons': 0
                    }
                else:
                    logger.warning(f"Category {cat_l} for model {model_name} not in comparison_counts_by_category found "
                                   f"while reset. Initialize.")
                    model_ratings[model_name]['comparison_counts_by_category'][cat_l] = {
                        'wins': 0, 'losses': 0, 'draws': 0, 'num_comparisons': 0
                    }
        else:
            logger.warning(f"Model {model_name} not found in model_ratings, while resetting the Statistics.")


    for model_a_name, model_b_name, category, score_a in current_matches_being_processed:
        cat_str = category.lower()

        if model_a_name not in model_ratings or model_b_name not in model_ratings:
            logger.warning(f"Statistic-Update: Model {model_a_name} or {model_b_name} not found. "
                           f"Skip Statistics for this match.")
            continue
        
        stats_a = model_ratings[model_a_name]['comparison_counts_by_category'].get(cat_str)
        stats_b = model_ratings[model_b_name]['comparison_counts_by_category'].get(cat_str)

        if stats_a is None or stats_b is None:
            logger.error(f"Critical Error: Categorize found for {cat_str} not for {model_a_name} or {model_b_name} "
                         f"after resetting. Skip this match for the statistic.")
            continue


        stats_a['num_comparisons'] += 1
        stats_b['num_comparisons'] += 1

        if score_a == 1.0:  # Model A wins
            stats_a['wins'] += 1
            stats_b['losses'] += 1
        elif score_a == 0.0:  # Model B wins
            stats_a['losses'] += 1
            stats_b['wins'] += 1
        elif score_a == 0.5:  # Tie
            stats_a['draws'] += 1
            stats_b['draws'] += 1
        else:
            logger.warning(f"Statistic-Update: Unknown Score {score_a} for match. "
                           f"Win/Loss/Draw-Statistics for this result not updated.")

    for model_name, model_data in model_ratings.items():
        model_data['num_comparisons'] = 0
        model_data['wins'] = 0
        model_data['losses'] = 0
        model_data['draws'] = 0
        if 'comparison_counts_by_category' in model_data:
            for cat_stats_dict in model_data['comparison_counts_by_category'].values():
                model_data['num_comparisons'] += cat_stats_dict.get('num_comparisons', 0)
                model_data['wins'] += cat_stats_dict.get('wins', 0)
                model_data['losses'] += cat_stats_dict.get('losses', 0)
                model_data['draws'] += cat_stats_dict.get('draws', 0)
    logger.debug("Update Match-statistics finished.")



# --- ELO Calculation Logic ---
def _calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculates the expected score for player A against player B.
    This is P(A wins) using the standard Elo formula.
    P(A wins) = 1 / (1 + 10^((rating_b - rating_a) / 400))
    This is equivalent to P(R_A, R_B) = 1 / (1 + exp(-C_ELO_CONSTANT * (rating_a - rating_b)))
    """
    return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))


def update_elo_iteratively(
    model_ratings: ModelRatingsType,
    all_matches: List[MatchType],
    k_factor: int,
    initial_rating: float
) -> None:
    """
    Updates ELO ratings for all matches iteratively.
    This is the traditional Elo approach, applied one match at a time.
    The function modifies `model_ratings` in place.

    Args:
        model_ratings: The dictionary containing all current model ratings.
        all_matches: A list of all matches to process.
        k_factor: The K-factor determining rating change sensitivity.
        initial_rating: The rating assigned if a model/category doesn't exist yet.
    """
    _initialize_ratings(model_ratings, all_matches, initial_rating)

    logger.info(f"Starting iterative ELO update for {len(all_matches)} matches...")
    for model_a, model_b, category, score_a in all_matches:
        if not all([model_a, model_b, category]) or score_a not in [0.0, 0.5, 1.0]:
            logger.warning(f"Skipping invalid match data: {(model_a, model_b, category, score_a)}")
            continue

        cat_str = category.lower()

        rating_a = model_ratings[model_a]['elo_rating_by_category'][cat_str]
        rating_b = model_ratings[model_b]['elo_rating_by_category'][cat_str]

        expected_a = _calculate_expected_score(rating_a, rating_b)
        score_b = 1.0 - score_a
        expected_b = 1.0 - expected_a

        new_rating_a = rating_a + k_factor * (score_a - expected_a)
        new_rating_b = rating_b + k_factor * (score_b - expected_b)

        model_ratings[model_a]['elo_rating_by_category'][cat_str] = new_rating_a
        model_ratings[model_b]['elo_rating_by_category'][cat_str] = new_rating_b

    logger.info("Iterative ELO updates complete.")
    # Update all statistics once at the end
    _update_match_statistics(model_ratings, all_matches)



# --- Core m-ELO Functions ---

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
    epsilon = 1e-9 

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
    initial_rating: float,
    learning_rate: float,
    epochs: int
) -> None:
    """
    Calculates m-ELO ratings by optimizing the log-likelihood function using gradient descent.
    Modifies `model_ratings` in place.
    Based on Section 4.1 of the paper (m-ELO).
    Adapted to use model_ratings[model_name]['elo_rating_by_category'][cat_str].
    """
    _initialize_ratings_for_mELO(model_ratings, all_matches, initial_rating)


    logger.info(f"Starting m-ELO calculation: {epochs} epochs, learning rate {learning_rate}")

    for epoch in range(epochs):
        gradients: DefaultDict[str, DefaultDict[str, float]] = \
            defaultdict(lambda: defaultdict(float))

        for model_a_name, model_b_name, category, score_a in all_matches:
            cat_str = category.lower()
            
            try:
                rating_a = model_ratings[model_a_name]['elo_rating_by_category'][cat_str]
                rating_b = model_ratings[model_b_name]['elo_rating_by_category'][cat_str]
            except KeyError:
                logger.error(f"m-ELO Gradient: Rating not found for '{model_a_name}' or '{model_b_name}' "
                               f"in category '{cat_str}' during epoch {epoch}. This is unexpected.")
                continue

            prob_a_wins = _calculate_expected_score(rating_a, rating_b)
            score_b = 1.0 - score_a

            grad_contrib_a = C_ELO_CONSTANT * (score_a - prob_a_wins)
            gradients[model_a_name][cat_str] += grad_contrib_a

            grad_contrib_b = C_ELO_CONSTANT * (score_b - (1.0 - prob_a_wins))
            gradients[model_b_name][cat_str] += grad_contrib_b

        for model_name, category_grads in gradients.items():
            for cat_str, grad_value in category_grads.items():
                if model_name in model_ratings and cat_str in model_ratings[model_name]['elo_rating_by_category']:
                    model_ratings[model_name]['elo_rating_by_category'][cat_str] += learning_rate * grad_value
                else:
                    logger.error(f"m-ELO Gradient Update: Path for {model_name} / {cat_str} not found in model_ratings.")
        
        if (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0 or epoch == epochs -1 :
            current_log_likelihood = calculate_log_likelihood_mELO(model_ratings, all_matches) 
            logger.info(f"m-ELO Epoch {epoch+1}/{epochs} completed. Log-Likelihood: {current_log_likelihood:.4f}")
            
    _update_match_statistics(model_ratings, all_matches)









