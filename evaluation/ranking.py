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
                    logger.warning(f"Kategorie {cat_l} für Modell {model_name} nicht in comparison_counts_by_category gefunden "
                                   f"während des Zurücksetzens. Initialisiere sie jetzt.")
                    model_ratings[model_name]['comparison_counts_by_category'][cat_l] = {
                        'wins': 0, 'losses': 0, 'draws': 0, 'num_comparisons': 0
                    }
        else:
            logger.warning(f"Modell {model_name} nicht in model_ratings gefunden während des Zurücksetzens der Kategoriestatistiken.")


    for model_a_name, model_b_name, category, score_a in current_matches_being_processed:
        cat_str = category.lower()

        if model_a_name not in model_ratings or model_b_name not in model_ratings:
            logger.warning(f"Statistic-Update: Model {model_a_name} or {model_b_name} not found. "
                           f"Überspringe Statistiken für dieses Match.")
            continue
        
        stats_a = model_ratings[model_a_name]['comparison_counts_by_category'].get(cat_str)
        stats_b = model_ratings[model_b_name]['comparison_counts_by_category'].get(cat_str)

        if stats_a is None or stats_b is None:
            logger.error(f"Kritischer Fehler: Kategoriestatistiken für {cat_str} nicht für {model_a_name} oder {model_b_name} gefunden "
                         f"nach dem Zurücksetz-Schritt. Überspringe dieses Match für die Statistik.")
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
            logger.warning(f"Statistik-Update: Unbekannter Score {score_a} für Match. "
                           f"Win/Loss/Draw-Statistiken für dieses Ergebnis nicht aktualisiert.")

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
    logger.debug("Aktualisierung der Match-Statistiken abgeschlossen.")



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
    _update_match_statistics(model_ratings, all_matches)


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
                # Log, wenn eine Kategorie/Modell, das in all_matches ist, nicht in model_ratings gefunden wird.
                # _initialize_ratings_for_mELO sollte dies verhindern.
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








