# lottery_handler/utils.py
"""
Utility functions for lottery handler
"""

import logging
import sys
import os
import warnings
from typing import Dict, Any

def setup_logging():
    """
    Optimalizált logging konfiguráció a lottószám predikciókhoz - Windows kompatibilis
    """
    
    # Windows-kompatibilis formázó (emoji nélkül)
    formatter = logging.Formatter(
        '[LOTTERY] %(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler UTF-8 encoding-gal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Encoding beállítás Windows-ra
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except:
            pass
    
    # Root logger konfigurálása
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    # Külső könyvtárak logging szintjének csökkentése
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('mlxtend').setLevel(logging.ERROR)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)
    
    return logging.getLogger('lottery_handler')


def suppress_warnings():
    """
    Nem kritikus figyelmeztetések elnémítása
    """
    
    # TensorFlow figyelmeztetések
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    
    # Python warnings szűrése
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='mlxtend')
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
    warnings.filterwarnings('ignore', message='DataFrames with non-bool types*')
    warnings.filterwarnings('ignore', message='oneDNN custom operations*')
    
    # TensorFlow specifikus elnémítás
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
    except ImportError:
        pass


def get_algorithm_display_name(algorithm_name: str) -> str:
    """
    Algoritmus nevek szebb megjelenítése - Windows kompatibilis
    """
    
    display_names = {
        'quantum_machine_learning_prediction': '[QUANTUM] Kvantum ML',
        'advanced_time_series_deep_learning_prediction': '[DEEP] Deep Learning Idosor',
        'deep_reinforcement_learning_predictor': '[RL] Deep RL',
        'chaos_theory_fractal_predictor': '[CHAOS] Kaosz & Fraktal',
        'swarm_intelligence_predictor': '[SWARM] Raj Intelligencia',
        'adaptive_neuro_fuzzy_predictor': '[FUZZY] Neuro-Fuzzy',
        'advanced_statistical_ensemble_predictor': '[STAT] Statisztikai Ensemble',
        'advanced_hybrid_intelligent_prediction': '[HYBRID] Hibrid AI',
        'most_frequent_numbers_prediction': '[FREQ] Leggyakoribb',
        'neural_network_prediction': '[NN] Neuralis Halo',
        'lstm_neural_network_prediction': '[LSTM] LSTM',
        'ensemble_prediction': '[ENS] Ensemble',
        'genetic_algorithm_prediction': '[GA] Genetikus',
        'markov_chain_prediction': '[MARKOV] Markov',
        'monte_carlo_simulation_prediction': '[MC] Monte Carlo',
        'quantum_rng_simulation': '[QR] Kvantum RNG',
        'linear_regression_prediction': '[LR] Linearis Regresszio',
        'decision_tree_prediction': '[DT] Dontesi Fa',
        'k_nearest_neighbors_prediction': '[KNN] K-Nearest',
        'bayesian_inference_prediction': '[BAYES] Bayesian',
        'hidden_markov_model_prediction': '[HMM] Hidden Markov'
    }
    
    return display_names.get(algorithm_name, f"[ALG] {algorithm_name}")


def log_algorithm_performance(algorithm_name: str, score: float, execution_time: float, 
                            predicted_main: list, predicted_additional: list = None) -> None:
    """
    Algoritmus teljesítmény naplózása szép formátumban - Windows kompatibilis
    """
    
    logger = logging.getLogger('lottery_handler')
    
    display_name = get_algorithm_display_name(algorithm_name)
    
    # Főszámok formázása
    main_str = ", ".join(map(str, predicted_main[:5])) if predicted_main else "N/A"
    
    # Kiegészítő számok formázása
    additional_str = ""
    if predicted_additional:
        additional_str = f" + [{', '.join(map(str, predicted_additional))}]"
    
    # Teljesítmény kategorizálás
    if score >= 50:
        performance_level = "KIVALO"
    elif score >= 25:
        performance_level = "JO"
    elif score >= 10:
        performance_level = "KOZEPES"
    else:
        performance_level = "ALAP"
    
    # Windows-kompatibilis naplózás
    logger.info(f"=== {display_name} - {performance_level} ===")
    logger.info(f"Pontszam: {score:.2f} | Ido: {execution_time:.2f}ms")
    logger.info(f"Szamok: [{main_str}]{additional_str}")
    logger.info(f"=" * 50)


def format_algorithm_summary(algorithms: list) -> str:
    """
    Algoritmusok összefoglalójának formázása - Windows kompatibilis
    """
    
    if not algorithms:
        return "NINCSENEK EREDMENYEK"
    
    total_algorithms = len(algorithms)
    successful_algorithms = len([a for a in algorithms if a.get('success', True)])
    
    best_algorithm = max(algorithms, key=lambda x: x.get('score', 0))
    average_score = sum(a.get('score', 0) for a in algorithms) / len(algorithms)
    
    summary = f"""
ALGORITMUS TESZT OSSZEFOGLALO:
Osszes algoritmus: {total_algorithms}
Sikeres futtatasok: {successful_algorithms}
Legjobb: {get_algorithm_display_name(best_algorithm.get('name', 'N/A'))} ({best_algorithm.get('score', 0):.2f} pont)
Atlagos pontszam: {average_score:.2f}
"""
    
    return summary
