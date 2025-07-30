# hyperparameter_optimized_prediction.py
"""
Hiperparaméter optimalizált lottószám predikció
Optuna-val optimalizálja a legjobb algoritmusok hiperparamétereit
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import logging

# Import optimization libraries
try:
    import optuna
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    import xgboost as xgb
    import lightgbm as lgb
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Hiperparaméter optimalizált predikció főfunkciója
    """
    try:
        if not OPTIMIZATION_AVAILABLE:
            logger.warning("Optimalizációs könyvtárak nem elérhetők, fallback módra váltás")
            return generate_simple_fallback(lottery_type_instance)
        
        # Fő számok generálása
        main_numbers = generate_optimized_numbers(
            lottery_type_instance,
            min_num=int(lottery_type_instance.min_number),
            max_num=int(lottery_type_instance.max_number),
            required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            is_main=True
        )
        
        # Kiegészítő számok generálása
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_optimized_numbers(
                lottery_type_instance,
                min_num=int(lottery_type_instance.additional_min_number),
                max_num=int(lottery_type_instance.additional_max_number),
                required_numbers=int(lottery_type_instance.additional_numbers_count),
                is_main=False
            )
        
        return main_numbers, additional_numbers
        
    except Exception as e:
        logger.error(f"Hiba az optimalizált predikcióban: {e}")
        return generate_simple_fallback(lottery_type_instance)


def generate_optimized_numbers(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    required_numbers: int,
    is_main: bool
) -> List[int]:
    """
    Optimalizált számgenerálás
    """
    # Történeti adatok lekérése
    past_draws = get_historical_data(lottery_type_instance, is_main)
    
    if len(past_draws) < 30:
        return generate_frequency_based_fallback(past_draws, min_num, max_num, required_numbers)
    
    try:
        # Egyszerűsített optimalizáció XGBoost-tal
        X, y = prepare_features(past_draws)
        
        if X.shape[0] < 10:
            return generate_frequency_based_fallback(past_draws, min_num, max_num, required_numbers)
        
        # Gyors hiperparaméter keresés
        best_params = quick_optimize_xgboost(X, y)
        
        # Modell tanítása
        models = []
        for pos in range(y.shape[1]):
            model = xgb.XGBRegressor(**best_params, random_state=42, verbosity=0)
            model.fit(X, y[:, pos])
            models.append(model)
        
        # Predikció
        last_features = extract_features_from_sequence(past_draws[-10:])
        X_pred = np.array([last_features])
        
        predictions = []
        for model in models:
            pred = model.predict(X_pred)[0]
            predictions.append(pred)
        
        # Feldolgozás
        processed_numbers = process_predictions(
            predictions, min_num, max_num, required_numbers, past_draws
        )
        
        return sorted(processed_numbers)
        
    except Exception as e:
        logger.error(f"Optimalizálás hiba: {e}")
        return generate_frequency_based_fallback(past_draws, min_num, max_num, required_numbers)


def prepare_features(past_draws: List[List[int]], lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Feature engineering a múltbeli húzásokból
    """
    if len(past_draws) < lookback + 1:
        return np.array([]), np.array([])
    
    X = []
    y = []
    
    for i in range(len(past_draws) - lookback):
        sequence = past_draws[i:i + lookback]
        features = extract_features_from_sequence(sequence)
        X.append(features)
        
        target = past_draws[i + lookback]
        y.append(target)
    
    return np.array(X), np.array(y)


def extract_features_from_sequence(sequence: List[List[int]]) -> List[float]:
    """
    Statisztikai jellemzők kinyerése
    """
    flat_numbers = [num for draw in sequence for num in draw]
    
    if not flat_numbers:
        return [0] * 15
    
    features = [
        np.mean(flat_numbers),
        np.median(flat_numbers),
        np.std(flat_numbers),
        np.min(flat_numbers),
        np.max(flat_numbers),
        len(set(flat_numbers)),
        sum(1 for x in flat_numbers if x % 2 == 0),
        sum(1 for x in flat_numbers if x % 2 == 1),
    ]
    
    # Gyakoriság jellemzők
    freq_dict = {}
    for num in flat_numbers:
        freq_dict[num] = freq_dict.get(num, 0) + 1
    
    if freq_dict:
        features.extend([
            max(freq_dict.values()),
            min(freq_dict.values()),
            len([v for v in freq_dict.values() if v > 1]),
        ])
    else:
        features.extend([0, 0, 0])
    
    # Összeg jellemzők
    draw_sums = [sum(draw) for draw in sequence]
    features.extend([
        np.mean(draw_sums),
        np.std(draw_sums) if len(draw_sums) > 1 else 0,
    ])
    
    # Tartomány jellemzők
    draw_ranges = [max(draw) - min(draw) for draw in sequence]
    features.extend([
        np.mean(draw_ranges),
        np.std(draw_ranges) if len(draw_ranges) > 1 else 0,
    ])
    
    return features


def quick_optimize_xgboost(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Gyors XGBoost optimalizáció
    """
    try:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            }
            
            scores = []
            for pos in range(min(3, y.shape[1])):  # Csak első 3 pozíció
                model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
                tscv = TimeSeriesSplit(n_splits=2)
                cv_scores = cross_val_score(model, X, y[:, pos], cv=tscv, scoring='neg_mean_squared_error')
                scores.append(-np.mean(cv_scores))
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        return study.best_params
        
    except Exception:
        # Alapértelmezett paraméterek
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }


def process_predictions(
    predictions: List[float],
    min_num: int,
    max_num: int,
    required_numbers: int,
    past_draws: List[List[int]]
) -> List[int]:
    """
    Predikciók feldolgozása
    """
    numbers = [int(np.clip(np.round(pred), min_num, max_num)) for pred in predictions]
    
    unique_numbers = []
    for num in numbers:
        if num not in unique_numbers:
            unique_numbers.append(num)
    
    # Hiányzó számok pótlása
    if len(unique_numbers) < required_numbers:
        freq_dict = {}
        for draw in past_draws[-20:]:
            for num in draw:
                if min_num <= num <= max_num:
                    freq_dict[num] = freq_dict.get(num, 0) + 1
        
        sorted_by_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        for num, _ in sorted_by_freq:
            if num not in unique_numbers:
                unique_numbers.append(num)
                if len(unique_numbers) >= required_numbers:
                    break
    
    # Random pótlás
    while len(unique_numbers) < required_numbers:
        num = random.randint(min_num, max_num)
        if num not in unique_numbers:
            unique_numbers.append(num)
    
    return unique_numbers[:required_numbers]


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """
    Történeti adatok lekérése
    """
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100])
    
    if is_main:
        draws = [draw.lottery_type_number for draw in past_draws
                 if isinstance(draw.lottery_type_number, list)]
    else:
        draws = [draw.additional_numbers for draw in past_draws
                 if hasattr(draw, 'additional_numbers') and 
                 isinstance(draw.additional_numbers, list)]
    
    return [[int(num) for num in draw] for draw in draws if draw]


def generate_frequency_based_fallback(
    past_draws: List[List[int]],
    min_num: int,
    max_num: int,
    required_numbers: int
) -> List[int]:
    """
    Gyakoriság alapú fallback
    """
    if not past_draws:
        return sorted(random.sample(range(min_num, max_num + 1), required_numbers))
    
    freq_dict = {}
    for draw in past_draws:
        for num in draw:
            if min_num <= num <= max_num:
                freq_dict[num] = freq_dict.get(num, 0) + 1
    
    sorted_nums = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    selected = [num for num, _ in sorted_nums[:required_numbers]]
    
    while len(selected) < required_numbers:
        num = random.randint(min_num, max_num)
        if num not in selected:
            selected.append(num)
    
    return sorted(selected[:required_numbers])


def generate_simple_fallback(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Egyszerű fallback
    """
    main_numbers = sorted(random.sample(
        range(int(lottery_type_instance.min_number), 
              int(lottery_type_instance.max_number) + 1),
        int(lottery_type_instance.pieces_of_draw_numbers)
    ))
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = sorted(random.sample(
            range(int(lottery_type_instance.additional_min_number),
                  int(lottery_type_instance.additional_max_number) + 1),
            int(lottery_type_instance.additional_numbers_count)
        ))
    
    return main_numbers, additional_numbers 