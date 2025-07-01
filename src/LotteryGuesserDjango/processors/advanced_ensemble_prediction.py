# advanced_ensemble_prediction.py
"""
Fejlett Ensemble Predikció Eurojackpot-hoz
Kombinálja a legjobb algoritmusokat súlyozott módon a teljesítmény alapján
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import random
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
from lottery_handler.models import lg_algorithm_score, lg_algorithm_performance

# Import the best performing algorithms
try:
    from .advanced_transformer_prediction import get_numbers as transformer_get_numbers
except ImportError:
    transformer_get_numbers = None

try:
    from .lstm_neural_network_prediction import get_numbers as lstm_get_numbers
except ImportError:
    lstm_get_numbers = None

try:
    from .xgboost_prediction import get_numbers as xgboost_get_numbers
except ImportError:
    xgboost_get_numbers = None

try:
    from .lightgbm_prediction import get_numbers as lightgbm_get_numbers
except ImportError:
    lightgbm_get_numbers = None

try:
    from .neural_network_prediction import get_numbers as neural_net_get_numbers
except ImportError:
    neural_net_get_numbers = None

try:
    from .quantum_inspired_optimization_prediction import get_numbers as quantum_get_numbers
except ImportError:
    quantum_get_numbers = None

# Fallback algorithms
try:
    from .markov_chain_prediction import get_numbers as markov_get_numbers
except ImportError:
    markov_get_numbers = None

try:
    from .genetic_algorithm_prediction import get_numbers as genetic_get_numbers
except ImportError:
    genetic_get_numbers = None

try:
    from .optimized_hybrid_predictor import get_numbers as hybrid_get_numbers
except ImportError:
    hybrid_get_numbers = None


class AdvancedEnsemblePredictor:
    """
    Fejlett ensemble predikció osztály
    """
    
    def __init__(self):
        self.algorithm_functions = {
            'optimized_hybrid_predictor': hybrid_get_numbers,
            'advanced_transformer_prediction': transformer_get_numbers,
            'lstm_neural_network_prediction': lstm_get_numbers,
            'xgboost_prediction': xgboost_get_numbers,
            'lightgbm_prediction': lightgbm_get_numbers,
            'neural_network_prediction': neural_net_get_numbers,
            'quantum_inspired_optimization_prediction': quantum_get_numbers,
            'markov_chain_prediction': markov_get_numbers,
            'genetic_algorithm_prediction': genetic_get_numbers,
        }
        
        # Remove None functions
        self.algorithm_functions = {k: v for k, v in self.algorithm_functions.items() if v is not None}
        
        self.default_weights = {
            'optimized_hybrid_predictor': 0.30,
            'advanced_transformer_prediction': 0.20,
            'lstm_neural_network_prediction': 0.18,
            'xgboost_prediction': 0.12,
            'lightgbm_prediction': 0.10,
            'neural_network_prediction': 0.05,
            'quantum_inspired_optimization_prediction': 0.03,
            'markov_chain_prediction': 0.01,
            'genetic_algorithm_prediction': 0.01,
        }
    
    def get_algorithm_weights(self) -> Dict[str, float]:
        """
        Lekéri az algoritmusok súlyait a teljesítmény alapján
        """
        try:
            # Lekérjük az algoritmus pontszámokat
            scores = lg_algorithm_score.objects.all()
            performance_data = lg_algorithm_performance.objects.all()
            
            weights = {}
            total_score = 0
            
            # Számoljuk ki a súlyokat a pontszámok és teljesítmény alapján
            for score in scores:
                if score.algorithm_name in self.algorithm_functions:
                    # Kombináljuk a pontszámot és a teljesítményt
                    base_score = max(0.1, score.current_score)  # Minimum 0.1
                    
                    # Teljesítmény bónusz (gyorsabb algoritmusok kisebb bónuszt kapnak)
                    perf = performance_data.filter(algorithm_name=score.algorithm_name).first()
                    if perf and perf.average_execution_time > 0:
                        speed_factor = min(2.0, 1000 / perf.average_execution_time)  # Gyorsabb = jobb
                        final_score = base_score * (1 + speed_factor * 0.1)
                    else:
                        final_score = base_score
                    
                    weights[score.algorithm_name] = final_score
                    total_score += final_score
            
            # Normalizáljuk a súlyokat
            if total_score > 0:
                weights = {k: v / total_score for k, v in weights.items()}
            else:
                # Fallback az alapértelmezett súlyokra
                weights = {k: v for k, v in self.default_weights.items() 
                          if k in self.algorithm_functions}
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
            
            return weights
            
        except Exception:
            # Fallback az alapértelmezett súlyokra hiba esetén
            weights = {k: v for k, v in self.default_weights.items() 
                      if k in self.algorithm_functions}
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            return weights
    
    def generate_predictions(self, lottery_type_instance: lg_lottery_type) -> Dict[str, Tuple[List[int], List[int]]]:
        """
        Generál predikciót az összes algoritmussal
        """
        predictions = {}
        
        for algo_name, algo_func in self.algorithm_functions.items():
            try:
                main_nums, additional_nums = algo_func(lottery_type_instance)
                predictions[algo_name] = (main_nums, additional_nums)
            except Exception as e:
                print(f"Hiba {algo_name} algoritmusban: {e}")
                continue
        
        return predictions
    
    def weighted_ensemble_vote(self, 
                              predictions: Dict[str, Tuple[List[int], List[int]]],
                              weights: Dict[str, float],
                              required_main: int,
                              required_additional: int,
                              min_main: int, max_main: int,
                              min_additional: int, max_additional: int) -> Tuple[List[int], List[int]]:
        """
        Súlyozott szavazás az ensemble predikciókra
        """
        # Fő számok súlyozott szavazása
        main_votes = defaultdict(float)
        for algo_name, (main_nums, _) in predictions.items():
            weight = weights.get(algo_name, 0.0)
            for num in main_nums:
                main_votes[num] += weight
        
        # Rendezzük a fő számokat súly szerint
        sorted_main = sorted(main_votes.items(), key=lambda x: x[1], reverse=True)
        final_main = [num for num, _ in sorted_main if min_main <= num <= max_main][:required_main]
        
        # Ha nincs elég szám, töltsük fel a leggyakoribbakkal
        if len(final_main) < required_main:
            all_main_nums = [num for _, (main_nums, _) in predictions.items() for num in main_nums]
            freq_counter = Counter(all_main_nums)
            for num, _ in freq_counter.most_common():
                if num not in final_main and min_main <= num <= max_main:
                    final_main.append(num)
                    if len(final_main) >= required_main:
                        break
        
        # Végső fallback random számokkal
        while len(final_main) < required_main:
            num = random.randint(min_main, max_main)
            if num not in final_main:
                final_main.append(num)
        
        final_main = sorted(final_main[:required_main])
        
        # Kiegészítő számok (ha vannak)
        final_additional = []
        if required_additional > 0:
            additional_votes = defaultdict(float)
            for algo_name, (_, additional_nums) in predictions.items():
                weight = weights.get(algo_name, 0.0)
                for num in additional_nums:
                    additional_votes[num] += weight
            
            sorted_additional = sorted(additional_votes.items(), key=lambda x: x[1], reverse=True)
            final_additional = [num for num, _ in sorted_additional 
                              if min_additional <= num <= max_additional][:required_additional]
            
            # Feltöltés ha szükséges
            if len(final_additional) < required_additional:
                all_additional_nums = [num for _, (_, additional_nums) in predictions.items() 
                                     for num in additional_nums]
                freq_counter = Counter(all_additional_nums)
                for num, _ in freq_counter.most_common():
                    if num not in final_additional and min_additional <= num <= max_additional:
                        final_additional.append(num)
                        if len(final_additional) >= required_additional:
                            break
            
            # Végső fallback
            while len(final_additional) < required_additional:
                num = random.randint(min_additional, max_additional)
                if num not in final_additional:
                    final_additional.append(num)
            
            final_additional = sorted(final_additional[:required_additional])
        
        return final_main, final_additional


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Főbejárási pont a fejlett ensemble predikcióhoz
    """
    try:
        predictor = AdvancedEnsemblePredictor()
        
        # Súlyok lekérése
        weights = predictor.get_algorithm_weights()
        
        # Predikciók generálása
        predictions = predictor.generate_predictions(lottery_type_instance)
        
        if not predictions:
            # Ha minden algoritmus hibázik, fallback random számokra
            return generate_fallback_numbers(lottery_type_instance)
        
        # Ensemble szavazás
        main_numbers, additional_numbers = predictor.weighted_ensemble_vote(
            predictions=predictions,
            weights=weights,
            required_main=int(lottery_type_instance.pieces_of_draw_numbers),
            required_additional=int(lottery_type_instance.additional_numbers_count) if lottery_type_instance.has_additional_numbers else 0,
            min_main=int(lottery_type_instance.min_number),
            max_main=int(lottery_type_instance.max_number),
            min_additional=int(lottery_type_instance.additional_min_number) if lottery_type_instance.has_additional_numbers else 0,
            max_additional=int(lottery_type_instance.additional_max_number) if lottery_type_instance.has_additional_numbers else 0
        )
        
        return main_numbers, additional_numbers
        
    except Exception as e:
        print(f"Hiba az ensemble predikcióban: {e}")
        return generate_fallback_numbers(lottery_type_instance)


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Fallback számgenerálás amikor minden más hibázik
    """
    # Fő számok
    main_numbers = sorted(random.sample(
        range(int(lottery_type_instance.min_number), 
              int(lottery_type_instance.max_number) + 1),
        int(lottery_type_instance.pieces_of_draw_numbers)
    ))
    
    # Kiegészítő számok
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = sorted(random.sample(
            range(int(lottery_type_instance.additional_min_number),
                  int(lottery_type_instance.additional_max_number) + 1),
            int(lottery_type_instance.additional_numbers_count)
        ))
    
    return main_numbers, additional_numbers


def analyze_ensemble_performance(lottery_type_instance: lg_lottery_type) -> Dict[str, float]:
    """
    Elemzi az ensemble teljesítményét
    """
    predictor = AdvancedEnsemblePredictor()
    weights = predictor.get_algorithm_weights()
    
    # Visszaadja a súlyokat elemzéshez
    return weights 