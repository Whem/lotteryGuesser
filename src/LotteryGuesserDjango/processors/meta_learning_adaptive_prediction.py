# meta_learning_adaptive_prediction.py
"""
Meta-tanulás Adaptív Predikció
Tanul a különböző algoritmusok teljesítményéből és adaptálódik
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict, deque
import random
import math
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class MetaLearningAdaptivePredictor:
    """
    Meta-tanulás alapú adaptív predikció
    """
    
    def __init__(self):
        self.algorithm_performance = defaultdict(lambda: deque(maxlen=100))
        self.meta_weights = defaultdict(float)
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.performance_window = 50
        
        # Algoritmus típusok
        self.algorithms = {
            'frequency': self._frequency_algorithm,
            'trend': self._trend_algorithm,
            'gap': self._gap_algorithm,
            'pattern': self._pattern_algorithm,
            'statistical': self._statistical_algorithm,
            'ensemble': self._ensemble_algorithm
        }
        
        # Kezdeti súlyok
        for alg in self.algorithms:
            self.meta_weights[alg] = 1.0 / len(self.algorithms)
    
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Főbejárási pont a meta-tanulás predikcióhoz
        """
        try:
            main_numbers = self._generate_meta_numbers(
                lottery_type_instance,
                min_num=int(lottery_type_instance.min_number),
                max_num=int(lottery_type_instance.max_number),
                required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
                is_main=True
            )
            
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._generate_meta_numbers(
                    lottery_type_instance,
                    min_num=int(lottery_type_instance.additional_min_number),
                    max_num=int(lottery_type_instance.additional_max_number),
                    required_numbers=int(lottery_type_instance.additional_numbers_count),
                    is_main=False
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
            
        except Exception as e:
            print(f"Hiba a meta_learning_adaptive_prediction-ben: {e}")
            return self._generate_fallback_numbers(lottery_type_instance)
    
    def _generate_meta_numbers(self, lottery_type_instance: lg_lottery_type,
                             min_num: int, max_num: int, required_numbers: int,
                             is_main: bool) -> List[int]:
        """
        Meta-tanulás alapú számgenerálás
        """
        historical_data = self._get_historical_data(lottery_type_instance, is_main)
        
        if len(historical_data) < 20:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Algoritmus teljesítmények frissítése
        self._update_algorithm_performance(historical_data, min_num, max_num)
        
        # Meta-súlyok frissítése
        self._update_meta_weights()
        
        # Algoritmus predikciók generálása
        predictions = {}
        for alg_name, alg_func in self.algorithms.items():
            try:
                predictions[alg_name] = alg_func(historical_data, min_num, max_num, required_numbers)
            except:
                predictions[alg_name] = self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Meta-ensemble
        final_numbers = self._meta_ensemble(predictions, min_num, max_num, required_numbers)
        
        return final_numbers
    
    def _update_algorithm_performance(self, historical_data: List[List[int]], 
                                    min_num: int, max_num: int) -> None:
        """
        Algoritmus teljesítmények frissítése
        """
        if len(historical_data) < 10:
            return
        
        # Backtesting az utolsó 10 húzásra
        for i in range(min(10, len(historical_data) - 1)):
            train_data = historical_data[i+1:]
            actual_draw = historical_data[i]
            
            # Minden algoritmus predikciója
            for alg_name, alg_func in self.algorithms.items():
                try:
                    prediction = alg_func(train_data, min_num, max_num, len(actual_draw))
                    score = self._calculate_prediction_score(prediction, actual_draw)
                    self.algorithm_performance[alg_name].append(score)
                except:
                    self.algorithm_performance[alg_name].append(0.0)
    
    def _calculate_prediction_score(self, prediction: List[int], actual: List[int]) -> float:
        """
        Predikció pontszám számítás
        """
        if not prediction or not actual:
            return 0.0
        
        # Találatok száma
        hits = len(set(prediction) & set(actual))
        
        # Normalizált pontszám
        score = hits / len(actual)
        
        # Bónusz pontok
        if hits > 0:
            score += 0.1 * hits  # Találat bónusz
        
        # Pozíció bónusz (ha a sorrend is számít)
        position_bonus = 0
        for i, num in enumerate(prediction):
            if i < len(actual) and num == actual[i]:
                position_bonus += 0.05
        
        return score + position_bonus
    
    def _update_meta_weights(self) -> None:
        """
        Meta-súlyok frissítése
        """
        # Átlagos teljesítmények számítása
        avg_performances = {}
        for alg_name, performances in self.algorithm_performance.items():
            if performances:
                avg_performances[alg_name] = np.mean(list(performances))
            else:
                avg_performances[alg_name] = 0.0
        
        # Softmax normalizálás
        if avg_performances:
            max_perf = max(avg_performances.values())
            exp_perfs = {alg: math.exp(perf - max_perf) for alg, perf in avg_performances.items()}
            total_exp = sum(exp_perfs.values())
            
            if total_exp > 0:
                for alg_name in self.algorithms:
                    new_weight = exp_perfs.get(alg_name, 0.1) / total_exp
                    # Exponenciális mozgóátlag
                    self.meta_weights[alg_name] = (
                        (1 - self.learning_rate) * self.meta_weights[alg_name] +
                        self.learning_rate * new_weight
                    )
    
    def _meta_ensemble(self, predictions: Dict[str, List[int]], 
                      min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """
        Meta-ensemble kombinálás
        """
        vote_counter = Counter()
        
        # Súlyozott szavazás
        for alg_name, numbers in predictions.items():
            weight = self.meta_weights.get(alg_name, 0.1)
            
            # Exploration vs exploitation
            if random.random() < self.exploration_rate:
                weight = 1.0 / len(self.algorithms)  # Egyenletes súlyozás
            
            for i, num in enumerate(numbers):
                position_weight = 1.0 / (i + 1)
                vote_counter[num] += weight * position_weight
        
        # Legmagasabb szavazatú számok
        top_numbers = [num for num, _ in vote_counter.most_common(required_numbers)]
        
        # Kiegészítés
        if len(top_numbers) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in top_numbers]
            random.shuffle(remaining)
            top_numbers.extend(remaining[:required_numbers - len(top_numbers)])
        
        return top_numbers[:required_numbers]
    
    def _frequency_algorithm(self, historical_data: List[List[int]], 
                           min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """Frekvencia alapú algoritmus"""
        counter = Counter(num for draw in historical_data[:30] for num in draw)
        return [num for num, _ in counter.most_common(required_numbers)]
    
    def _trend_algorithm(self, historical_data: List[List[int]], 
                        min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """Trend alapú algoritmus"""
        if len(historical_data) < 20:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        recent = Counter(num for draw in historical_data[:10] for num in draw)
        older = Counter(num for draw in historical_data[10:20] for num in draw)
        
        trends = {num: recent.get(num, 0) - older.get(num, 0) * 0.5 
                 for num in range(min_num, max_num + 1)}
        
        return sorted(trends.keys(), key=lambda x: trends[x], reverse=True)[:required_numbers]
    
    def _gap_algorithm(self, historical_data: List[List[int]], 
                      min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """Gap alapú algoritmus"""
        last_seen = {}
        for i, draw in enumerate(historical_data):
            for num in draw:
                last_seen[num] = i
        
        gaps = {num: last_seen.get(num, len(historical_data)) 
               for num in range(min_num, max_num + 1)}
        
        return sorted(gaps.keys(), key=lambda x: gaps[x], reverse=True)[:required_numbers]
    
    def _pattern_algorithm(self, historical_data: List[List[int]], 
                          min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """Minta alapú algoritmus"""
        if len(historical_data) < 5:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        patterns = defaultdict(list)
        for i in range(len(historical_data) - 1):
            current_sum = sum(historical_data[i])
            next_draw = historical_data[i + 1]
            patterns[current_sum % 10].extend(next_draw)
        
        recent_sum = sum(historical_data[0])
        pattern_key = recent_sum % 10
        
        if pattern_key in patterns:
            candidates = patterns[pattern_key]
            counter = Counter(candidates)
            return [num for num, _ in counter.most_common(required_numbers)]
        
        return self._generate_smart_random(min_num, max_num, required_numbers)
    
    def _statistical_algorithm(self, historical_data: List[List[int]], 
                             min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """Statisztikai algoritmus"""
        all_nums = [num for draw in historical_data for num in draw]
        if not all_nums:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        mean_val = np.mean(all_nums)
        std_val = np.std(all_nums)
        
        scores = {}
        for num in range(min_num, max_num + 1):
            z_score = abs(num - mean_val) / max(std_val, 1)
            scores[num] = 1 / (1 + z_score)
        
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:required_numbers]
    
    def _ensemble_algorithm(self, historical_data: List[List[int]], 
                          min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """Ensemble algoritmus"""
        # Mini ensemble a többi algoritmusból
        freq_nums = self._frequency_algorithm(historical_data, min_num, max_num, required_numbers)
        trend_nums = self._trend_algorithm(historical_data, min_num, max_num, required_numbers)
        gap_nums = self._gap_algorithm(historical_data, min_num, max_num, required_numbers)
        
        vote_counter = Counter()
        for nums in [freq_nums, trend_nums, gap_nums]:
            for i, num in enumerate(nums):
                vote_counter[num] += 1.0 / (i + 1)
        
        return [num for num, _ in vote_counter.most_common(required_numbers)]
    
    def _generate_smart_random(self, min_num: int, max_num: int, count: int) -> List[int]:
        """Intelligens véletlen generálás"""
        center = (min_num + max_num) / 2
        std = (max_num - min_num) / 6
        
        numbers = set()
        attempts = 0
        
        while len(numbers) < count and attempts < 1000:
            num = int(np.random.normal(center, std))
            if min_num <= num <= max_num:
                numbers.add(num)
            attempts += 1
        
        if len(numbers) < count:
            remaining = [num for num in range(min_num, max_num + 1) if num not in numbers]
            random.shuffle(remaining)
            numbers.update(remaining[:count - len(numbers)])
        
        return list(numbers)[:count]
    
    def _get_historical_data(self, lottery_type_instance: lg_lottery_type, 
                           is_main: bool) -> List[List[int]]:
        """Történeti adatok lekérése"""
        field_name = 'lottery_type_number' if is_main else 'additional_numbers'
        
        try:
            queryset = lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance
            ).order_by('-id').values_list(field_name, flat=True)[:100]
            
            historical_data = []
            for draw in queryset:
                if isinstance(draw, list) and len(draw) > 0:
                    valid_numbers = [int(num) for num in draw if isinstance(num, (int, float))]
                    if valid_numbers:
                        historical_data.append(valid_numbers)
            
            return historical_data
        except Exception:
            return []
    
    def _generate_fallback_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """Fallback számgenerálás"""
        main_numbers = self._generate_smart_random(
            int(lottery_type_instance.min_number),
            int(lottery_type_instance.max_number),
            int(lottery_type_instance.pieces_of_draw_numbers)
        )
        
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = self._generate_smart_random(
                int(lottery_type_instance.additional_min_number),
                int(lottery_type_instance.additional_max_number),
                int(lottery_type_instance.additional_numbers_count)
            )
        
        return sorted(main_numbers), sorted(additional_numbers)


# Globális instance
meta_learning_predictor = MetaLearningAdaptivePredictor()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Főbejárási pont a meta-tanulás predikcióhoz"""
    return meta_learning_predictor.get_numbers(lottery_type_instance)
