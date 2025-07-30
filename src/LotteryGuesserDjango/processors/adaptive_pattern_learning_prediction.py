# adaptive_pattern_learning_prediction.py
"""
Adaptív Minta Tanuló Predikció
Dinamikusan tanul a múltbeli mintákból és adaptálódik az új trendekhez
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter, defaultdict, deque
import random
import math
from datetime import datetime
from scipy.stats import chi2_contingency, pearsonr
from sklearn.preprocessing import MinMaxScaler
import json

from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class AdaptivePatternLearner:
    """
    Adaptív minta tanuló osztály
    Folyamatosan tanul a múltbeli mintákból és adaptálódik
    """
    
    def __init__(self):
        self.pattern_memory = {}  # Minta memória
        self.performance_history = deque(maxlen=100)  # Teljesítmény történet
        self.learning_rate = 0.1
        self.pattern_threshold = 0.3
        self.max_pattern_length = 10
        
        # Különböző minta típusok súlyai
        self.pattern_weights = {
            'sequence_patterns': 0.25,
            'sum_patterns': 0.20,
            'gap_patterns': 0.20,
            'position_patterns': 0.15,
            'parity_patterns': 0.10,
            'divisibility_patterns': 0.10
        }
    
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Főbejárási pont az adaptív minta tanuló predikcióhoz
        """
        try:
            # Fő számok generálása
            main_numbers = self._generate_adaptive_numbers(
                lottery_type_instance,
                min_num=int(lottery_type_instance.min_number),
                max_num=int(lottery_type_instance.max_number),
                required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
                is_main=True
            )
            
            # Kiegészítő számok generálása
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._generate_adaptive_numbers(
                    lottery_type_instance,
                    min_num=int(lottery_type_instance.additional_min_number),
                    max_num=int(lottery_type_instance.additional_max_number),
                    required_numbers=int(lottery_type_instance.additional_numbers_count),
                    is_main=False
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
            
        except Exception as e:
            print(f"Hiba az adaptive_pattern_learning_prediction-ben: {e}")
            return self._generate_fallback_numbers(lottery_type_instance)
    
    def _generate_adaptive_numbers(self, lottery_type_instance: lg_lottery_type,
                                 min_num: int, max_num: int, required_numbers: int,
                                 is_main: bool) -> List[int]:
        """
        Adaptív számgenerálás minta alapon
        """
        # Történeti adatok lekérése
        historical_data = self._get_historical_data(lottery_type_instance, is_main)
        
        if len(historical_data) < 20:
            return self._generate_intelligent_random(min_num, max_num, required_numbers)
        
        # Minták felismerése és tanulás
        patterns = self._learn_patterns(historical_data, min_num, max_num)
        
        # Predikciók generálása különböző minta típusokkal
        predictions = {}
        
        # 1. Szekvencia minták
        predictions['sequence_patterns'] = self._predict_from_sequences(
            patterns['sequences'], historical_data, min_num, max_num, required_numbers
        )
        
        # 2. Összeg minták
        predictions['sum_patterns'] = self._predict_from_sums(
            patterns['sums'], historical_data, min_num, max_num, required_numbers
        )
        
        # 3. Gap minták
        predictions['gap_patterns'] = self._predict_from_gaps(
            patterns['gaps'], historical_data, min_num, max_num, required_numbers
        )
        
        # 4. Pozíció minták
        predictions['position_patterns'] = self._predict_from_positions(
            patterns['positions'], historical_data, min_num, max_num, required_numbers
        )
        
        # 5. Páros/páratlan minták
        predictions['parity_patterns'] = self._predict_from_parity(
            patterns['parity'], historical_data, min_num, max_num, required_numbers
        )
        
        # 6. Oszthatósági minták
        predictions['divisibility_patterns'] = self._predict_from_divisibility(
            patterns['divisibility'], historical_data, min_num, max_num, required_numbers
        )
        
        # Adaptív ensemble
        final_numbers = self._adaptive_ensemble(
            predictions, min_num, max_num, required_numbers
        )
        
        return final_numbers
    
    def _get_historical_data(self, lottery_type_instance: lg_lottery_type,
                           is_main: bool) -> List[List[int]]:
        """
        Történeti adatok lekérése
        """
        field_name = 'lottery_type_number' if is_main else 'additional_numbers'
        
        queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id').values_list(field_name, flat=True)
        
        historical_data = []
        for draw in queryset:
            if isinstance(draw, list) and len(draw) > 0:
                valid_numbers = [int(num) for num in draw if isinstance(num, (int, float))]
                if valid_numbers:
                    historical_data.append(sorted(valid_numbers))
        
        return historical_data[:150]  # Legutóbbi 150 húzás
    
    def _learn_patterns(self, historical_data: List[List[int]], 
                       min_num: int, max_num: int) -> Dict:
        """
        Minták tanulása a történeti adatokból
        """
        patterns = {
            'sequences': self._learn_sequence_patterns(historical_data),
            'sums': self._learn_sum_patterns(historical_data),
            'gaps': self._learn_gap_patterns(historical_data),
            'positions': self._learn_position_patterns(historical_data),
            'parity': self._learn_parity_patterns(historical_data),
            'divisibility': self._learn_divisibility_patterns(historical_data, min_num, max_num)
        }
        
        return patterns
    
    def _learn_sequence_patterns(self, historical_data: List[List[int]]) -> Dict:
        """
        Szekvencia minták tanulása
        """
        sequence_patterns = defaultdict(list)
        
        for i in range(len(historical_data) - 1):
            current_draw = historical_data[i]
            next_draw = historical_data[i + 1]
            
            # Különböző hosszúságú szekvenciák
            for seq_len in range(2, min(len(current_draw) + 1, self.max_pattern_length)):
                for start_idx in range(len(current_draw) - seq_len + 1):
                    sequence = tuple(current_draw[start_idx:start_idx + seq_len])
                    sequence_patterns[sequence].append(next_draw)
        
        return dict(sequence_patterns)
    
    def _learn_sum_patterns(self, historical_data: List[List[int]]) -> Dict:
        """
        Összeg minták tanulása
        """
        sum_patterns = defaultdict(list)
        
        for i in range(len(historical_data) - 1):
            current_sum = sum(historical_data[i])
            next_draw = historical_data[i + 1]
            
            # Összeg kategóriák
            sum_category = self._categorize_sum(current_sum)
            sum_patterns[sum_category].append(next_draw)
        
        return dict(sum_patterns)
    
    def _learn_gap_patterns(self, historical_data: List[List[int]]) -> Dict:
        """
        Gap minták tanulása (számok közötti távolságok)
        """
        gap_patterns = defaultdict(list)
        
        for i in range(len(historical_data) - 1):
            current_draw = sorted(historical_data[i])
            next_draw = historical_data[i + 1]
            
            # Számok közötti gap-ek
            gaps = []
            for j in range(len(current_draw) - 1):
                gap = current_draw[j + 1] - current_draw[j]
                gaps.append(gap)
            
            gap_signature = tuple(gaps)
            gap_patterns[gap_signature].append(next_draw)
        
        return dict(gap_patterns)
    
    def _learn_position_patterns(self, historical_data: List[List[int]]) -> Dict:
        """
        Pozíció minták tanulása
        """
        position_patterns = defaultdict(lambda: defaultdict(list))
        
        for i in range(len(historical_data) - 1):
            current_draw = historical_data[i]
            next_draw = historical_data[i + 1]
            
            for pos, number in enumerate(current_draw):
                position_patterns[pos][number].append(next_draw)
        
        return dict(position_patterns)
    
    def _learn_parity_patterns(self, historical_data: List[List[int]]) -> Dict:
        """
        Páros/páratlan minták tanulása
        """
        parity_patterns = defaultdict(list)
        
        for i in range(len(historical_data) - 1):
            current_draw = historical_data[i]
            next_draw = historical_data[i + 1]
            
            even_count = sum(1 for num in current_draw if num % 2 == 0)
            odd_count = len(current_draw) - even_count
            
            parity_signature = (even_count, odd_count)
            parity_patterns[parity_signature].append(next_draw)
        
        return dict(parity_patterns)
    
    def _learn_divisibility_patterns(self, historical_data: List[List[int]], 
                                   min_num: int, max_num: int) -> Dict:
        """
        Oszthatósági minták tanulása
        """
        divisibility_patterns = defaultdict(list)
        divisors = [2, 3, 5, 7]  # Alapvető osztók
        
        for i in range(len(historical_data) - 1):
            current_draw = historical_data[i]
            next_draw = historical_data[i + 1]
            
            div_signature = []
            for divisor in divisors:
                count = sum(1 for num in current_draw if num % divisor == 0)
                div_signature.append(count)
            
            div_signature = tuple(div_signature)
            divisibility_patterns[div_signature].append(next_draw)
        
        return dict(divisibility_patterns)
    
    def _predict_from_sequences(self, sequence_patterns: Dict, 
                              historical_data: List[List[int]],
                              min_num: int, max_num: int, 
                              required_numbers: int) -> List[int]:
        """
        Szekvencia minták alapján predikció
        """
        if not sequence_patterns or not historical_data:
            return self._generate_intelligent_random(min_num, max_num, required_numbers)
        
        recent_draw = historical_data[0]
        candidate_numbers = Counter()
        
        # Keresés a legutóbbi húzás szekvenciáira
        for seq_len in range(2, min(len(recent_draw) + 1, self.max_pattern_length)):
            for start_idx in range(len(recent_draw) - seq_len + 1):
                sequence = tuple(recent_draw[start_idx:start_idx + seq_len])
                
                if sequence in sequence_patterns:
                    for next_draw in sequence_patterns[sequence]:
                        for num in next_draw:
                            if min_num <= num <= max_num:
                                candidate_numbers[num] += 1
        
        # Leggyakoribb számok kiválasztása
        if candidate_numbers:
            most_common = [num for num, _ in candidate_numbers.most_common(required_numbers)]
            if len(most_common) < required_numbers:
                remaining = [num for num in range(min_num, max_num + 1) if num not in most_common]
                random.shuffle(remaining)
                most_common.extend(remaining[:required_numbers - len(most_common)])
            return most_common[:required_numbers]
        
        return self._generate_intelligent_random(min_num, max_num, required_numbers)
    
    def _predict_from_sums(self, sum_patterns: Dict, 
                         historical_data: List[List[int]],
                         min_num: int, max_num: int, 
                         required_numbers: int) -> List[int]:
        """
        Összeg minták alapján predikció
        """
        if not sum_patterns or not historical_data:
            return self._generate_intelligent_random(min_num, max_num, required_numbers)
        
        recent_sum = sum(historical_data[0])
        sum_category = self._categorize_sum(recent_sum)
        
        candidate_numbers = Counter()
        
        if sum_category in sum_patterns:
            for next_draw in sum_patterns[sum_category]:
                for num in next_draw:
                    if min_num <= num <= max_num:
                        candidate_numbers[num] += 1
        
        # Leggyakoribb számok kiválasztása
        if candidate_numbers:
            most_common = [num for num, _ in candidate_numbers.most_common(required_numbers)]
            if len(most_common) < required_numbers:
                remaining = [num for num in range(min_num, max_num + 1) if num not in most_common]
                random.shuffle(remaining)
                most_common.extend(remaining[:required_numbers - len(most_common)])
            return most_common[:required_numbers]
        
        return self._generate_intelligent_random(min_num, max_num, required_numbers)
    
    def _predict_from_gaps(self, gap_patterns: Dict, 
                         historical_data: List[List[int]],
                         min_num: int, max_num: int, 
                         required_numbers: int) -> List[int]:
        """
        Gap minták alapján predikció
        """
        if not gap_patterns or not historical_data:
            return self._generate_intelligent_random(min_num, max_num, required_numbers)
        
        recent_draw = sorted(historical_data[0])
        gaps = []
        for i in range(len(recent_draw) - 1):
            gap = recent_draw[i + 1] - recent_draw[i]
            gaps.append(gap)
        
        gap_signature = tuple(gaps)
        candidate_numbers = Counter()
        
        if gap_signature in gap_patterns:
            for next_draw in gap_patterns[gap_signature]:
                for num in next_draw:
                    if min_num <= num <= max_num:
                        candidate_numbers[num] += 1
        
        # Leggyakoribb számok kiválasztása
        if candidate_numbers:
            most_common = [num for num, _ in candidate_numbers.most_common(required_numbers)]
            if len(most_common) < required_numbers:
                remaining = [num for num in range(min_num, max_num + 1) if num not in most_common]
                random.shuffle(remaining)
                most_common.extend(remaining[:required_numbers - len(most_common)])
            return most_common[:required_numbers]
        
        return self._generate_intelligent_random(min_num, max_num, required_numbers)
    
    def _predict_from_positions(self, position_patterns: Dict, 
                              historical_data: List[List[int]],
                              min_num: int, max_num: int, 
                              required_numbers: int) -> List[int]:
        """
        Pozíció minták alapján predikció
        """
        if not position_patterns or not historical_data:
            return self._generate_intelligent_random(min_num, max_num, required_numbers)
        
        recent_draw = historical_data[0]
        candidate_numbers = Counter()
        
        for pos, number in enumerate(recent_draw):
            if pos in position_patterns and number in position_patterns[pos]:
                for next_draw in position_patterns[pos][number]:
                    for num in next_draw:
                        if min_num <= num <= max_num:
                            candidate_numbers[num] += 1
        
        # Leggyakoribb számok kiválasztása
        if candidate_numbers:
            most_common = [num for num, _ in candidate_numbers.most_common(required_numbers)]
            if len(most_common) < required_numbers:
                remaining = [num for num in range(min_num, max_num + 1) if num not in most_common]
                random.shuffle(remaining)
                most_common.extend(remaining[:required_numbers - len(most_common)])
            return most_common[:required_numbers]
        
        return self._generate_intelligent_random(min_num, max_num, required_numbers)
    
    def _predict_from_parity(self, parity_patterns: Dict, 
                           historical_data: List[List[int]],
                           min_num: int, max_num: int, 
                           required_numbers: int) -> List[int]:
        """
        Páros/páratlan minták alapján predikció
        """
        if not parity_patterns or not historical_data:
            return self._generate_intelligent_random(min_num, max_num, required_numbers)
        
        recent_draw = historical_data[0]
        even_count = sum(1 for num in recent_draw if num % 2 == 0)
        odd_count = len(recent_draw) - even_count
        
        parity_signature = (even_count, odd_count)
        candidate_numbers = Counter()
        
        if parity_signature in parity_patterns:
            for next_draw in parity_patterns[parity_signature]:
                for num in next_draw:
                    if min_num <= num <= max_num:
                        candidate_numbers[num] += 1
        
        # Leggyakoribb számok kiválasztása
        if candidate_numbers:
            most_common = [num for num, _ in candidate_numbers.most_common(required_numbers)]
            if len(most_common) < required_numbers:
                remaining = [num for num in range(min_num, max_num + 1) if num not in most_common]
                random.shuffle(remaining)
                most_common.extend(remaining[:required_numbers - len(most_common)])
            return most_common[:required_numbers]
        
        return self._generate_intelligent_random(min_num, max_num, required_numbers)
    
    def _predict_from_divisibility(self, divisibility_patterns: Dict, 
                                 historical_data: List[List[int]],
                                 min_num: int, max_num: int, 
                                 required_numbers: int) -> List[int]:
        """
        Oszthatósági minták alapján predikció
        """
        if not divisibility_patterns or not historical_data:
            return self._generate_intelligent_random(min_num, max_num, required_numbers)
        
        recent_draw = historical_data[0]
        divisors = [2, 3, 5, 7]
        
        div_signature = []
        for divisor in divisors:
            count = sum(1 for num in recent_draw if num % divisor == 0)
            div_signature.append(count)
        
        div_signature = tuple(div_signature)
        candidate_numbers = Counter()
        
        if div_signature in divisibility_patterns:
            for next_draw in divisibility_patterns[div_signature]:
                for num in next_draw:
                    if min_num <= num <= max_num:
                        candidate_numbers[num] += 1
        
        # Leggyakoribb számok kiválasztása
        if candidate_numbers:
            most_common = [num for num, _ in candidate_numbers.most_common(required_numbers)]
            if len(most_common) < required_numbers:
                remaining = [num for num in range(min_num, max_num + 1) if num not in most_common]
                random.shuffle(remaining)
                most_common.extend(remaining[:required_numbers - len(most_common)])
            return most_common[:required_numbers]
        
        return self._generate_intelligent_random(min_num, max_num, required_numbers)
    
    def _adaptive_ensemble(self, predictions: Dict[str, List[int]], 
                         min_num: int, max_num: int, 
                         required_numbers: int) -> List[int]:
        """
        Adaptív ensemble szavazás
        """
        # Dinamikus súlyok számítása teljesítmény alapján
        adaptive_weights = self._calculate_adaptive_weights()
        
        vote_scores = defaultdict(float)
        
        for method, numbers in predictions.items():
            weight = adaptive_weights.get(method, self.pattern_weights.get(method, 0.1))
            for i, num in enumerate(numbers):
                # Pozíció alapú súlyozás
                position_weight = 1.0 / (i + 1)
                vote_scores[num] += weight * position_weight
        
        # Legmagasabb score-ú számok kiválasztása
        sorted_votes = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [num for num, _ in sorted_votes[:required_numbers]]
        
        # Kiegészítés szükség esetén
        if len(selected) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[:required_numbers - len(selected)])
        
        return selected[:required_numbers]
    
    def _calculate_adaptive_weights(self) -> Dict[str, float]:
        """
        Adaptív súlyok számítása teljesítmény alapján
        """
        if not self.performance_history:
            return self.pattern_weights.copy()
        
        # Egyszerű adaptív súlyozás
        adaptive_weights = self.pattern_weights.copy()
        
        # Itt lehetne implementálni a teljesítmény alapú súlyozást
        # Egyelőre az alapértelmezett súlyokat használjuk
        
        return adaptive_weights
    
    def _categorize_sum(self, sum_value: int) -> str:
        """
        Összeg kategorizálása
        """
        if sum_value < 100:
            return "low"
        elif sum_value < 200:
            return "medium"
        elif sum_value < 300:
            return "high"
        else:
            return "very_high"
    
    def _generate_intelligent_random(self, min_num: int, max_num: int, 
                                   required_numbers: int) -> List[int]:
        """
        Intelligens véletlen számgenerálás
        """
        # Normál eloszlás alapú generálás
        center = (min_num + max_num) / 2
        std = (max_num - min_num) / 6
        
        numbers = set()
        attempts = 0
        
        while len(numbers) < required_numbers and attempts < 1000:
            num = int(np.random.normal(center, std))
            if min_num <= num <= max_num:
                numbers.add(num)
            attempts += 1
        
        # Kiegészítés egyenletes eloszlással
        if len(numbers) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in numbers]
            random.shuffle(remaining)
            numbers.update(remaining[:required_numbers - len(numbers)])
        
        return list(numbers)[:required_numbers]
    
    def _generate_fallback_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Fallback számgenerálás hiba esetén
        """
        main_numbers = self._generate_intelligent_random(
            int(lottery_type_instance.min_number),
            int(lottery_type_instance.max_number),
            int(lottery_type_instance.pieces_of_draw_numbers)
        )
        
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = self._generate_intelligent_random(
                int(lottery_type_instance.additional_min_number),
                int(lottery_type_instance.additional_max_number),
                int(lottery_type_instance.additional_numbers_count)
            )
        
        return sorted(main_numbers), sorted(additional_numbers)


# Globális instance
pattern_learner = AdaptivePatternLearner()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Főbejárási pont az adaptív minta tanuló predikcióhoz
    """
    return pattern_learner.get_numbers(lottery_type_instance)
