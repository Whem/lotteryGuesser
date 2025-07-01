# optimized_hybrid_predictor.py
"""
Optimalizált Hibrid Predikció
Kombinálja a legjobb technikákat: quantum resonance, pattern evolution, frequency analysis
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random
import math


class FastQuantumLayer:
    """Gyorsított kvantum réteg"""
    
    def __init__(self, size=8):
        self.size = size
        self.weights = np.random.rand(size)
        self.phases = np.random.rand(size) * 2 * np.pi
    
    def process(self, inputs: np.ndarray) -> np.ndarray:
        """Fast quantum processing"""
        # Ensure input compatibility
        if len(inputs) != len(self.weights):
            # Resize weights to match input dimension
            if len(inputs) > len(self.weights):
                self.weights = np.resize(self.weights, len(inputs))
                self.phases = np.resize(self.phases, len(inputs))
            else:
                inputs = np.resize(inputs, len(self.weights))
        
        amplitudes = np.dot(self.weights, inputs)
        return amplitudes ** 2  # Simplified activation


class PatternEvolutionEngine:
    """Gyors mintaevolúció motor"""
    
    def __init__(self, min_num: int, max_num: int):
        self.min_num = min_num
        self.max_num = max_num
        self.pattern_scores = defaultdict(float)
    
    def analyze_patterns(self, draws: List[List[int]]) -> Dict[int, float]:
        """Gyors minta elemzés"""
        scores = defaultdict(float)
        
        # Arithmetic progression analysis
        for draw in draws[:20]:  # Csak a legutóbbi 20
            sorted_nums = sorted(draw)
            for i in range(len(sorted_nums) - 2):
                diff1 = sorted_nums[i+1] - sorted_nums[i]
                diff2 = sorted_nums[i+2] - sorted_nums[i+1]
                if diff1 == diff2:  # Arithmetic progression found
                    next_num = sorted_nums[i+2] + diff1
                    if self.min_num <= next_num <= self.max_num:
                        scores[next_num] += 2.0
        
        # Golden ratio patterns
        for draw in draws[:15]:
            for num in draw:
                golden_candidates = [
                    int(num * 1.618),
                    int(num / 1.618),
                    int(num * 0.618)
                ]
                for candidate in golden_candidates:
                    if self.min_num <= candidate <= self.max_num:
                        scores[candidate] += 1.0
        
        # Fibonacci-like sequences
        for draw in draws[:10]:
            sorted_nums = sorted(draw)
            for i in range(len(sorted_nums) - 2):
                fib_next = sorted_nums[i] + sorted_nums[i+1]
                if self.min_num <= fib_next <= self.max_num:
                    scores[fib_next] += 1.5
        
        return scores


class FrequencyHarmonicsAnalyzer:
    """Gyors frekvencia harmonika elemző"""
    
    def __init__(self, min_num: int, max_num: int):
        self.min_num = min_num
        self.max_num = max_num
    
    def analyze_frequencies(self, draws: List[List[int]]) -> Dict[int, float]:
        """FFT alapú frekvencia elemzés"""
        scores = defaultdict(float)
        
        try:
            # Convert draws to time series
            time_series = []
            for draw in draws:
                time_series.extend(sorted(draw))
            
            if len(time_series) < 8:
                return scores
            
            # FFT analysis
            fft_result = np.fft.fft(time_series)
            frequencies = np.abs(fft_result)
            
            # Find dominant frequencies
            dominant_indices = np.argsort(frequencies)[-5:]  # Top 5
            
            for idx in dominant_indices:
                if idx > 0:  # Skip DC component
                    # Project harmonic values
                    harmonic_value = int(frequencies[idx] % (self.max_num - self.min_num + 1) + self.min_num)
                    if self.min_num <= harmonic_value <= self.max_num:
                        scores[harmonic_value] += frequencies[idx] / np.sum(frequencies)
            
            # Harmonic resonance
            for i, num in enumerate(time_series[-10:]):  # Last 10 numbers
                for harmonic in [2, 3, 5]:  # Common harmonics
                    resonant = int(num * harmonic / harmonic)  # Simplified
                    if self.min_num <= resonant <= self.max_num:
                        scores[resonant] += 0.5 / (i + 1)
        
        except Exception:
            pass  # Silent fail for robustness
        
        return scores


class OptimizedHybridPredictor:
    """Optimalizált hibrid predikció főosztály"""
    
    def __init__(self, min_num: int, max_num: int):
        self.min_num = min_num
        self.max_num = max_num
        self.quantum_layer = FastQuantumLayer(8)
        self.pattern_engine = PatternEvolutionEngine(min_num, max_num)
        self.frequency_analyzer = FrequencyHarmonicsAnalyzer(min_num, max_num)
    
    def predict_numbers(self, draws: List[List[int]], required_count: int) -> List[int]:
        """Fő predikciós függvény"""
        if len(draws) < 5:
            return sorted(random.sample(range(self.min_num, self.max_num + 1), required_count))
        
        # Gyűjtjük a pontszámokat különböző módszerekkel
        final_scores = defaultdict(float)
        
        # 1. Frekvencia alapú scoring (40% súly)
        freq_counter = Counter()
        for draw in draws[:30]:  # Utolsó 30 húzás
            for num in draw:
                freq_counter[num] += 1
        
        for num, count in freq_counter.items():
            if self.min_num <= num <= self.max_num:
                final_scores[num] += count * 0.4
        
        # 2. Pattern evolution scoring (35% súly)
        pattern_scores = self.pattern_engine.analyze_patterns(draws)
        for num, score in pattern_scores.items():
            final_scores[num] += score * 0.35
        
        # 3. Frequency harmonics scoring (25% súly)
        harmonic_scores = self.frequency_analyzer.analyze_frequencies(draws)
        for num, score in harmonic_scores.items():
            final_scores[num] += score * 0.25
        
        # 4. Quantum resonance enhancement
        if draws:
            recent_draw = np.array(draws[0])
            if len(recent_draw) > 0:
                quantum_influence = self.quantum_layer.process(recent_draw / self.max_num)
                # Ensure quantum_influence is iterable
                if np.isscalar(quantum_influence):
                    quantum_influence = [quantum_influence]
                elif not hasattr(quantum_influence, '__iter__'):
                    quantum_influence = [quantum_influence]
                
                for i, influence in enumerate(quantum_influence):
                    projected_num = int((i / len(quantum_influence)) * (self.max_num - self.min_num) + self.min_num)
                    if self.min_num <= projected_num <= self.max_num:
                        final_scores[projected_num] += influence * 0.1
        
        # 5. Recent trend enhancement
        if len(draws) >= 3:
            recent_numbers = []
            for draw in draws[:3]:
                recent_numbers.extend(draw)
            
            recent_counter = Counter(recent_numbers)
            for num, count in recent_counter.items():
                if self.min_num <= num <= self.max_num:
                    final_scores[num] += count * 0.2  # Recent boost
        
        # Kiválasztás
        if not final_scores:
            return sorted(random.sample(range(self.min_num, self.max_num + 1), required_count))
        
        # Rendezés pontszám szerint
        sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_numbers = []
        for num, score in sorted_candidates:
            if self.min_num <= num <= self.max_num and num not in selected_numbers:
                selected_numbers.append(num)
                if len(selected_numbers) >= required_count:
                    break
        
        # Feltöltés ha szükséges
        while len(selected_numbers) < required_count:
            candidates = list(range(self.min_num, self.max_num + 1))
            remaining = [x for x in candidates if x not in selected_numbers]
            if remaining:
                # Weighted random selection from remaining
                weights = [final_scores.get(x, 0.1) for x in remaining]
                if sum(weights) > 0:
                    chosen = np.random.choice(remaining, p=np.array(weights)/sum(weights))
                    selected_numbers.append(chosen)
                else:
                    selected_numbers.append(random.choice(remaining))
            else:
                break
        
        return sorted(selected_numbers[:required_count])


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Optimalizált hibrid predikció mindkét számtípusra
    """
    # Múltbeli húzások lekérése (optimalizált mennyiség)
    past_draws = list(
        lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:50].values_list('lottery_type_number', flat=True)
    )
    
    # Szűrés és tisztítás
    valid_draws = []
    for draw in past_draws:
        if isinstance(draw, list) and len(draw) > 0:
            valid_draws.append([int(x) for x in draw])
    
    # Fő számok generálása
    main_predictor = OptimizedHybridPredictor(
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number)
    )
    
    main_numbers = main_predictor.predict_numbers(
        valid_draws, 
        int(lottery_type_instance.pieces_of_draw_numbers)
    )
    
    # Kiegészítő számok generálása
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_draws = list(
            lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance
            ).order_by('-id')[:50].values_list('additional_numbers', flat=True)
        )
        
        valid_additional_draws = []
        for draw in additional_draws:
            if isinstance(draw, list) and len(draw) > 0:
                valid_additional_draws.append([int(x) for x in draw])
        
        additional_predictor = OptimizedHybridPredictor(
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number)
        )
        
        additional_numbers = additional_predictor.predict_numbers(
            valid_additional_draws,
            int(lottery_type_instance.additional_numbers_count)
        )
    
    return main_numbers, additional_numbers 