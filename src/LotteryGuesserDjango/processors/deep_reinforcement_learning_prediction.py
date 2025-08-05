# deep_reinforcement_learning_prediction.py
"""
Mély Megerősítéses Tanulás Predikció
Q-Learning és Deep Q-Network alapú lottószám predikció
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque, Counter
import random
import math
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class DeepQLearningPredictor:
    """
    Mély Q-Learning alapú lottó prediktor
    """
    
    def __init__(self):
        self.state_size = 20  # Állapot méret
        self.action_size = 90  # Akció méret (max lottószám)
        self.memory_size = 10000
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
        # Memória és Q-táblázat
        self.memory = deque(maxlen=self.memory_size)
        self.q_table = {}
        self.reward_history = deque(maxlen=1000)
        
        # Állapot reprezentáció
        self.state_features = [
            'frequency', 'recency', 'gap', 'position', 'sum_category',
            'parity', 'divisibility', 'trend', 'volatility', 'correlation'
        ]
    
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Főbejárási pont a mély Q-learning predikcióhoz
        """
        try:
            # Fő számok generálása
            main_numbers = self._generate_rl_numbers(
                lottery_type_instance,
                min_num=int(lottery_type_instance.min_number),
                max_num=int(lottery_type_instance.max_number),
                required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
                is_main=True
            )
            
            # Kiegészítő számok generálása
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._generate_rl_numbers(
                    lottery_type_instance,
                    min_num=int(lottery_type_instance.additional_min_number),
                    max_num=int(lottery_type_instance.additional_max_number),
                    required_numbers=int(lottery_type_instance.additional_numbers_count),
                    is_main=False
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
            
        except Exception as e:
            print(f"Hiba a deep_reinforcement_learning_prediction-ben: {e}")
            return self._generate_fallback_numbers(lottery_type_instance)
    
    def _generate_rl_numbers(self, lottery_type_instance: lg_lottery_type,
                           min_num: int, max_num: int, required_numbers: int,
                           is_main: bool) -> List[int]:
        """
        Megerősítéses tanulás alapú számgenerálás
        """
        # Történeti adatok lekérése
        historical_data = self._get_historical_data(lottery_type_instance, is_main)
        
        if len(historical_data) < 20:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Q-learning tanítás
        self._train_q_learning(historical_data, min_num, max_num)
        
        # Predikció generálása
        current_state = self._get_current_state(historical_data, min_num, max_num)
        predicted_numbers = self._predict_with_q_learning(
            current_state, min_num, max_num, required_numbers
        )
        
        # Ensemble módszerrel kombinálás
        ensemble_numbers = self._ensemble_with_other_methods(
            historical_data, predicted_numbers, min_num, max_num, required_numbers
        )
        
        return ensemble_numbers
    
    def _train_q_learning(self, historical_data: List[List[int]], 
                         min_num: int, max_num: int) -> None:
        """
        Q-learning modell tanítása
        """
        # Tapasztalatok gyűjtése
        experiences = self._collect_experiences(historical_data, min_num, max_num)
        
        # Q-learning iterációk
        for episode in range(min(100, len(experiences))):
            if episode < len(experiences):
                state, action, reward, next_state = experiences[episode]
                
                # Q-value frissítés
                self._update_q_value(state, action, reward, next_state)
                
                # Epsilon decay
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
    
    def _collect_experiences(self, historical_data: List[List[int]], 
                           min_num: int, max_num: int) -> List[Tuple]:
        """
        Tapasztalatok gyűjtése a történeti adatokból
        """
        experiences = []
        
        for i in range(len(historical_data) - 1):
            current_draw = historical_data[i]
            next_draw = historical_data[i + 1]
            
            # Állapot reprezentáció
            state = self._encode_state(current_draw, historical_data[:i+1], min_num, max_num)
            
            # Akciók (következő húzás számai)
            for num in next_draw:
                if min_num <= num <= max_num:
                    action = num - min_num  # Normalizálás
                    
                    # Jutalom számítás
                    reward = self._calculate_reward(num, current_draw, next_draw, historical_data[:i+1])
                    
                    # Következő állapot
                    next_state = self._encode_state(next_draw, historical_data[:i+2], min_num, max_num)
                    
                    experiences.append((state, action, reward, next_state))
        
        return experiences
    
    def _encode_state(self, current_draw: List[int], history: List[List[int]], 
                     min_num: int, max_num: int) -> str:
        """
        Állapot kódolása
        """
        features = []
        
        # Frekvencia jellemzők
        all_numbers = [num for draw in history[-10:] for num in draw]  # Utolsó 10 húzás
        freq_counter = Counter(all_numbers)
        
        # Jellemzők számítása
        features.extend([
            np.mean(current_draw),  # Átlag
            np.std(current_draw),   # Szórás
            sum(current_draw),      # Összeg
            len(current_draw),      # Darabszám
            sum(1 for num in current_draw if num % 2 == 0),  # Páros számok
        ])
        
        # Frekvencia kategóriák
        for num in current_draw:
            freq_category = min(4, freq_counter.get(num, 0))  # Max 4 kategória
            features.append(freq_category)
        
        # Állapot hash
        state_hash = hash(tuple(features[:self.state_size]))
        return str(state_hash)
    
    def _calculate_reward(self, number: int, current_draw: List[int], 
                         next_draw: List[int], history: List[List[int]]) -> float:
        """
        Jutalom számítás
        """
        reward = 0.0
        
        # Alapjutalom: ha a szám megjelenik a következő húzásban
        if number in next_draw:
            reward += 10.0
        
        # Frekvencia alapú jutalom
        all_numbers = [num for draw in history[-20:] for num in draw]
        freq = all_numbers.count(number)
        avg_freq = len(all_numbers) / (max(all_numbers) - min(all_numbers) + 1) if all_numbers else 0
        
        if freq > avg_freq:
            reward += 2.0
        elif freq < avg_freq * 0.5:
            reward -= 1.0
        
        # Gap alapú jutalom
        last_seen = -1
        for i, draw in enumerate(reversed(history)):
            if number in draw:
                last_seen = i
                break
        
        if last_seen > 5:  # Régen nem jelent meg
            reward += 1.0
        elif last_seen == 0:  # Éppen most jelent meg
            reward -= 0.5
        
        # Trend alapú jutalom
        recent_freq = sum(1 for draw in history[-5:] for num in draw if num == number)
        older_freq = sum(1 for draw in history[-15:-5] for num in draw if num == number)
        
        if recent_freq > older_freq:
            reward += 1.0
        elif recent_freq < older_freq:
            reward -= 0.5
        
        return reward
    
    def _update_q_value(self, state: str, action: int, reward: float, next_state: str) -> None:
        """
        Q-value frissítés
        """
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # Következő állapot legjobb Q-value-ja
        next_q_values = self.q_table.get(next_state, {})
        max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Q-learning frissítés
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
        
        # Jutalom történet frissítése
        self.reward_history.append(reward)
    
    def _predict_with_q_learning(self, state: str, min_num: int, max_num: int, 
                               required_numbers: int) -> List[int]:
        """
        Q-learning alapú predikció
        """
        if state not in self.q_table:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Q-values lekérése
        q_values = self.q_table[state]
        
        # Epsilon-greedy stratégia
        selected_numbers = []
        available_actions = list(range(max_num - min_num + 1))
        
        for _ in range(required_numbers):
            if random.random() < self.epsilon:
                # Exploration: véletlen választás
                if available_actions:
                    action = random.choice(available_actions)
                    available_actions.remove(action)
                else:
                    action = random.randint(0, max_num - min_num)
            else:
                # Exploitation: legjobb Q-value
                best_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
                action = None
                
                for act, _ in best_actions:
                    if act in available_actions:
                        action = act
                        available_actions.remove(act)
                        break
                
                if action is None:
                    if available_actions:
                        action = random.choice(available_actions)
                        available_actions.remove(action)
                    else:
                        action = random.randint(0, max_num - min_num)
            
            number = action + min_num
            if min_num <= number <= max_num:
                selected_numbers.append(number)
        
        return selected_numbers
    
    def _ensemble_with_other_methods(self, historical_data: List[List[int]], 
                                   rl_numbers: List[int], min_num: int, max_num: int, 
                                   required_numbers: int) -> List[int]:
        """
        Ensemble más módszerekkel
        """
        # Frekvencia alapú módszer
        freq_numbers = self._frequency_method(historical_data, min_num, max_num, required_numbers)
        
        # Gap alapú módszer
        gap_numbers = self._gap_method(historical_data, min_num, max_num, required_numbers)
        
        # Trend alapú módszer
        trend_numbers = self._trend_method(historical_data, min_num, max_num, required_numbers)
        
        # Ensemble szavazás
        vote_counter = Counter()
        
        # Súlyozott szavazás
        methods = [
            (rl_numbers, 0.4),      # Q-learning legnagyobb súly
            (freq_numbers, 0.25),   # Frekvencia
            (gap_numbers, 0.2),     # Gap elemzés
            (trend_numbers, 0.15)   # Trend elemzés
        ]
        
        for numbers, weight in methods:
            for i, num in enumerate(numbers):
                position_weight = 1.0 / (i + 1)
                vote_counter[num] += weight * position_weight
        
        # Legmagasabb szavazatú számok
        top_numbers = [num for num, _ in vote_counter.most_common(required_numbers)]
        
        # Kiegészítés szükség esetén
        if len(top_numbers) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in top_numbers]
            random.shuffle(remaining)
            top_numbers.extend(remaining[:required_numbers - len(top_numbers)])
        
        return top_numbers[:required_numbers]
    
    def _frequency_method(self, historical_data: List[List[int]], 
                         min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """
        Frekvencia alapú módszer
        """
        counter = Counter()
        for draw in historical_data[:30]:  # Utolsó 30 húzás
            for num in draw:
                if min_num <= num <= max_num:
                    counter[num] += 1
        
        return [num for num, _ in counter.most_common(required_numbers)]
    
    def _gap_method(self, historical_data: List[List[int]], 
                   min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """
        Gap alapú módszer
        """
        last_seen = {}
        for i, draw in enumerate(historical_data):
            for num in draw:
                if min_num <= num <= max_num:
                    last_seen[num] = i
        
        gaps = {}
        for num in range(min_num, max_num + 1):
            gaps[num] = last_seen.get(num, len(historical_data))
        
        return sorted(gaps.keys(), key=lambda x: gaps[x], reverse=True)[:required_numbers]
    
    def _trend_method(self, historical_data: List[List[int]], 
                     min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """
        Trend alapú módszer
        """
        if len(historical_data) < 20:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        recent_freq = Counter(num for draw in historical_data[:10] for num in draw)
        older_freq = Counter(num for draw in historical_data[10:20] for num in draw)
        
        trends = {}
        for num in range(min_num, max_num + 1):
            recent_count = recent_freq.get(num, 0)
            older_count = older_freq.get(num, 0)
            trends[num] = recent_count - older_count * 0.5
        
        return sorted(trends.keys(), key=lambda x: trends[x], reverse=True)[:required_numbers]
    
    def _get_current_state(self, historical_data: List[List[int]], 
                          min_num: int, max_num: int) -> str:
        """
        Jelenlegi állapot lekérése
        """
        if not historical_data:
            return "empty_state"
        
        return self._encode_state(historical_data[0], historical_data, min_num, max_num)
    
    def _generate_smart_random(self, min_num: int, max_num: int, count: int) -> List[int]:
        """
        Intelligens véletlen generálás
        """
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
        """
        Történeti adatok lekérése
        """
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
        """
        Fallback számgenerálás
        """
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
deep_ql_predictor = DeepQLearningPredictor()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Főbejárási pont a mély Q-learning predikcióhoz
    """
    return deep_ql_predictor.get_numbers(lottery_type_instance)
