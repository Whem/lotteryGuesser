# advanced_bayesian_network_prediction.py
"""
Fejlett Bayesi Hálózat Predikció
Bayesi következtetés és valószínűségi grafikus modellek használata
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import random
import math
from scipy.stats import beta, gamma, norm
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class AdvancedBayesianNetworkPredictor:
    """
    Fejlett Bayesi hálózat predikció
    """
    
    def __init__(self):
        self.prior_alpha = 1.0  # Beta eloszlás alpha paramétere
        self.prior_beta = 1.0   # Beta eloszlás beta paramétere
        self.smoothing_factor = 0.1
        self.confidence_threshold = 0.7
        self.max_parents = 3  # Maximális szülő csomópontok száma
        
        # Bayesi hálózat struktúra
        self.network_structure = {}
        self.conditional_probabilities = {}
        self.marginal_probabilities = {}
        self.evidence_weights = {}
        
        # Hierarchikus Bayesi paraméterek
        self.hyperpriors = {
            'frequency_shape': 2.0,
            'frequency_rate': 1.0,
            'recency_mean': 0.0,
            'recency_std': 1.0
        }
    
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Főbejárási pont a Bayesi hálózat predikcióhoz
        """
        try:
            # Fő számok generálása
            main_numbers = self._generate_bayesian_numbers(
                lottery_type_instance,
                min_num=int(lottery_type_instance.min_number),
                max_num=int(lottery_type_instance.max_number),
                required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
                is_main=True
            )
            
            # Kiegészítő számok generálása
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._generate_bayesian_numbers(
                    lottery_type_instance,
                    min_num=int(lottery_type_instance.additional_min_number),
                    max_num=int(lottery_type_instance.additional_max_number),
                    required_numbers=int(lottery_type_instance.additional_numbers_count),
                    is_main=False
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
            
        except Exception as e:
            print(f"Hiba az advanced_bayesian_network_prediction-ben: {e}")
            return self._generate_fallback_numbers(lottery_type_instance)
    
    def _generate_bayesian_numbers(self, lottery_type_instance: lg_lottery_type,
                                 min_num: int, max_num: int, required_numbers: int,
                                 is_main: bool) -> List[int]:
        """
        Bayesi hálózat alapú számgenerálás
        """
        # Történeti adatok lekérése
        historical_data = self._get_historical_data(lottery_type_instance, is_main)
        
        if len(historical_data) < 20:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Bayesi hálózat építése
        self._build_bayesian_network(historical_data, min_num, max_num)
        
        # Prior valószínűségek számítása
        self._calculate_priors(historical_data, min_num, max_num)
        
        # Feltételes valószínűségek számítása
        self._calculate_conditional_probabilities(historical_data, min_num, max_num)
        
        # Bayesi következtetés
        posterior_probabilities = self._bayesian_inference(historical_data, min_num, max_num)
        
        # Hierarchikus Bayesi frissítés
        updated_probabilities = self._hierarchical_bayesian_update(
            posterior_probabilities, historical_data, min_num, max_num
        )
        
        # Számok kiválasztása
        selected_numbers = self._select_numbers_by_probability(
            updated_probabilities, min_num, max_num, required_numbers
        )
        
        # Monte Carlo mintavételezés
        mc_numbers = self._monte_carlo_sampling(
            updated_probabilities, min_num, max_num, required_numbers
        )
        
        # Ensemble kombinálás
        final_numbers = self._combine_bayesian_results(
            selected_numbers, mc_numbers, min_num, max_num, required_numbers
        )
        
        return final_numbers
    
    def _build_bayesian_network(self, historical_data: List[List[int]], 
                              min_num: int, max_num: int) -> None:
        """
        Bayesi hálózat struktúra építése
        """
        # Csomópontok: számok és jellemzők
        nodes = list(range(min_num, max_num + 1))
        feature_nodes = ['sum', 'mean', 'std', 'parity', 'trend']
        
        # Hálózat struktúra inicializálása
        self.network_structure = {node: [] for node in nodes + feature_nodes}
        
        # Függőségek felderítése mutual information alapján
        dependencies = self._discover_dependencies(historical_data, min_num, max_num)
        
        # Hálózat struktúra építése
        for child, parents in dependencies.items():
            if len(parents) <= self.max_parents:
                self.network_structure[child] = parents
    
    def _discover_dependencies(self, historical_data: List[List[int]], 
                             min_num: int, max_num: int) -> Dict:
        """
        Függőségek felderítése a történeti adatokból
        """
        dependencies = {}
        
        # Számok közötti függőségek
        for num in range(min_num, max_num + 1):
            correlations = {}
            
            for other_num in range(min_num, max_num + 1):
                if num != other_num:
                    correlation = self._calculate_mutual_information(
                        num, other_num, historical_data
                    )
                    correlations[other_num] = correlation
            
            # Legerősebb korrelációk kiválasztása
            strong_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            dependencies[num] = [parent for parent, corr in strong_correlations[:self.max_parents] 
                               if corr > 0.1]
        
        return dependencies
    
    def _calculate_mutual_information(self, num1: int, num2: int, 
                                    historical_data: List[List[int]]) -> float:
        """
        Mutual information számítás két szám között
        """
        # Együttes előfordulás számítása
        both_present = 0
        num1_present = 0
        num2_present = 0
        total_draws = len(historical_data)
        
        for draw in historical_data:
            num1_in_draw = num1 in draw
            num2_in_draw = num2 in draw
            
            if num1_in_draw and num2_in_draw:
                both_present += 1
            if num1_in_draw:
                num1_present += 1
            if num2_in_draw:
                num2_present += 1
        
        # Valószínűségek
        p_both = (both_present + self.smoothing_factor) / (total_draws + 4 * self.smoothing_factor)
        p_num1 = (num1_present + self.smoothing_factor) / (total_draws + 2 * self.smoothing_factor)
        p_num2 = (num2_present + self.smoothing_factor) / (total_draws + 2 * self.smoothing_factor)
        
        # Mutual information
        if p_both > 0 and p_num1 > 0 and p_num2 > 0:
            mi = p_both * math.log(p_both / (p_num1 * p_num2))
            return max(0, mi)
        
        return 0.0
    
    def _calculate_priors(self, historical_data: List[List[int]], 
                         min_num: int, max_num: int) -> None:
        """
        Prior valószínűségek számítása
        """
        self.marginal_probabilities = {}
        
        # Számok előfordulási gyakorisága
        frequency_counter = Counter()
        for draw in historical_data:
            for num in draw:
                if min_num <= num <= max_num:
                    frequency_counter[num] += 1
        
        total_occurrences = sum(frequency_counter.values())
        
        # Beta-Binomial konjugált prior
        for num in range(min_num, max_num + 1):
            count = frequency_counter.get(num, 0)
            
            # Bayesi frissítés
            posterior_alpha = self.prior_alpha + count
            posterior_beta = self.prior_beta + total_occurrences - count
            
            # Várt érték
            expected_prob = posterior_alpha / (posterior_alpha + posterior_beta)
            self.marginal_probabilities[num] = expected_prob
    
    def _calculate_conditional_probabilities(self, historical_data: List[List[int]], 
                                           min_num: int, max_num: int) -> None:
        """
        Feltételes valószínűségek számítása
        """
        self.conditional_probabilities = {}
        
        for child, parents in self.network_structure.items():
            if isinstance(child, int) and min_num <= child <= max_num:
                self.conditional_probabilities[child] = {}
                
                if not parents:
                    # Nincs szülő, marginális valószínűség
                    self.conditional_probabilities[child]['marginal'] = self.marginal_probabilities.get(child, 0.1)
                else:
                    # Feltételes valószínűségek számítása
                    self._calculate_conditional_for_node(child, parents, historical_data)
    
    def _calculate_conditional_for_node(self, child: int, parents: List[int], 
                                      historical_data: List[List[int]]) -> None:
        """
        Egy csomópont feltételes valószínűségeinek számítása
        """
        conditional_counts = defaultdict(lambda: defaultdict(int))
        parent_counts = defaultdict(int)
        
        for draw in historical_data:
            child_present = child in draw
            
            # Szülők állapotának meghatározása
            parent_state = tuple(parent in draw for parent in parents)
            
            conditional_counts[parent_state][child_present] += 1
            parent_counts[parent_state] += 1
        
        # Feltételes valószínűségek számítása
        for parent_state, counts in conditional_counts.items():
            total_count = parent_counts[parent_state]
            
            if total_count > 0:
                # Bayesi simítás
                true_count = counts[True] + self.smoothing_factor
                false_count = counts[False] + self.smoothing_factor
                total_smoothed = total_count + 2 * self.smoothing_factor
                
                prob_true = true_count / total_smoothed
                
                self.conditional_probabilities[child][parent_state] = prob_true
            else:
                # Alapértelmezett valószínűség
                self.conditional_probabilities[child][parent_state] = self.marginal_probabilities.get(child, 0.1)
    
    def _bayesian_inference(self, historical_data: List[List[int]], 
                          min_num: int, max_num: int) -> Dict[int, float]:
        """
        Bayesi következtetés végrehajtása
        """
        # Jelenlegi evidencia (legutóbbi húzások)
        recent_evidence = self._extract_evidence(historical_data[:5])  # Utolsó 5 húzás
        
        # Posterior valószínűségek számítása
        posterior_probs = {}
        
        for num in range(min_num, max_num + 1):
            # Prior valószínűség
            prior = self.marginal_probabilities.get(num, 0.1)
            
            # Likelihood számítás
            likelihood = self._calculate_likelihood(num, recent_evidence, historical_data)
            
            # Posterior ∝ Prior × Likelihood
            posterior = prior * likelihood
            posterior_probs[num] = posterior
        
        # Normalizálás
        total_posterior = sum(posterior_probs.values())
        if total_posterior > 0:
            for num in posterior_probs:
                posterior_probs[num] /= total_posterior
        
        return posterior_probs
    
    def _extract_evidence(self, recent_draws: List[List[int]]) -> Dict:
        """
        Evidencia kinyerése a legutóbbi húzásokból
        """
        evidence = {
            'recent_numbers': set(),
            'sum_trend': [],
            'frequency_pattern': Counter()
        }
        
        for draw in recent_draws:
            evidence['recent_numbers'].update(draw)
            evidence['sum_trend'].append(sum(draw))
            
            for num in draw:
                evidence['frequency_pattern'][num] += 1
        
        return evidence
    
    def _calculate_likelihood(self, num: int, evidence: Dict, 
                            historical_data: List[List[int]]) -> float:
        """
        Likelihood számítás egy számra az evidencia alapján
        """
        likelihood = 1.0
        
        # Evidencia alapú likelihood komponensek
        
        # 1. Legutóbbi megjelenés alapján
        if num in evidence['recent_numbers']:
            likelihood *= 0.8  # Csökkentett valószínűség (anti-clustering)
        else:
            likelihood *= 1.2  # Növelt valószínűség
        
        # 2. Frekvencia minta alapján
        recent_freq = evidence['frequency_pattern'].get(num, 0)
        if recent_freq > 0:
            likelihood *= (1.0 / (recent_freq + 1))  # Csökkentett ismétlődés
        
        # 3. Összeg trend alapján
        if evidence['sum_trend']:
            avg_sum = np.mean(evidence['sum_trend'])
            # Likelihood a szám hozzájárulása alapján az átlagos összeghez
            contribution_factor = abs(num - avg_sum / 5) / (avg_sum / 5)  # Normalizált távolság
            likelihood *= math.exp(-contribution_factor)  # Exponenciális csökkenés
        
        # 4. Hálózat alapú likelihood
        if num in self.network_structure:
            parents = self.network_structure[num]
            if parents:
                parent_evidence = tuple(parent in evidence['recent_numbers'] for parent in parents)
                conditional_prob = self.conditional_probabilities[num].get(parent_evidence, 0.1)
                likelihood *= conditional_prob
        
        return max(likelihood, 0.001)  # Minimum likelihood
    
    def _hierarchical_bayesian_update(self, posterior_probs: Dict[int, float], 
                                    historical_data: List[List[int]], 
                                    min_num: int, max_num: int) -> Dict[int, float]:
        """
        Hierarchikus Bayesi frissítés
        """
        updated_probs = {}
        
        # Hyperprior paraméterek frissítése
        self._update_hyperpriors(historical_data)
        
        for num, prob in posterior_probs.items():
            # Hierarchikus prior
            hierarchical_prior = self._calculate_hierarchical_prior(num, historical_data)
            
            # Hierarchikus posterior
            # Weighted kombináció
            alpha = 0.7  # Posterior súly
            beta = 0.3   # Hierarchikus prior súly
            
            updated_prob = alpha * prob + beta * hierarchical_prior
            updated_probs[num] = updated_prob
        
        # Normalizálás
        total_prob = sum(updated_probs.values())
        if total_prob > 0:
            for num in updated_probs:
                updated_probs[num] /= total_prob
        
        return updated_probs
    
    def _update_hyperpriors(self, historical_data: List[List[int]]) -> None:
        """
        Hyperprior paraméterek frissítése
        """
        # Frekvencia hyperprior frissítése
        all_frequencies = []
        for draw in historical_data:
            all_frequencies.extend(draw)
        
        if all_frequencies:
            freq_counter = Counter(all_frequencies)
            frequencies = list(freq_counter.values())
            
            # Gamma eloszlás paraméterek becslése
            if len(frequencies) > 1:
                mean_freq = np.mean(frequencies)
                var_freq = np.var(frequencies)
                
                if var_freq > 0:
                    self.hyperpriors['frequency_rate'] = mean_freq / var_freq
                    self.hyperpriors['frequency_shape'] = mean_freq * self.hyperpriors['frequency_rate']
    
    def _calculate_hierarchical_prior(self, num: int, historical_data: List[List[int]]) -> float:
        """
        Hierarchikus prior számítás
        """
        # Gamma-Poisson hierarchikus modell
        freq_count = sum(1 for draw in historical_data for n in draw if n == num)
        
        # Gamma prior paraméterek
        shape = self.hyperpriors['frequency_shape']
        rate = self.hyperpriors['frequency_rate']
        
        # Posterior gamma paraméterek
        posterior_shape = shape + freq_count
        posterior_rate = rate + len(historical_data)
        
        # Várt érték
        expected_rate = posterior_shape / posterior_rate
        
        # Normalizálás
        return expected_rate / (expected_rate + 1)
    
    def _select_numbers_by_probability(self, probabilities: Dict[int, float], 
                                     min_num: int, max_num: int, 
                                     required_numbers: int) -> List[int]:
        """
        Számok kiválasztása valószínűség alapján
        """
        # Valószínűség alapú rendezés
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Top számok kiválasztása
        selected = [num for num, _ in sorted_probs[:required_numbers]]
        
        return selected
    
    def _monte_carlo_sampling(self, probabilities: Dict[int, float], 
                            min_num: int, max_num: int, 
                            required_numbers: int) -> List[int]:
        """
        Monte Carlo mintavételezés
        """
        numbers = list(probabilities.keys())
        weights = list(probabilities.values())
        
        # Mintavételezés visszatevés nélkül
        selected = []
        remaining_numbers = numbers[:]
        remaining_weights = weights[:]
        
        for _ in range(required_numbers):
            if not remaining_numbers:
                break
            
            # Normalizálás
            total_weight = sum(remaining_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in remaining_weights]
            else:
                normalized_weights = [1.0 / len(remaining_weights)] * len(remaining_weights)
            
            # Mintavételezés
            chosen_idx = np.random.choice(len(remaining_numbers), p=normalized_weights)
            selected.append(remaining_numbers[chosen_idx])
            
            # Eltávolítás
            remaining_numbers.pop(chosen_idx)
            remaining_weights.pop(chosen_idx)
        
        return selected
    
    def _combine_bayesian_results(self, selected_numbers: List[int], mc_numbers: List[int],
                                min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """
        Bayesi eredmények kombinálása
        """
        # Szavazás alapú kombinálás
        vote_counter = Counter()
        
        # Determinisztikus kiválasztás súlyozása
        for i, num in enumerate(selected_numbers):
            weight = 1.0 / (i + 1)
            vote_counter[num] += weight * 0.6
        
        # Monte Carlo mintavételezés súlyozása
        for i, num in enumerate(mc_numbers):
            weight = 1.0 / (i + 1)
            vote_counter[num] += weight * 0.4
        
        # Legmagasabb szavazatú számok
        top_numbers = [num for num, _ in vote_counter.most_common(required_numbers)]
        
        # Kiegészítés szükség esetén
        if len(top_numbers) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in top_numbers]
            # Valószínűség alapú kiegészítés
            if hasattr(self, 'marginal_probabilities'):
                remaining_probs = [(num, self.marginal_probabilities.get(num, 0.1)) for num in remaining]
                remaining_probs.sort(key=lambda x: x[1], reverse=True)
                top_numbers.extend([num for num, _ in remaining_probs[:required_numbers - len(top_numbers)]])
            else:
                random.shuffle(remaining)
                top_numbers.extend(remaining[:required_numbers - len(top_numbers)])
        
        return top_numbers[:required_numbers]
    
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
            ).order_by('-id').values_list(field_name, flat=True)[:150]
            
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
bayesian_network_predictor = AdvancedBayesianNetworkPredictor()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Főbejárási pont a fejlett Bayesi hálózat predikcióhoz
    """
    return bayesian_network_predictor.get_numbers(lottery_type_instance)
