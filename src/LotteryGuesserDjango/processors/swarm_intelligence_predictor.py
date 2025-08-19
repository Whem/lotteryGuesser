# swarm_intelligence_predictor.py
"""
Swarm Intelligence Predictor
Raj intelligencia kombinálva hangyakolónia, méhraj és részecske raj algoritmusokkal
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Ant:
    """Hangya ágens az ACO algoritmushoz."""
    
    def __init__(self, colony_id: int, min_num: int, max_num: int, target_count: int):
        self.colony_id = colony_id
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        
        # Hangya állapota
        self.current_path = []
        self.visited_numbers = set()
        self.path_quality = 0.0
        
        # Hangya viselkedés
        self.alpha = 1.0  # Feromon fontosság
        self.beta = 2.0   # Heurisztikus információ fontosság
        
    def construct_solution(self, pheromone_matrix: Dict[Tuple[int, int], float], 
                          heuristic_info: Dict[int, float]) -> List[int]:
        """Megoldás konstruálása hangyaút alapján."""
        
        self.current_path = []
        self.visited_numbers = set()
        
        # Első szám választása
        if heuristic_info:
            start_number = max(heuristic_info.keys(), key=lambda x: heuristic_info[x])
        else:
            start_number = random.randint(self.min_num, self.max_num)
        
        self.current_path.append(start_number)
        self.visited_numbers.add(start_number)
        
        # További számok hozzáadása
        for _ in range(self.target_count - 1):
            next_number = self._choose_next_number(pheromone_matrix, heuristic_info)
            if next_number is not None:
                self.current_path.append(next_number)
                self.visited_numbers.add(next_number)
        
        return self.current_path.copy()
    
    def _choose_next_number(self, pheromone_matrix: Dict[Tuple[int, int], float], 
                           heuristic_info: Dict[int, float]) -> Optional[int]:
        """Következő szám választása valószínűség alapján."""
        
        if not self.current_path:
            return None
        
        current_number = self.current_path[-1]
        available_numbers = [num for num in range(self.min_num, self.max_num + 1) 
                           if num not in self.visited_numbers]
        
        if not available_numbers:
            return None
        
        # Valószínűségek számítása
        probabilities = []
        total_probability = 0.0
        
        for candidate in available_numbers:
            # Feromon értéke
            pheromone = pheromone_matrix.get((current_number, candidate), 0.1)
            
            # Heurisztikus információ
            heuristic = heuristic_info.get(candidate, 1.0)
            
            # Kombinált valószínűség
            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob)
            total_probability += prob
        
        # Normalizálás
        if total_probability > 0:
            probabilities = [p / total_probability for p in probabilities]
        else:
            probabilities = [1.0 / len(available_numbers)] * len(available_numbers)
        
        # Roulette wheel selection
        cumulative_prob = 0.0
        random_value = random.random()
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if random_value <= cumulative_prob:
                return available_numbers[i]
        
        return available_numbers[-1] if available_numbers else None
    
    def evaluate_solution(self, past_draws: List[List[int]]) -> float:
        """Megoldás minőségének értékelése."""
        
        if not self.current_path or not past_draws:
            return 0.0
        
        # Történeti illeszkedés
        historical_score = 0.0
        for draw in past_draws[-20:]:  # Legutóbbi 20 húzás
            intersection = len(set(self.current_path) & set(draw))
            historical_score += intersection / self.target_count
        
        historical_score /= min(20, len(past_draws))
        
        # Diverzitás pont
        diversity_score = 0.0
        if len(self.current_path) > 1:
            min_distance = min(abs(self.current_path[i] - self.current_path[j]) 
                             for i in range(len(self.current_path)) 
                             for j in range(i + 1, len(self.current_path)))
            diversity_score = min_distance / (self.max_num - self.min_num)
        
        # Matematikai tulajdonságok
        math_score = 0.0
        prime_count = sum(1 for num in self.current_path if self._is_prime(num))
        fibonacci_count = sum(1 for num in self.current_path if self._is_fibonacci(num))
        math_score = (prime_count + fibonacci_count) / self.target_count * 0.1
        
        # Kombinált pontszám
        self.path_quality = historical_score * 0.6 + diversity_score * 0.3 + math_score * 0.1
        
        return self.path_quality
    
    def _is_prime(self, n: int) -> bool:
        """Prímszám ellenőrzés."""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _is_fibonacci(self, n: int) -> bool:
        """Fibonacci szám ellenőrzés."""
        fib_sequence = [1, 1]
        while fib_sequence[-1] < n:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        
        return n in fib_sequence


class AntColonyOptimization:
    """Hangyakolónia optimalizáció."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int, 
                 num_ants: int = 20, max_iterations: int = 50):
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        
        # Feromon mátrix
        self.pheromone_matrix = defaultdict(lambda: 0.1)
        
        # ACO paraméterek
        self.evaporation_rate = 0.1
        self.pheromone_deposit = 1.0
        
        # Hangya kolónia
        self.ants = [Ant(i, min_num, max_num, target_count) for i in range(num_ants)]
        
        # Legjobb megoldások
        self.best_solution = []
        self.best_quality = 0.0
        self.solution_history = deque(maxlen=50)
    
    def optimize(self, past_draws: List[List[int]], heuristic_info: Dict[int, float]) -> List[int]:
        """ACO optimalizáció futtatása."""
        
        for iteration in range(self.max_iterations):
            # Minden hangya konstruál egy megoldást
            iteration_solutions = []
            
            for ant in self.ants:
                solution = ant.construct_solution(self.pheromone_matrix, heuristic_info)
                quality = ant.evaluate_solution(past_draws)
                iteration_solutions.append((solution, quality, ant))
            
            # Legjobb megoldás frissítése
            iteration_best = max(iteration_solutions, key=lambda x: x[1])
            if iteration_best[1] > self.best_quality:
                self.best_quality = iteration_best[1]
                self.best_solution = iteration_best[0].copy()
            
            # Feromon frissítés
            self._update_pheromones(iteration_solutions)
            
            # Konvergencia ellenőrzés
            self.solution_history.append(iteration_best[1])
            if len(self.solution_history) >= 10:
                recent_improvement = max(list(self.solution_history)[-10:]) - min(list(self.solution_history)[-10:])
                if recent_improvement < 0.01:  # Kis javulás
                    break
        
        return self.best_solution if self.best_solution else self._generate_fallback()
    
    def _update_pheromones(self, solutions: List[Tuple[List[int], float, Ant]]):
        """Feromon szintek frissítése."""
        
        # Elpárolgás
        for key in list(self.pheromone_matrix.keys()):
            self.pheromone_matrix[key] *= (1 - self.evaporation_rate)
            if self.pheromone_matrix[key] < 0.01:
                self.pheromone_matrix[key] = 0.01
        
        # Feromon lerakás
        for solution, quality, ant in solutions:
            if len(solution) > 1:
                deposit_amount = self.pheromone_deposit * quality
                
                for i in range(len(solution) - 1):
                    edge = (solution[i], solution[i + 1])
                    self.pheromone_matrix[edge] += deposit_amount
                    
                    # Szimmetrikus mátrix
                    reverse_edge = (solution[i + 1], solution[i])
                    self.pheromone_matrix[reverse_edge] += deposit_amount
    
    def _generate_fallback(self) -> List[int]:
        """Fallback megoldás."""
        return random.sample(range(self.min_num, self.max_num + 1), self.target_count)


class Bee:
    """Méh ágens a méhraj algoritmushoz."""
    
    def __init__(self, bee_id: int, min_num: int, max_num: int, target_count: int):
        self.bee_id = bee_id
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        
        # Méh állapota
        self.solution = random.sample(range(min_num, max_num + 1), target_count)
        self.fitness = 0.0
        self.trial_count = 0
        self.max_trials = 10
        
    def local_search(self, past_draws: List[List[int]]):
        """Helyi keresés a méh által."""
        
        # Véletlen módosítás
        if self.solution:
            modify_index = random.randint(0, len(self.solution) - 1)
            
            # Új szám generálása
            new_number = random.randint(self.min_num, self.max_num)
            while new_number in self.solution:
                new_number = random.randint(self.min_num, self.max_num)
            
            # Próba megoldás
            trial_solution = self.solution.copy()
            trial_solution[modify_index] = new_number
            
            # Fitness értékelés
            trial_fitness = self._evaluate_fitness(trial_solution, past_draws)
            
            # Elfogadás/elutasítás
            if trial_fitness > self.fitness:
                self.solution = trial_solution
                self.fitness = trial_fitness
                self.trial_count = 0
            else:
                self.trial_count += 1
    
    def _evaluate_fitness(self, solution: List[int], past_draws: List[List[int]]) -> float:
        """Fitness értékelés."""
        
        if not past_draws:
            return random.random()
        
        # Történeti egyezés
        historical_fitness = 0.0
        for draw in past_draws[-15:]:
            overlap = len(set(solution) & set(draw))
            historical_fitness += overlap / self.target_count
        
        historical_fitness /= min(15, len(past_draws))
        
        # Egyenletesség (spread)
        sorted_solution = sorted(solution)
        spread_fitness = 0.0
        if len(sorted_solution) > 1:
            gaps = [sorted_solution[i+1] - sorted_solution[i] for i in range(len(sorted_solution)-1)]
            avg_gap = sum(gaps) / len(gaps)
            expected_gap = (self.max_num - self.min_num) / self.target_count
            spread_fitness = 1.0 - abs(avg_gap - expected_gap) / expected_gap
        
        return historical_fitness * 0.7 + spread_fitness * 0.3


class BeeColonyOptimization:
    """Méhraj optimalizáció."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int, 
                 num_bees: int = 30, max_iterations: int = 100):
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        self.num_bees = num_bees
        self.max_iterations = max_iterations
        
        # Méh populáció
        self.employed_bees = [Bee(i, min_num, max_num, target_count) for i in range(num_bees // 2)]
        self.onlooker_bees = [Bee(i + num_bees // 2, min_num, max_num, target_count) for i in range(num_bees // 2)]
        
        # Legjobb megoldás
        self.best_solution = []
        self.best_fitness = 0.0
    
    def optimize(self, past_draws: List[List[int]]) -> List[int]:
        """Méhraj optimalizáció futtatása."""
        
        # Kezdeti fitness értékelés
        for bee in self.employed_bees:
            bee.fitness = bee._evaluate_fitness(bee.solution, past_draws)
        
        for iteration in range(self.max_iterations):
            # Employed bee fázis
            for bee in self.employed_bees:
                bee.local_search(past_draws)
            
            # Onlooker bee fázis
            self._onlooker_phase(past_draws)
            
            # Scout bee fázis
            self._scout_phase()
            
            # Legjobb megoldás frissítése
            current_best = max(self.employed_bees, key=lambda x: x.fitness)
            if current_best.fitness > self.best_fitness:
                self.best_fitness = current_best.fitness
                self.best_solution = current_best.solution.copy()
        
        return self.best_solution if self.best_solution else self._generate_fallback()
    
    def _onlooker_phase(self, past_draws: List[List[int]]):
        """Onlooker méhek fázisa."""
        
        # Valószínűségek számítása
        total_fitness = sum(bee.fitness for bee in self.employed_bees)
        probabilities = [bee.fitness / total_fitness if total_fitness > 0 else 1.0 / len(self.employed_bees) 
                        for bee in self.employed_bees]
        
        for onlooker in self.onlooker_bees:
            # Employed bee kiválasztása roulette wheel-lel
            selected_bee = self._roulette_wheel_selection(self.employed_bees, probabilities)
            
            # Onlooker követi a kiválasztott employed bee-t
            onlooker.solution = selected_bee.solution.copy()
            onlooker.fitness = selected_bee.fitness
            onlooker.local_search(past_draws)
    
    def _scout_phase(self):
        """Scout méhek fázisa."""
        
        for bee in self.employed_bees:
            if bee.trial_count >= bee.max_trials:
                # Új véletlen megoldás
                bee.solution = random.sample(range(self.min_num, self.max_num + 1), self.target_count)
                bee.trial_count = 0
                bee.fitness = 0.0
    
    def _roulette_wheel_selection(self, bees: List[Bee], probabilities: List[float]) -> Bee:
        """Roulette wheel kiválasztás."""
        
        random_value = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if random_value <= cumulative_prob:
                return bees[i]
        
        return bees[-1]
    
    def _generate_fallback(self) -> List[int]:
        """Fallback megoldás."""
        return random.sample(range(self.min_num, self.max_num + 1), self.target_count)


class Particle:
    """Részecske a PSO algoritmushoz."""
    
    def __init__(self, particle_id: int, min_num: int, max_num: int, target_count: int):
        self.particle_id = particle_id
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        
        # Pozíció (valós értékű, majd diszkretizált)
        self.position = np.random.uniform(min_num, max_num, target_count)
        self.velocity = np.random.uniform(-1, 1, target_count)
        
        # Személyes legjobb
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = 0.0
        
        # Aktuális fitness
        self.fitness = 0.0
        
    def update_velocity(self, global_best_position: np.ndarray, 
                       w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """Sebesség frissítése."""
        
        r1 = np.random.random(self.target_count)
        r2 = np.random.random(self.target_count)
        
        # PSO sebesség frissítési egyenlet
        cognitive_component = c1 * r1 * (self.personal_best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive_component + social_component
        
        # Sebesség korlátozás
        max_velocity = (self.max_num - self.min_num) * 0.1
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
    
    def update_position(self):
        """Pozíció frissítése."""
        
        self.position += self.velocity
        
        # Pozíció korlátozás
        self.position = np.clip(self.position, self.min_num, self.max_num)
    
    def discretize_position(self) -> List[int]:
        """Pozíció diszkretizálása egész számokra."""
        
        # Kerekítés és duplikátumok kezelése
        discrete_position = [int(round(pos)) for pos in self.position]
        discrete_position = list(set(discrete_position))
        
        # Kiegészítés ha szükséges
        while len(discrete_position) < self.target_count:
            new_num = random.randint(self.min_num, self.max_num)
            if new_num not in discrete_position:
                discrete_position.append(new_num)
        
        return discrete_position[:self.target_count]
    
    def evaluate_fitness(self, past_draws: List[List[int]]) -> float:
        """Fitness értékelés."""
        
        solution = self.discretize_position()
        
        if not past_draws:
            return random.random()
        
        # Történeti teljesítmény
        historical_score = 0.0
        for draw in past_draws[-10:]:
            matches = len(set(solution) & set(draw))
            historical_score += matches / self.target_count
        
        historical_score /= min(10, len(past_draws))
        
        # Diverzitás
        diversity_score = 0.0
        if len(solution) > 1:
            sorted_solution = sorted(solution)
            avg_gap = sum(sorted_solution[i+1] - sorted_solution[i] 
                         for i in range(len(sorted_solution)-1)) / (len(sorted_solution)-1)
            expected_gap = (self.max_num - self.min_num) / self.target_count
            diversity_score = 1.0 - abs(avg_gap - expected_gap) / max(expected_gap, 1)
        
        self.fitness = historical_score * 0.8 + diversity_score * 0.2
        
        # Személyes legjobb frissítése
        if self.fitness > self.personal_best_fitness:
            self.personal_best_fitness = self.fitness
            self.personal_best_position = self.position.copy()
        
        return self.fitness


class ParticleSwarmOptimization:
    """Részecske raj optimalizáció."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int, 
                 num_particles: int = 25, max_iterations: int = 100):
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        
        # Részecske raj
        self.particles = [Particle(i, min_num, max_num, target_count) for i in range(num_particles)]
        
        # Globális legjobb
        self.global_best_position = np.random.uniform(min_num, max_num, target_count)
        self.global_best_fitness = 0.0
        
        # PSO paraméterek
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
    
    def optimize(self, past_draws: List[List[int]]) -> List[int]:
        """PSO optimalizáció futtatása."""
        
        # Kezdeti értékelés
        for particle in self.particles:
            fitness = particle.evaluate_fitness(past_draws)
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
        
        for iteration in range(self.max_iterations):
            for particle in self.particles:
                # Sebesség és pozíció frissítése
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
                
                # Fitness értékelés
                fitness = particle.evaluate_fitness(past_draws)
                
                # Globális legjobb frissítése
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
            
            # Adaptív paraméterek
            self.w *= 0.99  # Csökkenő inertia
        
        # Legjobb pozíció diszkretizálása
        best_particle = max(self.particles, key=lambda x: x.fitness)
        return best_particle.discretize_position()


class SwarmIntelligencePredictor:
    """Raj intelligencia prediktor főosztály."""
    
    def __init__(self, min_num: int, max_num: int, target_count: int):
        self.min_num = min_num
        self.max_num = max_num
        self.target_count = target_count
        
        # Raj algoritmusok
        self.aco = AntColonyOptimization(min_num, max_num, target_count, 15, 30)
        self.bco = BeeColonyOptimization(min_num, max_num, target_count, 20, 50)
        self.pso = ParticleSwarmOptimization(min_num, max_num, target_count, 20, 50)
        
        # Ensemble súlyok
        self.algorithm_weights = {'aco': 0.4, 'bco': 0.3, 'pso': 0.3}
        self.performance_history = defaultdict(list)
    
    def predict(self, past_draws: List[List[int]]) -> List[int]:
        """Raj intelligencia alapú predikció."""
        
        if len(past_draws) < 5:
            return self._generate_swarm_fallback()
        
        # Heurisztikus információ ACO-hoz
        heuristic_info = self._calculate_heuristic_info(past_draws)
        
        # Minden algoritmus futtatása
        predictions = {}
        
        try:
            # Ant Colony Optimization
            aco_result = self.aco.optimize(past_draws, heuristic_info)
            predictions['aco'] = aco_result
        except Exception as e:
            logger.error(f"ACO hiba: {e}")
            predictions['aco'] = self._generate_swarm_fallback()
        
        try:
            # Bee Colony Optimization
            bco_result = self.bco.optimize(past_draws)
            predictions['bco'] = bco_result
        except Exception as e:
            logger.error(f"BCO hiba: {e}")
            predictions['bco'] = self._generate_swarm_fallback()
        
        try:
            # Particle Swarm Optimization
            pso_result = self.pso.optimize(past_draws)
            predictions['pso'] = pso_result
        except Exception as e:
            logger.error(f"PSO hiba: {e}")
            predictions['pso'] = self._generate_swarm_fallback()
        
        # Ensemble kombináció
        final_prediction = self._combine_swarm_predictions(predictions, past_draws)
        
        return sorted(final_prediction)
    
    def _calculate_heuristic_info(self, past_draws: List[List[int]]) -> Dict[int, float]:
        """Heurisztikus információ számítása ACO-hoz."""
        
        heuristic = defaultdict(float)
        
        # Frekvencia alapú heurisztika
        frequency = defaultdict(int)
        for draw in past_draws:
            for num in draw:
                if self.min_num <= num <= self.max_num:
                    frequency[num] += 1
        
        # Normalizálás
        max_freq = max(frequency.values()) if frequency else 1
        for num in range(self.min_num, self.max_num + 1):
            heuristic[num] = frequency.get(num, 0) / max_freq
        
        # Recency bias
        if past_draws:
            recent_numbers = set(past_draws[-1])
            for num in recent_numbers:
                if self.min_num <= num <= self.max_num:
                    heuristic[num] *= 1.5
        
        return dict(heuristic)
    
    def _combine_swarm_predictions(self, predictions: Dict[str, List[int]], 
                                  past_draws: List[List[int]]) -> List[int]:
        """Raj predikciók kombinálása."""
        
        # Algoritmus teljesítmények értékelése
        self._evaluate_algorithm_performance(predictions, past_draws)
        
        # Súlyozott voting
        number_votes = defaultdict(float)
        
        for algorithm, prediction in predictions.items():
            weight = self.algorithm_weights.get(algorithm, 0.33)
            
            for num in prediction:
                number_votes[num] += weight
        
        # Legjobb számok kiválasztása
        sorted_votes = sorted(number_votes.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_votes[:self.target_count * 2]]
        
        # Raj-specifikus diverzitás
        final_numbers = self._ensure_swarm_diversity(top_numbers)
        
        return final_numbers[:self.target_count]
    
    def _evaluate_algorithm_performance(self, predictions: Dict[str, List[int]], 
                                       past_draws: List[List[int]]):
        """Algoritmus teljesítmények értékelése és súlyok frissítése."""
        
        if len(past_draws) < 2:
            return
        
        test_draw = past_draws[-1]
        
        for algorithm, prediction in predictions.items():
            # Teljesítmény mérése
            matches = len(set(prediction) & set(test_draw))
            performance = matches / self.target_count
            
            self.performance_history[algorithm].append(performance)
            
            # Súlyok adaptív frissítése
            if len(self.performance_history[algorithm]) >= 5:
                recent_performance = np.mean(self.performance_history[algorithm][-5:])
                
                # Exponenciális súlyfrissítés
                improvement_factor = 1.0 + recent_performance
                self.algorithm_weights[algorithm] *= improvement_factor
        
        # Súlyok renormalizálása
        total_weight = sum(self.algorithm_weights.values())
        if total_weight > 0:
            self.algorithm_weights = {
                alg: weight / total_weight 
                for alg, weight in self.algorithm_weights.items()
            }
    
    def _ensure_swarm_diversity(self, numbers: List[int]) -> List[int]:
        """Raj-specifikus diverzitás biztosítása."""
        
        diverse_numbers = []
        
        for num in numbers:
            # Raj viselkedés: elkerüljük a túl közel lévő számokat
            if all(abs(num - existing) >= 3 for existing in diverse_numbers):
                diverse_numbers.append(num)
            
            if len(diverse_numbers) >= self.target_count:
                break
        
        # Kiegészítés raj heurisztikával
        while len(diverse_numbers) < self.target_count:
            remaining = [num for num in range(self.min_num, self.max_num + 1) 
                        if num not in diverse_numbers]
            if remaining:
                # Raj-alapú kiválasztás (preferálja a középső értékeket)
                center = (self.min_num + self.max_num) / 2
                remaining.sort(key=lambda x: abs(x - center))
                diverse_numbers.append(remaining[0])
            else:
                break
        
        return diverse_numbers
    
    def _generate_swarm_fallback(self) -> List[int]:
        """Raj-alapú fallback generálás."""
        
        # Egyszerű raj viselkedés szimuláció
        numbers = set()
        
        # "Méhek" kezdeti pozíciói
        bee_positions = [random.randint(self.min_num, self.max_num) for _ in range(5)]
        
        for _ in range(self.target_count):
            # Raj közeledés a legjobb pozíciókhoz
            if bee_positions:
                center = sum(bee_positions) / len(bee_positions)
                noise = random.randint(-5, 5)
                new_number = int(center + noise)
                new_number = max(self.min_num, min(new_number, self.max_num))
                
                if new_number not in numbers:
                    numbers.add(new_number)
                    bee_positions.append(new_number)  # Új "méh" pozíció
        
        # Kiegészítés ha szükséges
        while len(numbers) < self.target_count:
            num = random.randint(self.min_num, self.max_num)
            numbers.add(num)
        
        return sorted(list(numbers)[:self.target_count])


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Swarm Intelligence alapú lottószám predikció.
    Returns a tuple (main_numbers, additional_numbers).
    """
    try:
        # Fő számok generálása
        main_numbers = generate_swarm_intelligence_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.min_number),
            max_number=int(lottery_type_instance.max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            is_main=True
        )

        # Kiegészítő számok generálása
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_swarm_intelligence_numbers(
                lottery_type_instance,
                min_number=int(lottery_type_instance.additional_min_number),
                max_number=int(lottery_type_instance.additional_max_number),
                pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
                is_main=False
            )

        return sorted(main_numbers), sorted(additional_numbers)
    
    except Exception as e:
        logger.error(f"Hiba a swarm intelligence predikció során: {e}")
        return generate_fallback_numbers(lottery_type_instance)


def generate_swarm_intelligence_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    pieces_of_draw_numbers: int,
    is_main: bool
) -> List[int]:
    """Swarm intelligence számgenerálás."""
    
    # Történeti adatok lekérése
    past_draws = get_historical_data(lottery_type_instance, is_main)
    
    if len(past_draws) < 5:
        return generate_swarm_fallback(min_number, max_number, pieces_of_draw_numbers)
    
    # Swarm intelligence predictor
    predictor = SwarmIntelligencePredictor(
        min_num=min_number, 
        max_num=max_number, 
        target_count=pieces_of_draw_numbers
    )
    
    # Predikció
    predictions = predictor.predict(past_draws)
    
    return predictions


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Történeti adatok lekérése."""
    try:
        queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:100]
        
        past_draws = []
        for draw in queryset:
            try:
                if is_main:
                    numbers = draw.lottery_type_number
                else:
                    numbers = getattr(draw, 'additional_numbers', None)
                
                if isinstance(numbers, list) and len(numbers) > 0:
                    valid_numbers = [int(num) for num in numbers if isinstance(num, (int, float))]
                    if valid_numbers:
                        past_draws.append(valid_numbers)
            except (ValueError, TypeError, AttributeError):
                continue
        
        return past_draws
    
    except Exception as e:
        logger.error(f"Hiba a történeti adatok lekérésében: {e}")
        return []


def generate_swarm_fallback(min_number: int, max_number: int, count: int) -> List[int]:
    """Raj-alapú fallback számgenerálás."""
    
    # Egyszerű raj szimuláció
    center = (min_number + max_number) / 2
    swarm_positions = []
    
    # 5 "ágens" kezdeti pozíciója
    for _ in range(5):
        pos = random.uniform(min_number, max_number)
        swarm_positions.append(pos)
    
    # Raj mozgás szimulációja
    for _ in range(10):
        # Ágens pozíciók frissítése a raj középpontja felé
        avg_pos = sum(swarm_positions) / len(swarm_positions)
        
        for i in range(len(swarm_positions)):
            # Mozgás a raj középpontja felé + zaj
            direction = avg_pos - swarm_positions[i]
            noise = random.uniform(-2, 2)
            swarm_positions[i] += direction * 0.1 + noise
            
            # Határok ellenőrzése
            swarm_positions[i] = max(min_number, min(swarm_positions[i], max_number))
    
    # Pozíciók diszkretizálása
    discrete_positions = [int(round(pos)) for pos in swarm_positions]
    discrete_positions = list(set(discrete_positions))
    
    # Kiegészítés ha szükséges
    while len(discrete_positions) < count:
        num = random.randint(min_number, max_number)
        if num not in discrete_positions:
            discrete_positions.append(num)
    
    return sorted(discrete_positions[:count])


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Fallback számgenerálás."""
    main_numbers = generate_swarm_fallback(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers)
    )
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_swarm_fallback(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count)
        )
    
    return sorted(main_numbers), sorted(additional_numbers) 