# evolutionary_swarm_intelligence_prediction.py
"""
Evolúciós Raj Intelligencia Predikció
Kombinálja a genetikus algoritmusokat, particle swarm optimization-t és ant colony optimization-t
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import random
import math
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


class EvolutionarySwarmPredictor:
    """
    Evolúciós raj intelligencia predikció
    """
    
    def __init__(self):
        # Genetikus algoritmus paraméterek
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
        # Particle Swarm Optimization paraméterek
        self.swarm_size = 30
        self.iterations = 50
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        
        # Ant Colony Optimization paraméterek
        self.num_ants = 25
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic importance
        self.evaporation_rate = 0.1
        self.pheromone_deposit = 1.0
        
        # Ensemble súlyok
        self.ensemble_weights = {
            'genetic': 0.35,
            'pso': 0.35,
            'aco': 0.30
        }
    
    def get_numbers(self, lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
        """
        Főbejárási pont az evolúciós raj intelligencia predikcióhoz
        """
        try:
            # Fő számok generálása
            main_numbers = self._generate_swarm_numbers(
                lottery_type_instance,
                min_num=int(lottery_type_instance.min_number),
                max_num=int(lottery_type_instance.max_number),
                required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
                is_main=True
            )
            
            # Kiegészítő számok generálása
            additional_numbers = []
            if lottery_type_instance.has_additional_numbers:
                additional_numbers = self._generate_swarm_numbers(
                    lottery_type_instance,
                    min_num=int(lottery_type_instance.additional_min_number),
                    max_num=int(lottery_type_instance.additional_max_number),
                    required_numbers=int(lottery_type_instance.additional_numbers_count),
                    is_main=False
                )
            
            return sorted(main_numbers), sorted(additional_numbers)
            
        except Exception as e:
            print(f"Hiba az evolutionary_swarm_intelligence_prediction-ben: {e}")
            return self._generate_fallback_numbers(lottery_type_instance)
    
    def _generate_swarm_numbers(self, lottery_type_instance: lg_lottery_type,
                              min_num: int, max_num: int, required_numbers: int,
                              is_main: bool) -> List[int]:
        """
        Raj intelligencia alapú számgenerálás
        """
        # Történeti adatok lekérése
        historical_data = self._get_historical_data(lottery_type_instance, is_main)
        
        if len(historical_data) < 20:
            return self._generate_smart_random(min_num, max_num, required_numbers)
        
        # Fitness függvény előkészítése
        fitness_data = self._prepare_fitness_data(historical_data, min_num, max_num)
        
        # Három algoritmus futtatása párhuzamosan
        genetic_result = self._genetic_algorithm(fitness_data, min_num, max_num, required_numbers)
        pso_result = self._particle_swarm_optimization(fitness_data, min_num, max_num, required_numbers)
        aco_result = self._ant_colony_optimization(fitness_data, min_num, max_num, required_numbers)
        
        # Ensemble kombinálás
        final_numbers = self._ensemble_combination(
            genetic_result, pso_result, aco_result, min_num, max_num, required_numbers
        )
        
        return final_numbers
    
    def _prepare_fitness_data(self, historical_data: List[List[int]], 
                            min_num: int, max_num: int) -> Dict:
        """
        Fitness függvény adatok előkészítése
        """
        fitness_data = {
            'frequency': Counter(),
            'recency': {},
            'gaps': {},
            'trends': {},
            'correlations': {}
        }
        
        # Frekvencia számítás
        for draw in historical_data:
            for num in draw:
                if min_num <= num <= max_num:
                    fitness_data['frequency'][num] += 1
        
        # Recency számítás (utolsó megjelenés)
        for i, draw in enumerate(historical_data):
            for num in draw:
                if min_num <= num <= max_num and num not in fitness_data['recency']:
                    fitness_data['recency'][num] = i
        
        # Gap elemzés
        for num in range(min_num, max_num + 1):
            last_seen = fitness_data['recency'].get(num, len(historical_data))
            fitness_data['gaps'][num] = last_seen
        
        # Trend elemzés
        if len(historical_data) >= 20:
            recent_freq = Counter(num for draw in historical_data[:10] for num in draw)
            older_freq = Counter(num for draw in historical_data[10:20] for num in draw)
            
            for num in range(min_num, max_num + 1):
                recent_count = recent_freq.get(num, 0)
                older_count = older_freq.get(num, 0)
                fitness_data['trends'][num] = recent_count - older_count * 0.5
        
        return fitness_data
    
    def _genetic_algorithm(self, fitness_data: Dict, min_num: int, max_num: int, 
                         required_numbers: int) -> List[int]:
        """
        Genetikus algoritmus implementáció
        """
        # Kezdeti populáció generálása
        population = []
        for _ in range(self.population_size):
            individual = random.sample(range(min_num, max_num + 1), required_numbers)
            population.append(individual)
        
        # Evolúciós ciklus
        for generation in range(self.generations):
            # Fitness értékelés
            fitness_scores = [self._calculate_fitness(individual, fitness_data) for individual in population]
            
            # Szelekció (tournament selection)
            new_population = []
            
            # Elit egyedek megtartása
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx][:])
            
            # Új egyedek generálása
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Keresztezés
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, min_num, max_num, required_numbers)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                # Mutáció
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, min_num, max_num, required_numbers)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, min_num, max_num, required_numbers)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Legjobb egyed visszaadása
        final_fitness = [self._calculate_fitness(individual, fitness_data) for individual in population]
        best_index = max(range(len(final_fitness)), key=lambda i: final_fitness[i])
        
        return population[best_index]
    
    def _particle_swarm_optimization(self, fitness_data: Dict, min_num: int, max_num: int, 
                                   required_numbers: int) -> List[int]:
        """
        Particle Swarm Optimization implementáció
        """
        # Részecskék inicializálása
        particles = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        
        for _ in range(self.swarm_size):
            # Pozíció (számok)
            position = np.random.uniform(min_num, max_num, required_numbers)
            particles.append(position)
            
            # Sebesség
            velocity = np.random.uniform(-1, 1, required_numbers)
            velocities.append(velocity)
            
            # Személyes legjobb
            personal_best.append(position.copy())
            personal_best_fitness.append(self._calculate_continuous_fitness(position, fitness_data, min_num, max_num))
        
        # Globális legjobb
        global_best_index = max(range(len(personal_best_fitness)), key=lambda i: personal_best_fitness[i])
        global_best = personal_best[global_best_index].copy()
        
        # PSO iterációk
        for iteration in range(self.iterations):
            for i in range(self.swarm_size):
                # Sebesség frissítés
                r1, r2 = random.random(), random.random()
                
                velocities[i] = (self.w * velocities[i] + 
                               self.c1 * r1 * (personal_best[i] - particles[i]) +
                               self.c2 * r2 * (global_best - particles[i]))
                
                # Pozíció frissítés
                particles[i] += velocities[i]
                
                # Határok ellenőrzése
                particles[i] = np.clip(particles[i], min_num, max_num)
                
                # Fitness értékelés
                current_fitness = self._calculate_continuous_fitness(particles[i], fitness_data, min_num, max_num)
                
                # Személyes legjobb frissítés
                if current_fitness > personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = current_fitness
                    
                    # Globális legjobb frissítés
                    if current_fitness > personal_best_fitness[global_best_index]:
                        global_best = particles[i].copy()
                        global_best_index = i
        
        # Legjobb megoldás konvertálása egész számokra
        best_solution = [int(round(x)) for x in global_best]
        best_solution = list(set(best_solution))  # Duplikátumok eltávolítása
        
        # Kiegészítés szükség esetén
        if len(best_solution) < required_numbers:
            remaining = [num for num in range(min_num, max_num + 1) if num not in best_solution]
            random.shuffle(remaining)
            best_solution.extend(remaining[:required_numbers - len(best_solution)])
        
        return best_solution[:required_numbers]
    
    def _ant_colony_optimization(self, fitness_data: Dict, min_num: int, max_num: int, 
                               required_numbers: int) -> List[int]:
        """
        Ant Colony Optimization implementáció
        """
        # Feromon mátrix inicializálása
        num_range = max_num - min_num + 1
        pheromone = np.ones((num_range, num_range)) * 0.1
        
        # Heurisztikus információ
        heuristic = np.zeros((num_range, num_range))
        for i in range(num_range):
            for j in range(num_range):
                num1, num2 = i + min_num, j + min_num
                # Heurisztikus érték a fitness adatok alapján
                heuristic[i][j] = (fitness_data['frequency'].get(num1, 0) + 
                                 fitness_data['frequency'].get(num2, 0)) / 2
        
        best_solution = None
        best_fitness = -float('inf')
        
        # ACO iterációk
        for iteration in range(self.iterations):
            solutions = []
            
            # Hangyák útjainak generálása
            for ant in range(self.num_ants):
                solution = self._construct_ant_solution(pheromone, heuristic, min_num, max_num, required_numbers)
                solutions.append(solution)
                
                # Fitness értékelés
                fitness = self._calculate_fitness(solution, fitness_data)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution[:]
            
            # Feromon frissítés
            pheromone *= (1 - self.evaporation_rate)  # Elpárolgás
            
            # Feromon lerakás
            for solution in solutions:
                fitness = self._calculate_fitness(solution, fitness_data)
                deposit = self.pheromone_deposit * fitness
                
                for i in range(len(solution) - 1):
                    idx1, idx2 = solution[i] - min_num, solution[i + 1] - min_num
                    pheromone[idx1][idx2] += deposit
                    pheromone[idx2][idx1] += deposit  # Szimmetrikus
        
        return best_solution if best_solution else self._generate_smart_random(min_num, max_num, required_numbers)
    
    def _construct_ant_solution(self, pheromone: np.ndarray, heuristic: np.ndarray,
                              min_num: int, max_num: int, required_numbers: int) -> List[int]:
        """
        Hangya megoldás konstruálása
        """
        solution = []
        available = list(range(min_num, max_num + 1))
        
        # Első szám véletlen választása
        current = random.choice(available)
        solution.append(current)
        available.remove(current)
        
        # Többi szám választása feromon és heurisztikus információ alapján
        for _ in range(required_numbers - 1):
            if not available:
                break
            
            probabilities = []
            current_idx = current - min_num
            
            for num in available:
                num_idx = num - min_num
                if num_idx < len(pheromone) and current_idx < len(pheromone):
                    prob = (pheromone[current_idx][num_idx] ** self.alpha) * (heuristic[current_idx][num_idx] ** self.beta)
                    probabilities.append(prob)
                else:
                    probabilities.append(0.1)  # Alapértelmezett valószínűség
            
            # Normalizálás
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
            else:
                probabilities = [1.0 / len(available)] * len(available)
            
            # Választás
            current = np.random.choice(available, p=probabilities)
            solution.append(current)
            available.remove(current)
        
        return solution
    
    def _calculate_fitness(self, individual: List[int], fitness_data: Dict) -> float:
        """
        Fitness függvény számítás
        """
        fitness = 0.0
        
        # Frekvencia alapú fitness
        for num in individual:
            fitness += fitness_data['frequency'].get(num, 0)
        
        # Recency alapú fitness
        for num in individual:
            recency = fitness_data['recency'].get(num, len(fitness_data['recency']))
            fitness += 1.0 / (recency + 1)  # Újabb számok magasabb fitness
        
        # Gap alapú fitness
        for num in individual:
            gap = fitness_data['gaps'].get(num, 0)
            fitness += gap * 0.1  # Régen nem látott számok magasabb fitness
        
        # Trend alapú fitness
        for num in individual:
            trend = fitness_data['trends'].get(num, 0)
            fitness += trend * 0.5
        
        # Diverzitás bónusz
        if len(set(individual)) == len(individual):  # Nincs duplikátum
            fitness += 10.0
        
        # Összeg alapú fitness (optimális tartomány)
        total_sum = sum(individual)
        optimal_sum = len(individual) * (max(individual) + min(individual)) / 2
        sum_penalty = abs(total_sum - optimal_sum) / optimal_sum
        fitness -= sum_penalty * 5.0
        
        return fitness
    
    def _calculate_continuous_fitness(self, position: np.ndarray, fitness_data: Dict, 
                                    min_num: int, max_num: int) -> float:
        """
        Folytonos fitness függvény PSO-hoz
        """
        # Egész számokra kerekítés
        rounded_position = [int(round(x)) for x in position]
        rounded_position = [max(min_num, min(max_num, x)) for x in rounded_position]
        
        return self._calculate_fitness(rounded_position, fitness_data)
    
    def _tournament_selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """
        Tournament szelekció
        """
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        winner_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[winner_index][:]
    
    def _crossover(self, parent1: List[int], parent2: List[int], 
                  min_num: int, max_num: int, required_numbers: int) -> Tuple[List[int], List[int]]:
        """
        Keresztezés operátor
        """
        # Order crossover (OX)
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1 = [None] * size
        child2 = [None] * size
        
        # Középső rész másolása
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # Hiányzó elemek kitöltése
        self._fill_child(child1, parent2, start, end, min_num, max_num)
        self._fill_child(child2, parent1, start, end, min_num, max_num)
        
        return child1, child2
    
    def _fill_child(self, child: List[int], parent: List[int], start: int, end: int,
                   min_num: int, max_num: int) -> None:
        """
        Gyermek kitöltése keresztezés után
        """
        parent_cycle = parent[end:] + parent[:end]
        child_idx = end
        
        for num in parent_cycle:
            if num not in child:
                child[child_idx % len(child)] = num
                child_idx += 1
        
        # Hiányzó pozíciók kitöltése
        for i in range(len(child)):
            if child[i] is None:
                available = [num for num in range(min_num, max_num + 1) if num not in child]
                if available:
                    child[i] = random.choice(available)
    
    def _mutate(self, individual: List[int], min_num: int, max_num: int, 
               required_numbers: int) -> List[int]:
        """
        Mutáció operátor
        """
        mutated = individual[:]
        
        # Swap mutáció
        if len(mutated) >= 2:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        
        # Véletlen szám cseréje
        if random.random() < 0.3:
            idx = random.randint(0, len(mutated) - 1)
            available = [num for num in range(min_num, max_num + 1) if num not in mutated]
            if available:
                mutated[idx] = random.choice(available)
        
        return mutated
    
    def _ensemble_combination(self, genetic_result: List[int], pso_result: List[int], 
                            aco_result: List[int], min_num: int, max_num: int, 
                            required_numbers: int) -> List[int]:
        """
        Ensemble kombinálás
        """
        vote_counter = Counter()
        
        # Súlyozott szavazás
        results = [
            (genetic_result, self.ensemble_weights['genetic']),
            (pso_result, self.ensemble_weights['pso']),
            (aco_result, self.ensemble_weights['aco'])
        ]
        
        for result, weight in results:
            for i, num in enumerate(result):
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
evolutionary_swarm_predictor = EvolutionarySwarmPredictor()

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Főbejárási pont az evolúciós raj intelligencia predikcióhoz
    """
    return evolutionary_swarm_predictor.get_numbers(lottery_type_instance)
