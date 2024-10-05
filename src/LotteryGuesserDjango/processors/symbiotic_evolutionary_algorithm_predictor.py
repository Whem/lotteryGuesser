# symbiotic_evolutionary_algorithm_predictor.py

import random
import math
from collections import defaultdict
from algorithms.models import lg_lottery_winner_number

class LotteryOrganism:
    def __init__(self, genome, min_num, max_num):
        self.genome = genome
        self.fitness = 0
        self.min_num = min_num
        self.max_num = max_num

    def mutate(self, mutation_rate):
        return [
            random.randint(self.min_num, self.max_num) if random.random() < mutation_rate else gene
            for gene in self.genome
        ]

def create_initial_population(pop_size, genome_size, min_num, max_num):
    return [
        LotteryOrganism([random.randint(min_num, max_num) for _ in range(genome_size)], min_num, max_num)
        for _ in range(pop_size)
    ]

def fitness_function(organism, past_draws):
    fitness = 0
    for draw in past_draws:
        matches = len(set(organism.genome) & set(draw))
        fitness += matches ** 2  # Quadratic reward for matches
    return fitness

def symbiotic_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1.genome) - 1)
    child1_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
    child2_genome = parent2.genome[:crossover_point] + parent1.genome[crossover_point:]
    return (
        LotteryOrganism(child1_genome, parent1.min_num, parent1.max_num),
        LotteryOrganism(child2_genome, parent1.min_num, parent1.max_num)
    )

def evolve_population(population, past_draws, generations=50, mutation_rate=0.1):
    for _ in range(generations):
        # Evaluate fitness
        for organism in population:
            organism.fitness = fitness_function(organism, past_draws)

        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Select top performers
        elite_size = max(2, int(len(population) * 0.1))
        new_population = population[:elite_size]

        # Symbiotic crossover and mutation
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(population[:len(population)//2], 2)
            child1, child2 = symbiotic_crossover(parent1, parent2)
            child1.genome = child1.mutate(mutation_rate)
            child2.genome = child2.mutate(mutation_rate)
            new_population.extend([child1, child2])

        population = new_population[:len(population)]

    return population

def get_numbers(lottery_type_instance):
    try:
        min_num = int(lottery_type_instance.min_number)
        max_num = int(lottery_type_instance.max_number)
        total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

        # Retrieve past winning numbers
        past_draws = list(lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:100].values_list('lottery_type_number', flat=True))

        if len(past_draws) < 10:
            # If not enough past draws, return random numbers
            return random.sample(range(min_num, max_num + 1), total_numbers)

        # Create initial population
        population_size = 100
        population = create_initial_population(population_size, total_numbers, min_num, max_num)

        # Evolve population
        evolved_population = evolve_population(population, past_draws)

        # Select the best organism
        best_organism = max(evolved_population, key=lambda x: x.fitness)

        # Ensure uniqueness and correct range
        predicted_numbers = list(set(best_organism.genome))
        predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]

        # If not enough unique numbers, fill with random selection
        if len(predicted_numbers) < total_numbers:
            remaining = set(range(min_num, max_num + 1)) - set(predicted_numbers)
            predicted_numbers += random.sample(list(remaining), total_numbers - len(predicted_numbers))

        return sorted(predicted_numbers[:total_numbers])

    except Exception as e:
        # Log the error (you might want to use a proper logging system)
        print(f"Error in symbiotic_evolutionary_algorithm_predictor: {str(e)}")
        # Fall back to random number generation
        return random.sample(range(min_num, max_num + 1), total_numbers)