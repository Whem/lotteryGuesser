import random
from typing import List
from collections import Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    """
    Generates lottery numbers using an optimized genetic algorithm.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    # Retrieve past draws
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    past_draws = [
        draw for draw in past_draws_queryset if isinstance(draw, list)
    ]

    if not past_draws:
        # If no past draws, return random numbers
        return sorted(random.sample(
            range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
            lottery_type_instance.pieces_of_draw_numbers
        ))

    # Flatten past draws to get all numbers
    all_past_numbers = [num for draw in past_draws for num in draw]

    # Precompute the frequency of each number in past draws
    number_counter = Counter(all_past_numbers)

    # Precompute the fitness value for each number
    number_fitness = {
        num: number_counter.get(num, 0)
        for num in range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1)
    }

    # Genetic Algorithm Parameters
    population_size = 100  # Increased population size for better diversity
    generations = 50       # Increased generations for better convergence
    mutation_rate = 0.1    # Adjusted mutation rate

    # Initialize population
    population = initialize_population(population_size, lottery_type_instance)

    for _ in range(generations):
        # Calculate fitness scores
        fitness_scores = [fitness(individual, number_fitness) for individual in population]

        # Select parents based on fitness scores
        parents = select_parents(population, fitness_scores)

        # Generate offspring through crossover
        offspring = crossover(parents, population_size, lottery_type_instance)

        # Apply mutation to offspring
        population = mutate(offspring, mutation_rate, lottery_type_instance)

    # Final fitness evaluation
    fitness_scores = [fitness(individual, number_fitness) for individual in population]
    best_individual = max(zip(population, fitness_scores), key=lambda x: x[1])[0]

    return sorted(best_individual)


def initialize_population(size: int, lottery_type_instance: lg_lottery_type) -> List[List[int]]:
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    num_pieces = lottery_type_instance.pieces_of_draw_numbers

    number_range = range(min_num, max_num + 1)
    population = [
        sorted(random.sample(number_range, num_pieces))
        for _ in range(size)
    ]
    return population


def fitness(individual: List[int], number_fitness: dict) -> float:
    # Fitness is the sum of precomputed fitness values of the individual's numbers
    return sum(number_fitness.get(num, 0) for num in individual)


def select_parents(population: List[List[int]], fitness_scores: List[float]) -> List[List[int]]:
    # Use roulette wheel selection based on fitness scores
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        # If all fitness scores are zero, select parents randomly
        selection_probabilities = [1 / len(population)] * len(population)
    else:
        selection_probabilities = [score / total_fitness for score in fitness_scores]

    parents = random.choices(population, weights=selection_probabilities, k=len(population))
    return parents


def crossover(parents: List[List[int]], offspring_size: int, lottery_type_instance: lg_lottery_type) -> List[List[int]]:
    offspring = []
    num_pieces = lottery_type_instance.pieces_of_draw_numbers
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    number_range = set(range(min_num, max_num + 1))

    for _ in range(offspring_size):
        parent1, parent2 = random.sample(parents, 2)

        # Single-point crossover
        crossover_point = random.randint(1, num_pieces - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]

        # Ensure uniqueness
        child_set = set(child)
        if len(child_set) < num_pieces:
            # Fill missing numbers randomly from the remaining numbers
            remaining_numbers = list(number_range - child_set)
            random.shuffle(remaining_numbers)
            child = list(child_set) + remaining_numbers[:num_pieces - len(child_set)]
        else:
            child = list(child_set)

        # Sort the child for consistency
        child.sort()
        offspring.append(child)
    return offspring


def mutate(population: List[List[int]], rate: float, lottery_type_instance: lg_lottery_type) -> List[List[int]]:
    num_pieces = lottery_type_instance.pieces_of_draw_numbers
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    number_range = set(range(min_num, max_num + 1))

    for individual in population:
        if random.random() < rate:
            # Choose a random number to replace
            index = random.randint(0, num_pieces - 1)
            current_num = individual[index]
            # Choose a new number not already in the individual
            available_numbers = list(number_range - set(individual))
            if available_numbers:
                new_number = random.choice(available_numbers)
                individual[index] = new_number
                individual.sort()
    return population
