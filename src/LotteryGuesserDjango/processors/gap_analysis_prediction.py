#gap_analysis_prediction.py
import random
from typing import List, Tuple, Dict
from collections import Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Gap analysis based genetic algorithm predictor for combined lottery types.
    Returns (main_numbers, additional_numbers).
    """

    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True,
        lottery_type_instance
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            False,
            lottery_type_instance
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool,
        lottery_static_instance: lg_lottery_type
) -> List[int]:
    """Generate numbers using genetic algorithm with gap analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if not past_draws:
        return random_number_set(min_num, max_num, required_numbers)

    # Calculate fitness metrics
    number_fitness = calculate_fitness_metrics(
        past_draws,
        min_num,
        max_num
    )

    # Genetic algorithm parameters
    params = {
        'population_size': 100,
        'generations': 50,
        'mutation_rate': 0.1
    }

    # Run genetic algorithm
    best_solution = run_genetic_algorithm(
        number_fitness,
        min_num,
        max_num,
        required_numbers,
        params,
        lottery_static_instance
    )

    return sorted(best_solution)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id'))

    if is_main:
        draws = [draw.lottery_type_number for draw in past_draws
                 if isinstance(draw.lottery_type_number, list)]
    else:
        draws = [draw.additional_numbers for draw in past_draws
                 if hasattr(draw, 'additional_numbers') and
                 isinstance(draw.additional_numbers, list)]

    return draws


def calculate_fitness_metrics(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> Dict[int, float]:
    """Calculate fitness metrics for each number."""
    # Flatten past draws
    all_numbers = [num for draw in past_draws for num in draw]

    # Calculate frequency
    frequency = Counter(all_numbers)

    # Calculate fitness scores
    fitness = {
        num: frequency.get(num, 0)
        for num in range(min_num, max_num + 1)
    }

    return fitness


def run_genetic_algorithm(
        fitness_metrics: Dict[int, float],
        min_num: int,
        max_num: int,
        required_numbers: int,
        params: Dict,
        lottery_static_instance: lg_lottery_type
) -> List[int]:
    """Run genetic algorithm to find optimal number set."""
    # Initialize population
    population = initialize_population(
        params['population_size'],
        min_num,
        max_num,
        required_numbers
    )

    # Run generations
    for _ in range(params['generations']):
        # Calculate fitness scores
        fitness_scores = [
            calculate_individual_fitness(individual, fitness_metrics)
            for individual in population
        ]

        # Select parents
        parents = select_parents(population, fitness_scores)

        # Generate offspring
        offspring = crossover(
            parents,
            params['population_size'],
            lottery_static_instance
        )

        # Apply mutation
        population = mutate(
            offspring,
            params['mutation_rate'],
            lottery_static_instance
        )

    # Select best solution
    final_fitness_scores = [
        calculate_individual_fitness(individual, fitness_metrics)
        for individual in population
    ]
    best_solution = max(
        zip(population, final_fitness_scores),
        key=lambda x: x[1]
    )[0]

    return best_solution


def initialize_population(
        size: int,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[List[int]]:
    """Initialize random population."""
    number_range = range(min_num, max_num + 1)
    population = [
        sorted(random.sample(number_range, required_numbers))
        for _ in range(size)
    ]
    return population


def calculate_individual_fitness(
        individual: List[int],
        fitness_metrics: Dict[int, float]
) -> float:
    """Calculate fitness for an individual."""
    return sum(fitness_metrics.get(num, 0) for num in individual)


def select_parents(
        population: List[List[int]],
        fitness_scores: List[float]
) -> List[List[int]]:
    """Select parents using roulette wheel selection."""
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        probabilities = [1 / len(population)] * len(population)
    else:
        probabilities = [score / total_fitness for score in fitness_scores]

    return random.choices(
        population,
        weights=probabilities,
        k=len(population)
    )

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
