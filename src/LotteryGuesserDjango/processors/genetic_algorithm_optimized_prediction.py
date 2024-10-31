# genetic_algorithm_optimized_prediction.py
import random
from typing import List, Tuple
from deap import base, creator, tools, algorithms
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
from collections import Counter


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Optimized genetic algorithm predictor for combined lottery types.
    Returns (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate numbers using optimized genetic algorithm."""
    # Get historical data for fitness evaluation
    past_draws = get_historical_data(lottery_type_instance, is_main)

    # Initialize GA toolkit
    toolbox = setup_genetic_algorithm(
        past_draws,
        min_num,
        max_num,
        required_numbers
    )

    # Run optimization
    best_solution = run_genetic_algorithm(toolbox)

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


def setup_genetic_algorithm(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> base.Toolbox:
    """Setup DEAP genetic algorithm with optimized parameters."""
    # Create fitness and individual types
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # Initialize toolbox
    toolbox = base.Toolbox()

    # Calculate number frequencies for evaluation
    number_freq = calculate_frequencies(past_draws)

    # Define evaluation function with historical analysis
    def evaluate(individual: List[int]) -> Tuple[float]:
        return calculate_fitness(
            individual,
            past_draws,
            number_freq
        )

    # Register genetic operators
    toolbox.register(
        "attr_int",
        random.randint,
        min_num,
        max_num
    )
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_int,
        n=required_numbers
    )
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual
    )
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=min_num,
        up=max_num,
        indpb=0.05
    )
    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=3
    )

    return toolbox


def calculate_frequencies(past_draws: List[List[int]]) -> Counter:
    """Calculate number frequencies from historical data."""
    all_numbers = [num for draw in past_draws for num in draw]
    return Counter(all_numbers)


def calculate_fitness(
        individual: List[int],
        past_draws: List[List[int]],
        number_freq: Counter
) -> Tuple[float]:
    """Calculate fitness score for an individual."""
    if not past_draws:
        return (random.uniform(0, 1),)

    # Frequency score
    freq_score = sum(number_freq.get(num, 0) for num in individual)

    # Pattern matching score
    pattern_score = sum(
        len(set(individual) & set(draw))
        for draw in past_draws[-10:]  # Consider recent draws
    )

    # Distribution score
    avg = sum(individual) / len(individual)
    std_dev = (sum((x - avg) ** 2 for x in individual) / len(individual)) ** 0.5
    distribution_score = 1 / (1 + abs(std_dev - 15))  # Prefer reasonable spread

    # Combine scores
    total_score = (
            0.4 * freq_score +
            0.4 * pattern_score +
            0.2 * distribution_score
    )

    return (total_score,)


def run_genetic_algorithm(toolbox: base.Toolbox) -> List[int]:
    """Run genetic algorithm optimization."""
    try:
        # Create population
        population = toolbox.population(n=50)

        # Run evolution
        algorithms.eaSimple(
            population,
            toolbox,
            cxpb=0.5,  # Crossover probability
            mutpb=0.2,  # Mutation probability
            ngen=40,  # Number of generations
            verbose=False
        )

        # Get best solution
        best_individual = tools.selBest(population, k=1)[0]

        return best_individual

    except Exception as e:
        print(f"Error in genetic algorithm: {str(e)}")
        return [
            toolbox.attr_int()
            for _ in range(len(toolbox.individual()))
        ]