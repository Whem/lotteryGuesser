# genetic_algorithm_evolutionary_optimization.py
import random
from typing import List, Tuple
import numpy as np
from deap import base, creator, tools, algorithms
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Genetic algorithm predictor for combined lottery types.
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
    """Generate numbers using genetic algorithm optimization."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if not past_draws:
        return random_number_set(min_num, max_num, required_numbers)

    # Initialize DEAP framework
    toolbox = initialize_genetic_algorithm(
        past_draws,
        min_num,
        max_num,
        required_numbers
    )

    # Run optimization
    best_solution = run_evolutionary_optimization(toolbox)

    return sorted(set(best_solution))


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id'))

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def initialize_genetic_algorithm(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> base.Toolbox:
    """Initialize DEAP genetic algorithm framework."""
    # Create fitness and individual types if they don't exist
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # Initialize toolbox
    toolbox = base.Toolbox()

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

    # Define evaluation function
    def evaluate(individual: List[int]) -> Tuple[float]:
        score = sum(
            1 for draw in past_draws
            if len(set(individual) & set(draw)) >= 3
        )
        return (score,)

    # Register genetic algorithm operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=min_num,
        up=max_num,
        indpb=0.1
    )
    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=3
    )

    return toolbox


def run_evolutionary_optimization(toolbox: base.Toolbox) -> List[int]:
    """Run evolutionary optimization algorithm."""
    try:
        # Create initial population
        population = toolbox.population(n=100)

        # Run evolution
        algorithms.eaSimple(
            population,
            toolbox,
            cxpb=0.5,  # Crossover probability
            mutpb=0.2,  # Mutation probability
            ngen=50,  # Number of generations
            verbose=False
        )

        # Select best solution
        best_individual = tools.selBest(population, k=1)[0]

        return best_individual

    except Exception as e:
        print(f"Error in evolutionary optimization: {str(e)}")
        return [
            toolbox.attr_int()
            for _ in range(len(toolbox.individual()))
        ]


def random_number_set(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate a random set of numbers."""
    return sorted(random.sample(range(min_num, max_num + 1), required_numbers))