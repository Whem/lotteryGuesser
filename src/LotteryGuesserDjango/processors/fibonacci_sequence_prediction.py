import random


def get_numbers(lottery_type_instance):
    def is_near_fibonacci(number):
        a, b = 0, 1
        while b < number:
            a, b = b, a + b
        return b == number or (number - a) <= 2 or (b - number) <= 2

    predicted_numbers = set()
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        number = random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number)
        if is_near_fibonacci(number):
            predicted_numbers.add(number)

    return sorted(predicted_numbers)
