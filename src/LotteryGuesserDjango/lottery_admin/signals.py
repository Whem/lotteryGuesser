import statistics

import requests
from bs4 import BeautifulSoup

from datetime import datetime

from algorithms.models import lg_lottery_winner_number


def download_numbers_from_internet(lottery_item):
    lottery_collection = []
    response = requests.get(lottery_item.url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')

    table_rows = soup.find("table").find_all("tr")[1:]  # Skip header row
    table_rows.reverse()  # Reverse the order of rows as done in the C# code

    for row in table_rows:
        columns = row.find_all("td")
        if len(columns) > lottery_item.skip_items:
            year = int(columns[0].text.strip())
            week = int(columns[1].text.strip())
            try:
                numbers = [int(columns[i].text) for i in range(lottery_item.skip_items, len(columns))[:lottery_item.pieces_of_draw_numbers]]
                additional_numbers = [int(columns[i].text) for i in range(lottery_item.skip_items, len(columns))[lottery_item.pieces_of_draw_numbers:]]
            except Exception as e:
                print(f"Error converting numbers to integers: {e}")


            # Calculating statistics
            sum_of_numbers = sum(numbers)
            average_of_numbers = sum_of_numbers / len(numbers)
            median_of_numbers = statistics.median(numbers)
            mode_of_numbers = statistics.mode(numbers)
            standard_deviation_of_numbers = statistics.stdev(numbers)


            existing_winner_number = lg_lottery_winner_number.objects.filter(lottery_type=lottery_item, lottery_type_number_year=year, lottery_type_number_week=week).first()

            if existing_winner_number is not None:
                continue

            # Create an instance of lg_lottery_winner_number
            winner_number = lg_lottery_winner_number(
                lottery_type=lottery_item,
                lottery_type_number=numbers,
                lottery_type_number_year=year,
                lottery_type_number_week=week,
                sum=sum_of_numbers,
                average=average_of_numbers,
                median=median_of_numbers,
                mode=mode_of_numbers,
                standard_deviation=standard_deviation_of_numbers,
                additional_numbers=additional_numbers if lottery_item.has_additional_numbers else None
            )
            winner_number.save()
            lottery_collection.append(winner_number)


    return lottery_collection