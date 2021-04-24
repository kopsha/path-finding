from datetime import datetime
import timeit
import random


random.seed()
data = random.choices(range(10), k=1000)
acceptable_values = set([0, 5, 9])


def print_stage(text, row_size=80):
    filler = " " * (row_size - 4 - len(text))
    print(f'{"*" * row_size}')
    print(f"* {text}{filler} *")
    print(f'{"*" * row_size}')


def test_acceptable_sets():
    count = 0
    for val in data:
        if val in acceptable_values:
            count += 1


def test_acceptable_ifs():
    count = 0
    for val in data:
        if val == 0 or val == 5 or val == 9:
            count += 1


def main():
    print_stage("sets vs comparisons")
    n = 100_000
    sets_duration = timeit.timeit(test_acceptable_sets, number=n)
    ifs_duration = timeit.timeit(test_acceptable_ifs, number=n)

    print("   sets took", sets_duration)
    print("and ifs took", ifs_duration)


if __name__ == "__main__":
    duration = timeit.timeit(main, number=1)
    now = datetime.now().strftime("%H:%M:%S")
    print_stage(f"[{now}] Finished in {duration:.2f} seconds.")
