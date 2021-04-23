#!/usr/bin/python3

from pathfinding import WALL_MARK

from collections import namedtuple
import numpy
import csv


Position = namedtuple("Position", ["row", "col"])


def is_inside(position, maze):
    rows, cols = maze.shape
    return 0 <= position.row < rows and 0 <= position.col < cols


def load_maze(filename):
    data = []
    with open(filename, newline="") as csvfile:
        content = csv.reader(csvfile, delimiter=",")
        for row in content:
            row_data = list(map(lambda x: WALL_MARK if x == "Z" else int(x), row))
            data.append(row_data)

    rows = len(data)
    assert rows > 1
    cols = len(data[0])
    assert cols > 1

    matrix = numpy.array(data, dtype=int)
    print(f" >> loaded {filename}", matrix.shape)

    return matrix


def save_maze(maze, filename):
    with open(filename, "wt", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in maze:
            writer.writerow(row)

    print(f" >> wrote {filename}", maze.shape)
