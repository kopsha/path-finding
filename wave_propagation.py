from copy import deepcopy
from datetime import datetime
from collections import defaultdict, deque

import csv
import timeit
import numpy
import matplotlib.pyplot as plt


fig_counter = 0
WALL_MARK = -1
TEMP_WALL_MARK = -2


def print_stage(text, row_size=80):
    filler = " " * (row_size - 4 - len(text))
    print(f'{"*" * row_size}')
    print(f"* {text}{filler} *")
    print(f'{"*" * row_size}')


def load_csv_matrix(filename):
    print(f"loading {filename}")
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

    m = numpy.matrix(data, dtype=int)
    print(" >> shape", m.shape)

    return m


def save_figure(matrix, name_tag="fig"):
    global fig_counter
    fig_counter += 1

    fig, ax0 = plt.subplots(1)
    ax0.matshow(matrix)
    fig.savefig(f"fig_{fig_counter:>03}_{name_tag}.png")
    plt.close()

    # pprint(matrix)

    return


def scan_target_nodes(matrix):
    nodes = defaultdict(list)

    mit = numpy.nditer(matrix, flags=["multi_index"])
    for value in mit:
        if value > 0:
            nodes[int(value)].append(mit.multi_index)

    return dict(nodes)


def boxed_neighbours(matrix, node):
    r, c = node
    rows, cols = matrix.shape
    nb = {
        matrix[i, j]
        for i, j in [
            # top row, left to right
            (r - 1, c - 1),
            (r - 1, c),
            (r - 1, c + 1),
            (r, c + 1),  # right node
            # bottom row, right to left
            (r + 1, c + 1),
            (r + 1, c),
            (r + 1, c - 1),
            (r, c - 1),  # left node
        ]
        if (0 <= i < rows) and (0 <= j < cols)
    }
    return nb


def hide_other_targets(matrix, target):
    hidden = matrix.copy()

    mit = numpy.nditer(hidden, flags=["multi_index"], op_flags=["writeonly"])
    for value in mit:
        if value == 0:
            nb = boxed_neighbours(matrix, mit.multi_index) - {0, WALL_MARK, target}
            if nb:
                hidden[mit.multi_index] = WALL_MARK
        elif value > 0 and value != target:
            hidden[mit.multi_index] = WALL_MARK

    return hidden


def crossed_neighbours(matrix, node, target):
    rows, cols = matrix.shape
    r, c = node
    neighbours = [
        (i, j)
        for i, j in [
            (r - 1, c),
            (r, c - 1),
            (r + 1, c),
            (r, c + 1),
        ]
        if (0 <= i < rows) and (0 <= j < cols) and matrix[i, j] in {0}
    ]

    # print("neighbours of", node, "are", neighbours)

    return neighbours


def wave_from(matrix, start):
    visited = matrix.copy()
    target = matrix[start]
    discover = list([start])
    distance = 1

    while discover:
        explore = set()
        for node in discover:
            # print("visited", node)
            if matrix[node] == target:
                print(" >> target", node, "reached in", distance)
                explore.update(crossed_neighbours(visited, node, target))
            elif matrix[node] == 0:
                visited[node] = -10 - distance
                explore.update(crossed_neighbours(visited, node, target))
        # print("visited", node, "at distance", distance)
        # print("remaining", explore)
        distance += 1
        discover = explore

    return visited


class EdgeMeterDrone:
    def __init__(self, matrix, start, direction, steps=0):
        assert len(direction) == 2

        self.matrix = matrix
        self.pos = start
        self.direction = direction
        self.steps = steps

    def is_on_edge(self):
        r, c = self.pos
        rows, cols = self.matrix.shape
        return r in {0, rows - 1} or c in {0, cols - 1}

    def is_direction_valid(self):
        new_pos = tuple(numpy.add(self.pos, self.direction))
        r, c = new_pos
        rows, cols = self.matrix.shape

        # valid means I'm still on the edge
        if r in {0, rows - 1} and 0 <= c < cols:
            return True

        if c in {0, cols - 1} and 0 <= r < rows:
            return True

        return False

    def _step_direction(self):
        self.steps += 1
        new_pos = tuple(numpy.add(self.pos, self.direction))

        # print("from", self.pos, "D", self.direction, "going to", new_pos, "dist")
        value = self.matrix.item(new_pos)
        # print("new value", value, "my steps", self.steps)

        if value < 0:
            print("died. hit a block")
            return False

        if 0 < value <= self.steps:
            print("died. my path is longer")
            return False  # just died

        self.pos = new_pos
        self.matrix.itemset(new_pos, self.steps)
        return True

    def walk_step(self):
        clone = None

        if self.is_on_edge():
            if not self.is_direction_valid():
                directions = deque()
                save_direction = self.direction
                if self.direction == (-1, 0) or self.direction == (1, 0):  # up or down
                    self.direction = (0, -1)  # try left
                    if self.is_direction_valid():
                        directions.append(self.direction)
                    self.direction = (0, 1)  # try right
                    if self.is_direction_valid():
                        directions.append(self.direction)
                else:
                    self.direction = (-1, 0)  # try up
                    if self.is_direction_valid():
                        directions.append(self.direction)
                    self.direction = (1, 0)  # try down
                    if self.is_direction_valid():
                        directions.append(self.direction)

                self.direction = save_direction

                if len(directions) == 0:
                    print("died, no valid direction")
                    return False, None

                self.direction = directions.pop()
                print("changing direction to", self.direction)
                if directions:
                    clone = EdgeMeterDrone(
                        self.matrix, self.pos, directions.pop(), self.steps
                    )

        return self._step_direction(), clone


def edge_first_route(matrix, finish):
    routed = matrix.copy()

    rows, cols = matrix.shape
    r, c = finish
    drones = []

    drones.append(EdgeMeterDrone(routed, finish, (0, -1)))
    drones.append(EdgeMeterDrone(routed, finish, (0, 1)))
    drones.append(EdgeMeterDrone(routed, finish, (-1, 0)))
    drones.append(EdgeMeterDrone(routed, finish, (1, 0)))

    cnt = 0
    while drones and cnt < 200:
        new_drones = []
        just_died = set()
        for i, d in enumerate(drones):
            alive, clone = d.walk_step()
            if not alive:
                just_died.add(i)

            if clone:
                print("new drone was spanwed", clone.pos, clone.direction)
                new_drones.append(clone)

        drones = [d for i, d in enumerate(drones) if i not in just_died]

        drones.extend(new_drones)
        cnt += 1

    return routed


def main():

    data_files = [
        "step_one.csv",
        "step_two.csv",
    ]

    for filename in data_files:
        title = filename.rstrip(".csv")
        matrix = load_csv_matrix(filename)
        pins = scan_target_nodes(matrix)

        for pick in pins:
            clean_matrix = hide_other_targets(matrix, pick)
            acc_mat = deepcopy(clean_matrix)
            for node in pins[pick]:
                acc_mat = edge_first_route(acc_mat, node)

            save_figure(acc_mat, f"{title}_pin_{pick}")


if __name__ == "__main__":
    duration = timeit.timeit(main, number=1)
    now = datetime.now().strftime("%H:%M:%S")
    print_stage(f"[{now}] Finished in {duration:.2f} seconds.")
