from amaze import Position, is_inside, WALL_MARK, PATH_MARK

from heapq import heappop, heappush
from collections import deque, defaultdict
from itertools import combinations

import math
import numpy


def box_neighbour_values(maze, node):
    rows, cols = maze.shape
    nb = [
        maze.item(pos)
        for pos in [
            # top row, left to right
            Position(node.row - 1, node.col - 1),
            Position(node.row - 1, node.col),
            Position(node.row - 1, node.col + 1),
            Position(node.row, node.col + 1),  # right node
            # bottom row, right to left
            Position(node.row + 1, node.col + 1),
            Position(node.row + 1, node.col),
            Position(node.row + 1, node.col - 1),
            Position(node.row, node.col - 1),  # left node
        ]
        if is_inside(pos, maze)
    ]
    return nb


def mask_unreachable(maze, allow_value):
    masked = numpy.array(maze)
    for node, value in numpy.ndenumerate(maze):
        pos = Position(*node)
        if value == 0:
            nbs = set(box_neighbour_values(maze, pos))
            other = nbs - {0, allow_value}
            if other:
                masked.itemset(pos, WALL_MARK)
    return masked


def distance_heuristic(left, right):
    """uniform grid distance_heuristic"""
    left_row, left_col = left
    right_row, right_col = right

    # return math.sqrt((right_row - left_row) ** 2 + (right_col - left_col) ** 2)
    dist = max(abs(right_row - left_row), abs(right_col - left_col))
    # dist = (abs(right_row - left_row) + abs(right_col - left_col))

    return 2 * dist


def cross_neighbours_x(maze, node, goal, visited):
    goal_value = maze[goal]

    neighbours = [
        (distance_heuristic(left=pos, right=goal), pos)
        for pos in [
            Position(node.row - 1, node.col),
            Position(node.row, node.col - 1),
            Position(node.row + 1, node.col),
            Position(node.row, node.col + 1),
        ]
        if is_inside(pos, maze)
        and pos not in visited
        and maze.item(pos) in {0, goal_value}
    ]

    return neighbours


def reconstruct_path(trail, position):
    path = deque([position])
    while position in trail:
        position = trail[position]
        path.appendleft(position)

    return list(path)


def find_a_star(original_maze, start, finish):
    """A* finds a path from start to finish."""
    goal_value = original_maze.item(start) or original_maze.item(finish)
    maze = mask_unreachable(original_maze, goal_value)
    queue = [(0, start)]
    trail = {}

    # gScore[n] is the cost of the cheapest path from start to n
    g_score = {start: 0}

    # fScore[n] := gScore[n] + h(n). fScore[n] represents our pos best guess as to
    # how short a path from start to finish can be if it goes through n.
    f_score = {start: distance_heuristic(start, finish)}
    visited = set()

    while queue:
        pos_f_score, pos = heappop(queue)
        pos_g_score = g_score[pos]
        visited.add(pos)

        if pos == finish:
            return reconstruct_path(trail, pos)

        for h_dist, neighbour in cross_neighbours_x(maze, pos, finish, visited):
            # tentative_gScore is the distance_heuristic from start to the neighbour through pos
            came_from = trail.get(pos)
            if came_from is None:
                came_from = pos

            d_from = pos[0] - came_from[0], pos[1] - came_from[1]
            d_to = neighbour[0] - pos[0], neighbour[1] - pos[1]
            direction_change_penalty = 1 if d_from == d_to else 3
            new_g_score = pos_g_score + direction_change_penalty
            # must add conflict penalty

            if new_g_score < g_score.get(neighbour, math.inf):
                # This path to neighbour is better than any previous one. Record it!
                trail[neighbour] = pos
                g_score[neighbour] = new_g_score
                f_score[neighbour] = new_g_score + h_dist
                if neighbour not in queue:
                    heappush(queue, (f_score[neighbour], neighbour))

    return []


def free_neighbours(maze, node, visited):
    neighbours = [
        pos
        for pos in [
            Position(node.row - 1, node.col),
            Position(node.row, node.col - 1),
            Position(node.row + 1, node.col),
            Position(node.row, node.col + 1),
        ]
        if is_inside(pos, maze) and pos not in visited and maze.item(pos) == 0
    ]

    return neighbours


def propagate_wave(maze, position):
    wave = numpy.full_like(maze, WALL_MARK)

    queue = deque()
    queue.append((0, position))
    seen = set([position])

    while queue and len(queue) < 100:
        pulse, pos = queue.popleft()
        wave.itemset(pos, pulse)

        nbs = free_neighbours(maze, pos, seen)
        for nb in nbs:
            if nb not in seen:
                queue.append((pulse + 1, nb))
                seen.add(nb)

    return wave


def m_distance(left, right):
    """manhattan distance"""
    left_row, left_col = left
    right_row, right_col = right

    # return math.sqrt((right_row - left_row) ** 2 + (right_col - left_col) ** 2)
    # return max(abs(right_row - left_row), abs(right_col - left_col))
    return abs(right_row - left_row) + abs(right_col - left_col)


def pair_by_distance(nodes):
    n = len(nodes)
    if n < 2:
        return []

    dist_nodes = []
    for option in combinations(nodes, 2):
        left, right = option
        dist = m_distance(left, right)
        heappush(dist_nodes, (dist, option))

    seen = set()
    pairs = list()

    while len(seen) < n:
        dist, pair = heappop(dist_nodes)
        left, right = pair
        if left not in seen or right not in seen:
            seen.update(pair)
            pairs.append(pair)

    return pairs


def get_me_some(original_maze):
    maze = numpy.array(original_maze)
    outcome = numpy.array(original_maze)

    # scan pins
    pins = defaultdict(list)
    for pos, value in numpy.ndenumerate(maze):
        if value > 0:
            pins[value].append(Position(*pos))

    for i in pins:
        pairs = pair_by_distance(pins[i])
        # find a path between each pair
        for left, right in pairs:
            path = find_a_star(maze, left, right)

            if path:
                for node in path:
                    outcome.itemset(node, i)

    return outcome
