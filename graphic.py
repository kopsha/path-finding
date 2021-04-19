from picasso import PicassoEngine
from pathfinding import find_a_star, propagate_wave
from pathfinding import WALL_MARK, PATH_MARK
from palletes import WAVES_PALLETE

import pygame
import numpy
import csv
import glob
import heapq

from collections import Counter, defaultdict


WINDOW_SIZE = 1024, 768
CELL_SIZE = 13
PADDING = 2
PADDED_CELL = CELL_SIZE + PADDING
TOP_OFFSET = 20 + PADDED_CELL*2
LEFT_OFFSET = 20 + PADDED_CELL*2

BLACK = pygame.Color(0, 0, 0)
GRAY = pygame.Color(96, 96, 96)
WHITE = pygame.Color(255, 255, 255)

PALETTE = {
    PATH_MARK: GRAY,
    WALL_MARK: WHITE,
    0: BLACK,
    1: pygame.Color(173, 0, 252),
    2: pygame.Color(90, 0, 252),
    3: pygame.Color(185, 254, 3),
    4: pygame.Color(1, 250, 173),
    5: pygame.Color(1, 249, 255),
    6: pygame.Color(247, 255, 2),
    7: pygame.Color(255, 0, 86),
    8: pygame.Color(254, 170, 1),
    9: pygame.Color(1, 166, 254),
    10: pygame.Color(2, 83, 254),
    11: pygame.Color(62, 253, 4),
    12: pygame.Color(255, 0, 6),
    13: pygame.Color(8, 0, 253),
    14: pygame.Color(255, 0, 165),
    15: pygame.Color(2, 0, 253),
    16: pygame.Color(255, 0, 245),
    17: pygame.Color(255, 0, 0),
    18: pygame.Color(124, 254, 3),
    19: pygame.Color(0, 252, 10),
    20: pygame.Color(254, 85, 1),
    21: pygame.Color(0, 251, 92),
}


class DaliPathPainter(PicassoEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_counter = 0
        self.maze = None
        self.wave = None
        self.data_files = None
        self.route_rq = (None, None)

    def post_init(self):
        self.data_files = {i + 49: name for i, name in enumerate(sorted(glob.glob("*.csv")))}
        if self.data_files:
            self.maze = load_csv(self.data_files[49])
            self.paint_maze()

    def on_paint(self):
        self.frame_counter += 1
        if self.frame_counter % 1000 == 0:
            print(f" >> rendered {self.frame_counter // 1000}k paintings")

    def on_click(self, event):
        x, y = event.pos
        col = (x - LEFT_OFFSET - PADDING // 2) // PADDED_CELL
        row = (y - TOP_OFFSET - PADDING // 2) // PADDED_CELL
        rows, cols = self.maze.shape

        if 0 <= row < rows and 0 <= col < cols:
            if event.button == 1:
                self.route_rq = ((row, col), self.route_rq[1])
                # self.paint_wave_route((row, col))
                # self.paint_wave()
            elif event.button == 3:
                self.route_rq = (self.route_rq[0], (row, col))
                self.wave = propagate_wave(self.maze, (row, col))

            self.find_a_route()

    def on_key(self, event):
        if event.key == pygame.K_ESCAPE:
            bye = pygame.event.Event(pygame.QUIT)
            pygame.event.post(bye)
        elif event.key in self.data_files:
            self.maze = load_csv(self.data_files[event.key])
            self.route_rq = (None, None)
            self.paint_maze()
        elif event.key == pygame.K_F1:
            if self.data_files:
                print(" F1 >> all available data sets")
                for k, name in self.data_files.items():
                    print(f"\t'{chr(k)}' -> {name}")
        elif event.key == pygame.K_F2:
            self.paint_wave()
        elif event.key == pygame.K_KP_PLUS:
            self.step_grow()


    def paint_maze(self):
        if self.maze is None:
            print("skipping, no maze found.")
            return

        self.screen.fill(BLACK)
        rows, cols = self.maze.shape

        bounding_box = pygame.Rect(
            20, 20,
            PADDED_CELL * (cols + 4), PADDED_CELL * (rows + 4)
        )
        maze_box = pygame.Rect(
            20 + PADDED_CELL, 20 + PADDED_CELL,
            PADDED_CELL * (cols + 2), PADDED_CELL * (rows + 2)
        )
        pygame.draw.rect(self.screen, WHITE, bounding_box)
        pygame.draw.rect(self.screen, BLACK, maze_box)

        for pos, value in numpy.ndenumerate(self.maze):
            if value:
                col = PALETTE[value]
                r, c = pos
                y, x = TOP_OFFSET + r * PADDED_CELL, LEFT_OFFSET + c * PADDED_CELL
                cell_rect = pygame.Rect(x + PADDING//2, y + PADDING//2, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, col, cell_rect)

    def find_a_route(self):
        if not all(self.route_rq):
            return

        start, finish = self.route_rq
        s_value = self.maze.item(start)
        f_value = self.maze.item(finish)
        if s_value > 0 and f_value > 0 and s_value != f_value:
            print(f"cannot route {s_value} to {f_value}")
            return

        print(f"Looking for a route {s_value or f_value} from {start} to {finish}.")
        path = find_a_star(self.maze, start, finish)
        print("found:", path)
        for node in path:
            value = self.maze.item(node)
            if value == 0:
                self.maze.itemset(node, PATH_MARK)
        self.paint_maze()

    def paint_wave(self):
        if self.wave is None:
            print("wave pulse is missing.")
            return

        self.screen.fill(BLACK)
        rows, cols = self.maze.shape

        bounding_box = pygame.Rect(
            20, 20,
            PADDED_CELL * (cols + 4), PADDED_CELL * (rows + 4)
        )
        maze_box = pygame.Rect(
            20 + PADDED_CELL, 20 + PADDED_CELL,
            PADDED_CELL * (cols + 2), PADDED_CELL * (rows + 2)
        )
        pygame.draw.rect(self.screen, WHITE, bounding_box)
        pygame.draw.rect(self.screen, BLACK, maze_box)

        for pos, value in numpy.ndenumerate(self.wave):
            if value > 0:
                col = WAVES_PALLETE[value % 64]
                r, c = pos
                y, x = TOP_OFFSET + r * PADDED_CELL, LEFT_OFFSET + c * PADDED_CELL
                cell_rect = pygame.Rect(x + PADDING//2, y + PADDING//2, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, col, cell_rect)

    def paint_wave_route(self, start):
        if self.wave is None:
            print("wave pulse is missing.")
            return

        if self.wave.item(start) < 0:
            print("cannot start from wall", start)
            return

        print("showing route from", start)
        # self.screen.fill(BLACK)
        rows, cols = self.maze.shape

        # bounding_box = pygame.Rect(
        #     20, 20,
        #     PADDED_CELL * (cols + 4), PADDED_CELL * (rows + 4)
        # )
        # maze_box = pygame.Rect(
        #     20 + PADDED_CELL, 20 + PADDED_CELL,
        #     PADDED_CELL * (cols + 2), PADDED_CELL * (rows + 2)
        # )
        # pygame.draw.rect(self.screen, WHITE, bounding_box)
        # pygame.draw.rect(self.screen, BLACK, maze_box)

        pos = start
        finish = self.route_rq[1]

        cnt = 0
        while pos != finish and cnt < 30:
            cnt += 1
            # paint current cell
            col = pygame.Color(0, 255, 0)
            r, c = pos
            y, x = TOP_OFFSET + r * PADDED_CELL, LEFT_OFFSET + c * PADDED_CELL
            cell_rect = pygame.Rect(x + PADDING//2, y + PADDING//2, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, col, cell_rect)

            # pick the next cell
            neighbours = [
                (self.wave.item((ri, ci)), (ri, ci))
                for ri, ci in [(r, c+1), (r-1, c), (r, c-1), (r+1, c)]
                if (0 <= ri < rows) and (0 <= ci < cols) and self.wave.item((ri, ci)) >= 0
            ]
            heapq.heapify(neighbours)
            print(pos, neighbours)
            _, pos = heapq.heappop(neighbours)

        # paint last cell
        col = pygame.Color(0, 255, 0)
        r, c = pos
        y, x = TOP_OFFSET + r * PADDED_CELL, LEFT_OFFSET + c * PADDED_CELL
        cell_rect = pygame.Rect(x + PADDING//2, y + PADDING//2, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, col, cell_rect)


def load_csv(filename):
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


def main():
    with DaliPathPainter(window_size=WINDOW_SIZE) as engine:
        engine.post_init()
        engine.run()


if __name__ == "__main__":
    main()
