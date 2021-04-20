from picasso import PicassoEngine
from pathfinding import find_a_star, propagate_wave, is_inside, Position
from pathfinding import WALL_MARK, PATH_MARK
from palettes import WAVES_PALETTE, PINOUT_PALETTE, BLACK, GRAY, WHITE

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
TOP_OFFSET = 20 + PADDED_CELL * 2
LEFT_OFFSET = 20 + PADDED_CELL * 2


class DaliPathPainter(PicassoEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_counter = 0
        self.maze = None
        self.wave = None
        self.data_files = None
        self.route_rq = (None, None)
        self.live_routing = False

    def post_init(self):
        self.data_files = {
            i + 49: name for i, name in enumerate(sorted(glob.glob("*.csv")))
        }
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
        pos = Position(row, col)

        if not is_inside(pos, self.maze):
            return

        if event.button == 1:
            self.route_rq = (pos, self.route_rq[1])
            self.find_a_route()
        elif event.button == 3:
            self.route_rq = (self.route_rq[0], pos)
            self.wave = propagate_wave(self.maze, pos)

    def on_mouse_motion(self, event):
        if not self.live_routing:
            return

        x, y = event.pos
        col = (x - LEFT_OFFSET - PADDING // 2) // PADDED_CELL
        row = (y - TOP_OFFSET - PADDING // 2) // PADDED_CELL
        pos = Position(row, col)

        if not is_inside(pos, self.maze):
            return

        self.paint_maze()
        self.paint_wave_route(pos)

    def on_key(self, event):
        if event.key == pygame.K_ESCAPE:
            bye = pygame.event.Event(pygame.QUIT)
            pygame.event.post(bye)
        elif event.key in self.data_files:
            self.maze = load_csv(self.data_files[event.key])
            self.wave = None
            self.route_rq = (None, None)
            self.live_routing = False
            self.paint_maze()
        elif event.key == pygame.K_F1:
            if self.data_files:
                print(" >> all available data sets")
                for k, name in self.data_files.items():
                    print(f"\t'{chr(k)}' -> {name}")
        elif event.key == pygame.K_F2:
            # self.paint_wave()
            self.live_routing = not self.live_routing
        elif event.key == pygame.K_KP_PLUS:
            self.step_grow()

    def paint_cell(self, position, color):
        r, c = position
        y, x = TOP_OFFSET + r * PADDED_CELL, LEFT_OFFSET + c * PADDED_CELL
        cell_rect = pygame.Rect(
            x + PADDING // 2, y + PADDING // 2, CELL_SIZE, CELL_SIZE
        )
        pygame.draw.rect(self.screen, color, cell_rect)

    def paint_bounding_box(self):
        self.screen.fill(BLACK)

        rows, cols = self.maze.shape
        bounding_box = pygame.Rect(
            20, 20, PADDED_CELL * (cols + 4), PADDED_CELL * (rows + 4)
        )
        maze_box = pygame.Rect(
            20 + PADDED_CELL,
            20 + PADDED_CELL,
            PADDED_CELL * (cols + 2),
            PADDED_CELL * (rows + 2),
        )
        pygame.draw.rect(self.screen, WHITE, bounding_box)
        pygame.draw.rect(self.screen, BLACK, maze_box)

    def paint_maze(self):
        if self.maze is None:
            print("warning: no maze found, skipped painting.")
            return

        self.paint_bounding_box()

        for pos, value in numpy.ndenumerate(self.maze):
            self.paint_cell(pos, PINOUT_PALETTE[value])

    def paint_wave(self):
        if self.wave is None:
            print("warning: wave pulse is missing, skipped painting.")
            return

        self.paint_bounding_box()

        for pos, value in numpy.ndenumerate(self.wave):
            self.paint_cell(pos, WAVES_PALETTE[value % 64])

    def paint_wave_route(self, start):
        if self.wave is None:
            print("warning: wave pulse is missing.")
            return

        if self.wave.item(start) < 0:
            print("warning: cannot start from wall", start)
            return

        # print("showing route from", start)
        # self.screen.fill(BLACK)

        pos = start
        finish = self.route_rq[1]

        while pos != finish:
            # paint current cell
            col = pygame.Color(0, 255, 0)
            self.paint_cell(pos, WAVES_PALETTE[PATH_MARK])

            # pick the next cell
            neighbours = [
                (self.wave.item(nb_pos), nb_pos)
                for nb_pos in [
                    Position(pos.row, pos.col + 1),
                    Position(pos.row - 1, pos.col),
                    Position(pos.row, pos.col - 1),
                    Position(pos.row + 1, pos.col),
                ]
                if is_inside(nb_pos, self.wave) and self.wave.item(nb_pos) >= 0
            ]
            heapq.heapify(neighbours)
            _, pos = heapq.heappop(neighbours)

        # paint last cell
        self.paint_cell(pos, WAVES_PALETTE[PATH_MARK])

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

        if path:
            for node in path:
                value = self.maze.item(node)
                if value == 0:
                    self.maze.itemset(node, PATH_MARK)
            self.paint_maze()
        else:
            print("waning: no path found.")


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
