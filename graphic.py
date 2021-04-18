from picasso import PicassoEngine

import pygame
import numpy
import csv
import glob


WINDOW_SIZE = 1024, 768
CELL_SIZE = 13
PADDING = 2
PADDED_CELL = CELL_SIZE + PADDING
TOP_OFFSET = 20 + PADDED_CELL*2
LEFT_OFFSET = 20 + PADDED_CELL*2

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
WALL_MARK = -1
PALETTE = {
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

    matrix = numpy.matrix(data, dtype=int)
    print(f" >> loaded {filename}", matrix.shape)

    return matrix


class DaliPathPainter(PicassoEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_counter = 0
        self.maze = None
        self.data_files = None
        self.route_rq = (None, None)

    def on_init(self):
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
            elif event.button == 3:
                self.route_rq = (self.route_rq[0], (row, col))

    def on_key(self, event):
        if event.key == pygame.K_ESCAPE:
            bye = pygame.event.Event(pygame.QUIT)
            pygame.event.post(bye)
        elif event.key in self.data_files:
            self.maze = load_csv(self.data_files[event.key])
            self.paint_maze()
        elif event.key == pygame.K_F1:
            if self.data_files:
                print(" F1 >> all available data sets")
                for k, name in self.data_files.items():
                    print(f"\t'{chr(k)}' -> {name}")

    def paint_maze(self):
        if self.maze is None:
            print("skipping, no maze found.")
            return

        white = pygame.Color(255, 255, 255)
        black = pygame.Color(0, 0, 0)
        self.screen.fill(black)

        rows, cols = self.maze.shape

        bounding_box = pygame.Rect(
            20, 20,
            PADDED_CELL * (cols + 4), PADDED_CELL * (rows + 4)
        )
        maze_box = pygame.Rect(
            20 + PADDED_CELL, 20 + PADDED_CELL,
            PADDED_CELL * (cols + 2), PADDED_CELL * (rows + 2)
        )
        pygame.draw.rect(self.screen, white, bounding_box)
        pygame.draw.rect(self.screen, black, maze_box)

        mit = numpy.nditer(self.maze, flags=["multi_index"])
        for value in mit:
            if value:
                col = PALETTE[int(value)]
                r, c = mit.multi_index
                y, x = TOP_OFFSET + r * PADDED_CELL, LEFT_OFFSET + c * PADDED_CELL
                cell_rect = pygame.Rect(x + PADDING//2, y + PADDING//2, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, col, cell_rect)


def main():
    with DaliPathPainter(window_size=WINDOW_SIZE) as engine:
        engine.on_init()
        engine.run()


if __name__ == "__main__":
    main()
