from picasso import PicassoEngine

WINDOW_SIZE = 1024, 768
BLACK = (0, 0, 0, 0)

import pygame

class DaliPathPainter(PicassoEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_counter = 0

    def post_init(self):
        print(f" >> screen buffer {self.info.current_w}x{self.info.current_h}")
        # texture = pygame.image.load("fig_012_step_two_pin_3.png")
        # self.screen.blit(texture, (0, 0))

    def on_click(self, event):
        print("clicked", event.pos)

    def on_key(self, event):
        if event.key == pygame.K_ESCAPE:
            bye = pygame.event.Event(pygame.QUIT)
            pygame.event.post(bye)

    def render_frame(self):
        self.frame_counter += 1
        if self.frame_counter % 1000 == 0:
            print(f" >> {self.frame_counter}k paintings")


def main():
    """das spil"""

    with DaliPathPainter(window_size=WINDOW_SIZE) as engine:
        engine.post_init()
        engine.run()



if __name__ == "__main__":
    main()