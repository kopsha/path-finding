"""An opinionated drawing engine built on pygame"""
import pygame
import abc


class PicassoEngine(metaclass=abc.ABCMeta):

    def __init__(self, window_size, name=None):
        self.window_size = window_size
        self.name = name or self.__class__.__name__
        self.screen = None

    def __enter__(self):
        if not self.screen is None:
            raise RuntimeError("Video screen is already initialized.")

        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size, pygame.DOUBLEBUF)
        pygame.display.set_caption(self.name)
        self.info = pygame.display.Info()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pygame.display.quit()
        return exc_type is None

    def run(self):
        if self.screen is None:
            raise RuntimeError("Video screen is not initialized.")

        keep_playing = True
        while keep_playing:

            self.on_paint()
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    keep_playing = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.on_click(event)
                elif event.type == pygame.KEYDOWN:
                    self.on_key(event)

            pygame.time.wait(10)


    @abc.abstractmethod
    def on_click(self, event): pass

    @abc.abstractmethod
    def on_key(self, event): pass

    @abc.abstractmethod
    def on_paint(self): pass
