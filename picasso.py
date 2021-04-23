"""An opinionated drawing engine built on pygame"""
import pygame
import abc
import time


class PicassoEngine(metaclass=abc.ABCMeta):
    def __init__(self, window_size, name=None):
        self.window_size = window_size
        self.name = name or self.__class__.__name__
        self.screen = None

    def __enter__(self):
        print(" >> engine started")
        if self.screen is not None:
            raise RuntimeError("Video screen is already initialized.")

        pygame.init()
        self.screen = pygame.display.set_mode(
            self.window_size, pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.RESIZABLE
        )
        pygame.display.set_caption(self.name)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(" >> engine stopped")
        pygame.display.quit()
        return exc_type is None

    def run(self):
        if self.screen is None:
            raise RuntimeError("Video screen is not initialized.")

        keep_playing = True
        while keep_playing:
            cycle_start = time.perf_counter_ns()

            self.on_paint()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    keep_playing = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.on_click(event)
                elif event.type == pygame.MOUSEMOTION:
                    self.on_mouse_motion(event)
                elif event.type == pygame.KEYDOWN:
                    self.on_key(event)

            cycle_end = time.perf_counter_ns()
            duration = (cycle_end - cycle_start) // 1_000_000
            if duration < 33:
                pygame.time.wait(33 - duration)

    @abc.abstractmethod
    def on_click(self, event):
        pass

    @abc.abstractmethod
    def on_mouse_motion(self, event):
        pass

    @abc.abstractmethod
    def on_key(self, event):
        pass

    @abc.abstractmethod
    def on_paint(self):
        pass
