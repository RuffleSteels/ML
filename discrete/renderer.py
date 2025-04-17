# env.py or a new file like renderer.py
import pygame
import math

class PygameRenderer:
    def __init__(self, width=600, height=400):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CartPole")
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.scale = 100  # pixels per meter
        self.cart_width = 80
        self.cart_height = 30
        self.pole_length = 140  # in pixels

    def draw(self, state_tuple):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        x, _, theta, _ = state_tuple
        center_x = self.width // 2 + int(x * self.scale)
        cart_y = self.height // 2

        # Clear screen
        self.screen.fill((255, 255, 255))

        # Draw cart
        cart_rect = pygame.Rect(center_x - self.cart_width // 2, cart_y,
                                self.cart_width, self.cart_height)
        pygame.draw.rect(self.screen, (0, 200, 255), cart_rect)

        # Draw pole
        pole_x = center_x + self.pole_length * math.sin(theta)
        pole_y = cart_y - self.pole_length * math.cos(theta)
        pygame.draw.line(self.screen, (255, 100, 100),
                         (center_x, cart_y), (pole_x, pole_y), 6)

        pygame.display.flip()
        self.clock.tick(60)
