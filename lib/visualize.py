import pygame
import math
from lib.world import Terrain, Species

WIDTH = 800
HEIGHT = 800
CELL_SIZE = math.floor(HEIGHT / 100)

def draw_terrain(screen, world):
    terrain_surface = pygame.Surface((WIDTH, HEIGHT))
    for x in range(100):
        for y in range(100):
            cell = world[x * 100 + y]
            color = (0, 0, 255) if cell.terrain == Terrain.WATER else (0, 255, 0)
            pygame.draw.rect(terrain_surface, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    return terrain_surface

def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ecosystem simulation")
    return screen

def initial_draw(screen, world):
    screen.fill((255, 255, 255))
    terrain_surface = draw_terrain(screen, world)
    screen.blit(terrain_surface, (0, 0))
    pygame.display.flip()

def draw_world(screen, world):
    screen.fill((255, 255, 255))

    for x in range(100):
        for y in range(100):
            cell = world[x * 100 + y]
            color = (18, 53, 163) if cell.terrain == Terrain.WATER else (112, 180, 40)
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

            for species in Species:
                biomass = cell.biomass

                if biomass[species] > 0:
                    color = {
                        Species.PLANKTON: (255, 255, 255),
                        Species.ANCHOVY: (255, 0, 0),
                        Species.COD: (0, 0, 0)
                    }[species]
                    pygame.draw.circle(screen, color, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), math.sqrt(biomass[species]) * 5)



    pygame.display.flip()
            
def quit_pygame():
    pygame.quit()
