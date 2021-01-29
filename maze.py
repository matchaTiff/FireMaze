import pygame
import random

# colors
BLACK = (50, 50, 50)
WHITE = (240, 237, 235)
GREEN = (125, 148, 78)
RED = (148, 78, 78)

# initializes pygame
pygame.init()

# window
WINDOW_SIZE = (500, 500)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Fire Maze")
screen.fill(BLACK)

# change dimension and probability here
dim = 10
p = 0.1

MARGIN = 1
CELL_SIZE = WINDOW_SIZE[0]/dim-1

# creates grid based on the dimensions given
grid = []
def makeGrid(dim, p):
  for row in range(dim):
    grid.append([])
    for column in range(dim):

      # get a random number between 1 and 99 (included)
      # and determine at random if cell should be fire
      randomInt = random.randint(1, 99)
      if(randomInt <= p*100):
        # 0 = safe
        # 1 = fire
        # 2 = start
        # 3 = end
        grid[row].append(1)
        # create rectangle with margins based on it's position
        cell = pygame.Rect((MARGIN + CELL_SIZE) * column + MARGIN,
                           (MARGIN + CELL_SIZE) * row + MARGIN,
                            CELL_SIZE,
                            CELL_SIZE)
        # draw cells to the screen
        pygame.draw.rect(screen, BLACK, cell)
      
      # make cell empty
      else:
        grid[row].append(0)
        cell = pygame.Rect((MARGIN + CELL_SIZE) * column + MARGIN,
                           (MARGIN + CELL_SIZE) * row + MARGIN,
                            CELL_SIZE,
                            CELL_SIZE)
        pygame.draw.rect(screen, WHITE, cell)

  # color start cell
  cell = pygame.Rect((MARGIN + CELL_SIZE) * 0 + MARGIN,
                           (MARGIN + CELL_SIZE) * 0 + MARGIN,
                            CELL_SIZE,
                            CELL_SIZE)
  pygame.draw.rect(screen, GREEN, cell)
  
  # color goal cell
  cell = pygame.Rect((MARGIN + CELL_SIZE) * (dim-1) + MARGIN,
                           (MARGIN + CELL_SIZE) * (dim-1) + MARGIN,
                            CELL_SIZE,
                            CELL_SIZE)
  pygame.draw.rect(screen, RED, cell)

makeGrid(dim, p)

# update entire display so the rectangles are actually drawn on the screen
pygame.display.flip()



# keep program running until user exits the window
running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False