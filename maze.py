import pygame
import random


def get_maze(dim: int = 10, p: float = 0.1):
    # creates grid based on the dimensions given
    grid = []
    for row in range(dim):
        grid.append([])
        for column in range(dim):
            # choose if blocked or not via weights
            # 0 = safe, 1 = blocked
            print(f"{p}\t{1-p}")
            print(random.choices([True, False], weights=(p, 1 - p), k=3))
            if random.choices([True, False], weights=(p, 1 - p), k=3)[0]:
                grid[row].append(1)
            else:
                grid[row].append(0)
    return grid


def show_maze(maze: list):
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

    MARGIN = 1
    dim = len(maze)
    CELL_SIZE = WINDOW_SIZE[0] / dim - 1

    for row in range(dim):
        for col in range(dim):
            if maze[row][col]:
                # create rectangle with margins based on it's position
                cell = pygame.Rect((MARGIN + CELL_SIZE) * col + MARGIN,
                                   (MARGIN + CELL_SIZE) * row + MARGIN,
                                   CELL_SIZE,
                                   CELL_SIZE)
                # draw cells to the screen
                pygame.draw.rect(screen, BLACK, cell)
            else:
                cell = pygame.Rect((MARGIN + CELL_SIZE) * col + MARGIN,
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
    cell = pygame.Rect((MARGIN + CELL_SIZE) * (dim - 1) + MARGIN,
                       (MARGIN + CELL_SIZE) * (dim - 1) + MARGIN,
                       CELL_SIZE,
                       CELL_SIZE)
    pygame.draw.rect(screen, RED, cell)

    # update entire display so the rectangles are actually drawn on the screen
    pygame.display.flip()


maze = get_maze()
print(maze)

show_maze(maze)

# keep program running until user exits the window
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
