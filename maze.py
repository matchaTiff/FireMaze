import pygame
import random
import sys

# colors
BLACK = (50, 50, 50)
WHITE = (240, 237, 235)
GREEN = (125, 148, 78)
RED = (148, 78, 78)
GREY = (161, 157, 157)

# window
WINDOW_SIZE = (500, 500)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Fire Maze")
screen.fill(BLACK)

MARGIN = 1
dim = 4
CELL_SIZE = WINDOW_SIZE[0] / dim - 1

# initializes pygame
pygame.init()

if sys.version_info[0] < 3:
    raise Exception("Python 3 is required for this program.")


def get_maze(p: float = 0.1):
    """
    Creates dim x dim grid with p probability blocks
    :param p: probability of a single space being blocked
    :return: maze as a grid/2 dimensional list
    """
    grid = []
    for row in range(dim):
        grid.append([])
        for column in range(dim):
            # choose if blocked or not via weights
            # 0 = safe, 1 = blocked
            grid[row].append(random.choices([1, 0], weights=(p, 1 - p))[0])
    # start and end must be free
    grid[0][0] = 0
    grid[dim - 1][dim - 1] = 0
    return grid


def start_fire(_maze: list):
    """
    Selects a random valid (not start, end, or blocked) block and "starts a fire" there.
    :param _maze: maze as a grid
    :return: tuple of the new maze and the coordinates of the selected block.
    """
    dim = len(_maze)
    valid = False

    while not valid:
        row = random.randint(0, dim - 1)
        col = random.randint(0, dim - 1)
        # don't start fire on start block
        if row == 0 and col == 0:
            continue
        # don't start fire on end block
        elif row == dim - 1 and col == dim - 1:
            continue
        # check if the coordinate is blocked
        # if yes, then still not valid
        if _maze[row][col]:
            continue
        # otherwise, we start the fire there
        else:
            _maze[row][col] = 2
            return _maze, (row, col)


def show_maze(_maze: list):
    """
    Takes in a maze and displays it on a new pygame window.
    :param _maze:
    :return: n/a
    """

    for row in range(dim):
        for col in range(dim):
            if _maze[row][col] == 1:
                # create rectangle with margins based on it's position
                cell = pygame.Rect((MARGIN + CELL_SIZE) * col + MARGIN,
                                   (MARGIN + CELL_SIZE) * row + MARGIN,
                                   CELL_SIZE,
                                   CELL_SIZE)
                # draw cells to the screen
                pygame.draw.rect(screen, BLACK, cell)
            elif _maze[row][col] == 2:
                cell = pygame.Rect((MARGIN + CELL_SIZE) * col + MARGIN,
                                   (MARGIN + CELL_SIZE) * row + MARGIN,
                                   CELL_SIZE,
                                   CELL_SIZE)
                pygame.draw.rect(screen, RED, cell)
            else:
                cell = pygame.Rect((MARGIN + CELL_SIZE) * col + MARGIN,
                                   (MARGIN + CELL_SIZE) * row + MARGIN,
                                   CELL_SIZE,
                                   CELL_SIZE)
                pygame.draw.rect(screen, WHITE, cell)
    """
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
    pygame.draw.rect(screen, GREEN, cell)
    """
    # update entire display so the rectangles are actually drawn on the screen
    pygame.display.flip()


def get_valid_neighbors(_maze, current, visited):
    """
  Get adjacent neighbors which are not an obstacle
  :param _maze: maze as a grid
  :param current: current cell
  :param visited: list of visited cell positions
  :return: list of neighbors
  """
    neighbors = []
    row = current[0]
    col = current[1]

    # left
    if row > 0 and (row - 1, col) not in visited and _maze[row - 1][col] != 1 and _maze[row - 1][col] != 2:
        neighbors.append((row - 1, col))
    # right
    if row + 1 < dim and (row + 1, col) not in visited and _maze[row + 1][col] != 1 and _maze[row + 1][col] != 2:
        neighbors.append((row + 1, col))
    # down
    if col > 0 and (row, col - 1) not in visited and _maze[row][col - 1] != 1 and _maze[row][col - 1] != 2:
        neighbors.append((row, col - 1))
    # up
    if col + 1 < dim and (row, col + 1) not in visited and _maze[row][col + 1] != 1 and _maze[row][col + 1] != 2:
        neighbors.append((row, col + 1))

    return neighbors


def dfs(_maze, start, goal):
    """
  Runs dfs on the maze and determines whether the goal is reachable
  :param _maze: maze as a grid
  :param start: starting cell
  :param goal: goal cell
  :return: true if reachable, false otherwise
  """
    visited = []

    # fringe is a list stack
    fringe = [start]
    visited.append(start)

    # while fringe is not empty
    while fringe:
        current = fringe.pop()

        # color visited cell except for start and goal
        if (current != start and current != goal):
            cell = pygame.Rect((MARGIN + CELL_SIZE) * current[1] + MARGIN,
                               (MARGIN + CELL_SIZE) * current[0] + MARGIN,
                               CELL_SIZE,
                               CELL_SIZE)
            pygame.draw.rect(screen, GREY, cell)
            # animate path
            pygame.display.update()
            pygame.time.delay(30)

        if current == goal:

            print('\nVisited:')
            print(visited)

            print('\nElements in fringe:')
            print(fringe)

            pygame.display.flip()
            print('\nSUCCESS')
            return True

        else:
            neighbors = get_valid_neighbors(_maze, current, visited)
            for neighbor in neighbors:
                visited.append(neighbor)
                fringe.append(neighbor)

    print('\nVisited:')
    print(visited)

    print('\nElements in fringe:')
    print(fringe)

    print('\nFAILED')
    return False


def bfs(_maze, start, goal):
    """
  Runs bfs on the maze and determines the shortest path from start to goal
  :param _maze: maze as a grid
  :param start: starting cell
  :param goal: goal cell
  :return: shortest path
  """
    visited = []

    fringe = [(start, [start])]
    visited.append(start)

    while fringe:

        # get the first element from queue
        current, s_path = fringe.pop(0)

        if current == goal:

            # color shortest path
            for i in s_path:
                if (i != start and i != goal):
                    cell = pygame.Rect((MARGIN + CELL_SIZE) * i[1] + MARGIN,
                                       (MARGIN + CELL_SIZE) * i[0] + MARGIN,
                                       CELL_SIZE,
                                       CELL_SIZE)
                    pygame.draw.rect(screen, GREY, cell)
                    # animate path
                    pygame.display.update()
                    pygame.time.delay(30)

            print('\nVisited:')
            print(visited)

            print('\nElements in fringe:')
            print(fringe)

            pygame.display.flip()
            print('\nSUCCESS')
            print('Shortest path:')
            print(s_path + [goal])
            return True

        else:
            neighbors = get_valid_neighbors(_maze, current, visited)
            print('Neighbors:')
            print(neighbors)
            for neighbor in neighbors:
                visited.append(neighbor)
                fringe.append((neighbor, s_path + [current]))

    print('\nVisited:')
    print(visited)

    print('\nElements in fringe:')
    print(fringe)

    print('\nFAILED')
    return False


maze = get_maze()
print(maze)

show_maze(maze)

fired = start_fire(maze)
print(f"Fire starts: {fired[1]}")
show_maze(fired[0])

bfs(maze, (0, 0), (dim - 1, dim - 1))

# keep program running until user exits the window
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
