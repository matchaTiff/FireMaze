import pygame
import random
import sys
import time
import collections
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
from pygame.scrap import get

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
dim = 20
CELL_SIZE = WINDOW_SIZE[0] / dim - 1

# initializes pygame
pygame.init()

if sys.version_info[0] < 3:
    raise Exception("Python 3 is required for this program.")


def get_maze(p: float = 0.3):
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


def fire_get_neighbors(_maze, current):
    """
    Get fire cell's neighbors that are not an obstacle or on fire
    :param _maze: maze as a grid
    :param current: current cell
    :return: neighbors
    """
    neighbors = set()
    row = current[0]
    col = current[1]

    # left
    if row > 0 and (row - 1, col) and _maze[row - 1][col] != 1 and _maze[row - 1][col] != 2:
        neighbors.add((row - 1, col))
    # right
    if row + 1 < dim and (row + 1, col) and _maze[row + 1][col] != 1 and _maze[row + 1][col] != 2:
        neighbors.add((row + 1, col))
    # down
    if col > 0 and (row, col - 1) and _maze[row][col - 1] != 1 and _maze[row][col - 1] != 2:
        neighbors.add((row, col - 1))
    # up
    if col + 1 < dim and (row, col + 1) and _maze[row][col + 1] != 1 and _maze[row][col + 1] != 2:
        neighbors.add((row, col + 1))

    return neighbors


def count_fire(_maze, current):
    """
    Count number of adjacent cells that are on fire
    :param _maze: maze as a grid
    :param current: current cell
    :return: fire count
    """
    row = current[0]
    col = current[1]

    fire_count = 0

    # left
    if row > 0 and (row - 1, col) and _maze[row - 1][col] != 1 and _maze[row - 1][col] == 2:
        fire_count += 1
    # right
    if row + 1 < dim and (row + 1, col) and _maze[row + 1][col] != 1 and _maze[row + 1][col] == 2:
        fire_count += 1
    # down
    if col > 0 and (row, col - 1) and _maze[row][col - 1] != 1 and _maze[row][col - 1] == 2:
        fire_count += 1
    # up
    if col + 1 < dim and (row, col + 1) and _maze[row][col + 1] != 1 and _maze[row][col + 1] == 2:
        fire_count += 1

    return fire_count


def advance_fire_one_step(_maze, q):
    """
    Advance fire for every move
    :param _maze: maze as a grid
    :param q: flammability rate q
    :return: maze with fire
    """
    maze_copy = _maze
    np_maze = np.array(_maze)
    # get cells on fire
    fire_locs = np.asarray(np.where(np_maze == 2)).T.tolist()

    # go through each cell that is on fire
    for i in fire_locs:
        neighbors = fire_get_neighbors(_maze, (i[0], i[1]))
        for neighbor in neighbors:
            k = count_fire(_maze, neighbor)
            prob = 1 - (1 - q)**k
            # mark cell on fire given probability
            maze_copy[neighbor[0]][neighbor[1]] = (random.choices([2, 0], weights=(prob, 1 - prob))[0])
            # color cell red for fire
            color_fire(maze_copy, (neighbor[0], neighbor[1]))
    # print("\nfire locations: ")
    # print(fire_locs)
    return maze_copy


def color_fire(_maze: list, current):
    row = current[0]
    col = current[1]
    if _maze[row][col] == 2:
        cell = pygame.Rect((MARGIN + CELL_SIZE) * col + MARGIN,
                           (MARGIN + CELL_SIZE) * row + MARGIN,
                           CELL_SIZE,
                           CELL_SIZE)
        pygame.draw.rect(screen, RED, cell)


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

    # update entire display so the rectangles are actually drawn on the screen
    pygame.display.flip()


# for strategy 1
def get_neighbors_1(_maze, current, visited):
    """
  Get adjacent neighbors which are not an obstacle
  :param _maze: maze as a grid
  :param current: current cell
  :param visited: list of visited cell positions
  :return: list of neighbors
  """
    neighbors = set()
    row = current[0]
    col = current[1]

    # left
    if row > 0 and (row - 1, col) not in visited and _maze[row - 1][col] != 1:
        neighbors.add((row - 1, col))
    # right
    if row + 1 < dim and (row + 1, col) not in visited and _maze[row + 1][col] != 1:
        neighbors.add((row + 1, col))
    # down
    if col > 0 and (row, col - 1) not in visited and _maze[row][col - 1] != 1:
        neighbors.add((row, col - 1))
    # up
    if col + 1 < dim and (row, col + 1) not in visited and _maze[row][col + 1] != 1:
        neighbors.add((row, col + 1))

    return neighbors

def get_shortest_path_1(_maze, start, goal):
    """
    For Strategy 1, runs bfs on the maze and determines the shortest path from start to goal
    :param _maze: maze as a grid
    :param start: starting cell
    :param goal: goal cell
    :return: shortest path
    """
    visited = set(start)
    fringe = collections.deque([(start, [])])

    while fringe:

        # get the first element from queue
        current, s_path = fringe.popleft()

        if current == goal:

            # print('\nSUCCESS')
            # print('Shortest path:')
            # print(s_path + [goal])
            return True, s_path + [goal]

        else:
            neighbors = get_neighbors_1(_maze, current, visited)
            for neighbor in neighbors:
                visited.add(neighbor)
                fringe.append((neighbor, s_path + [current]))

    # print('\nFAILED')
    return False, s_path

def get_shortest_path_2(_maze, start, goal):
    """
    For Strategy 2, runs bfs on the maze and determines the shortest path from start to goal
    :param _maze: maze as a grid
    :param start: starting cell
    :param goal: goal cell
    :return: shortest path
    """
    visited = set(start)
    fringe = collections.deque([(start, [])])

    while fringe:

        # get the first element from queue
        current, s_path = fringe.popleft()

        if current == goal:

            # print('\nSUCCESS')
            # print('Shortest path:')
            # print(s_path + [goal])
            return True, s_path + [goal]

        else:
            neighbors = get_neighbors_2(_maze, current, visited)
            for neighbor in neighbors:
                visited.add(neighbor)
                fringe.append((neighbor, s_path + [current]))

    # print('\nFAILED')
    return False, s_path

def get_shortest_path_3(_maze, start, goal):
    """
    For Strategy 3, runs bfs on the maze and determines the shortest path from start to goal
    :param _maze: maze as a grid
    :param start: starting cell
    :param goal: goal cell
    :return: shortest path
    """
    visited = set(start)
    fringe = collections.deque([(start, [])])

    while fringe:

        # get the first element from queue
        current, s_path = fringe.popleft()

        if current == goal:

            # print('\nSUCCESS')
            # print('Shortest path:')
            # print(s_path + [goal])
            return True, s_path + [goal]

        else:
            neighbors = get_neighbors_3(_maze, current, visited)
            for neighbor in neighbors:
                visited.add(neighbor)
                fringe.append((neighbor, s_path + [current]))

    # print('\nFAILED')
    return False, s_path

# for strategy 2
def get_neighbors_2(_maze, current, visited):
    """
  Get adjacent neighbors which are not an obstacle and on fire
  :param _maze: maze as a grid
  :param current: current cell
  :param visited: list of visited cell positions
  :return: list of neighbors
  """
    neighbors = set()
    row = current[0]
    col = current[1]

    # left
    if row > 0 and (row - 1, col) not in visited and _maze[row - 1][col] != 1 and _maze[row - 1][col] != 2:
        neighbors.add((row - 1, col))
    # right
    if row + 1 < dim and (row + 1, col) not in visited and _maze[row + 1][col] != 1 and _maze[row + 1][col] != 2:
        neighbors.add((row + 1, col))
    # down
    if col > 0 and (row, col - 1) not in visited and _maze[row][col - 1] != 1 and _maze[row][col - 1] != 2:
        neighbors.add((row, col - 1))
    # up
    if col + 1 < dim and (row, col + 1) not in visited and _maze[row][col + 1] != 1 and _maze[row][col + 1] != 2:
        neighbors.add((row, col + 1))

    return neighbors


def is_next_to_fire(_maze, current):
    """
    Check if current cell is adjacent to fire
    :param _maze: generated maze
    :param current: current position
    :return: false if not fire, true otherwise
    """
    row = current[0]
    col = current[1]

    # left
    if row > 0 and _maze[row - 1][col] == 2:
        return True
    # right
    if row + 1 < dim and _maze[row + 1][col] == 2:
        return True
    # down
    if col > 0 and _maze[row][col - 1] == 2:
        return True
    # up
    if col + 1 < dim and _maze[row][col + 1] == 2:
        return True

    return False


def get_neighbors_3(_maze, current, visited):
    """
    Get adjacent neighbors that are not blocked, fire, or next to fire.
    :param _maze:
    :param current:
    :param visited:
    :return:
    """
    neighbors = set()
    row = current[0]
    col = current[1]

    neighbors = get_neighbors_2(_maze, current, visited)
    final = set()
    for n in neighbors:
        if not is_next_to_fire(_maze, n):
            final.add(n)

    return final


def color_s_path(current, s_path):
    # color shortest path
    for i in s_path:
        if (i != (0, 0) and i != (dim - 1, dim - 1)):
            cell = pygame.Rect((MARGIN + CELL_SIZE) * i[1] + MARGIN,
                               (MARGIN + CELL_SIZE) * i[0] + MARGIN,
                               CELL_SIZE,
                               CELL_SIZE)
            pygame.draw.rect(screen, GREEN, cell)
            # animate path
            pygame.display.update()
            pygame.time.delay(30)

    pygame.display.flip()


# for strategy 1
def bfs_1(_maze, q):
    """
    Runs strategy 1 on the maze using bfs
    :param q: flammability
    :return: false if burned, true otherwise
    """
    start = (0, 0)
    goal = (dim - 1, dim - 1)

    shortest_path = get_shortest_path_1(_maze, start, goal)

    # no path from start to goal
    if shortest_path[0] == False:
        return False

    # iterate through the shortest path
    for i in range(len(shortest_path[1])):
        current = (shortest_path[1])[i]

        # advance fire each time it moves
        advance_fire_one_step(_maze, q)

        if current == goal:
            # color_s_path(current, shortest_path[1])
            # print('\nSUCCESS')
            return True

        # fire was on the path, burned in fire
        if _maze[current[0]][current[1]] == 2:
            # print("\nFAILED")
            return False

# for strategy 2
def bfs_2(_maze, q):
    """
    Runs strategy 2 on the maze using bfs, recomputes path every step,
    check adjacent cells if they are on fire
    :param q: flammability
    :return: false if burned, true otherwise
    """
    start = (0, 0)
    goal = (dim - 1, dim - 1)
    current = start

    # get shortest path from start to goal
    shortest_path = get_shortest_path_2(_maze, start, goal)

    # no path from start to goal
    if shortest_path[0] == False:
        return False

    while current != goal:
        # no path from start to goal
        if shortest_path[0] == False:
            return False

        # make fire spread after each move
        advance_fire_one_step(_maze, q)
        # pygame.display.flip()
        # pygame.time.delay(40)

        # get next node on shortest path
        current = (shortest_path[1])[1]

        # # color current cell
        # cell = pygame.Rect((MARGIN + CELL_SIZE) * current[1] + MARGIN, (MARGIN + CELL_SIZE) * current[0] + MARGIN,
        #                    CELL_SIZE, CELL_SIZE)
        # pygame.draw.rect(screen, GREY, cell)
        # # animate path
        # pygame.display.update()
        # pygame.time.delay(40)

        # fire was on the path, burned in fire
        if _maze[current[0]][current[1]] == 2:
            # print("\nFAILED")
            return False
        
        # recompute shortest path from current node to goal
        shortest_path = get_shortest_path_2(_maze, current, goal)

    color_s_path(current, shortest_path[1])
    # print('\nSUCCESS')
    return True

# strategy 3
def bfs_3(_maze, q):
    """
    Runs strategy 3 on the maze using bfs, recomputes path every step,
    checks if adjacent cells of neighbors are on fire
    :param q: flammability
    :return: false if burned, true otherwise
    """
    start = (0, 0)
    goal = (dim - 1, dim - 1)
    current = start

    # get shortest path from start to goal
    shortest_path = get_shortest_path_3(_maze, start, goal)

    # no path from start to goal
    if shortest_path[0] == False:
        return False

    while current != goal:
        # no path from start to goal
        if shortest_path[0] == False:
            return False

        # make fire spread after each move
        advance_fire_one_step(_maze, q)
        pygame.display.flip()
        pygame.time.delay(40)

        # get next node on shortest path
        current = (shortest_path[1])[1]

        # # color current cell
        cell = pygame.Rect((MARGIN + CELL_SIZE) * current[1] + MARGIN, (MARGIN + CELL_SIZE) * current[0] + MARGIN,
                           CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, GREY, cell)
        # animate path
        pygame.display.update()
        pygame.time.delay(40)

        # fire was on the path, burned in fire
        if _maze[current[0]][current[1]] == 2:
            # print("\nFAILED")
            return False
        
        # recompute shortest path from current node to goal
        shortest_path = get_shortest_path_3(_maze, current, goal)

    color_s_path(current, shortest_path[1])
    # print('\nSUCCESS')
    return True



maze = get_maze(0.3)
fired = start_fire(maze)

# Strategy 1
# print(f"Fire starts: {fired[1]}")
# show_maze(maze)
# bfs_1(maze, 0.1)

# Strategy 2
# print(f"Fire starts: {fired[1]}")
# show_maze(maze)
# bfs_2(maze, 0.1)

# Strategy 3
print(f"Fire starts: {fired[1]}")
show_maze(maze)
bfs_3(maze, 0.3)

print(maze)

# keep program running until user exits the window
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
