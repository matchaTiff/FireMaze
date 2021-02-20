import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np

import random
import sys
import time
import collections
from heapq import heappop, heappush

dim = 500
if sys.version_info[0] < 3:
    raise Exception("Python 3 is required for this program.")


def get_maze(p):
    """
    Creates dim x dim grid with p probability blocks
    :param p: probability of a single space being blocked
    :return: maze as a grid/2 dimensional list
    """
    p = float(p)
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

def get_valid_neighbors(_maze, current, visited):
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

def dfs(_maze, start, goal):
    """
  Runs dfs on the maze and determines whether the goal is reachable
  :param _maze: maze as a grid
  :param start: starting cell
  :param goal: goal cell
  :return: true if reachable, false otherwise
  """
    # fringe is a stack
    fringe = [start]

    visited = set(start)

    # while fringe is not empty
    while fringe:
        current = fringe.pop()

        if current == goal:

            print('\nSUCCESS')
            return True

        else:
            neighbors = get_valid_neighbors(_maze, current, visited)
            # update() add items from other iterables.
            visited.update(neighbors)
            # adds list elements to fringe
            fringe.extend(neighbors)

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
    visited = set(start)
    fringe = collections.deque([start])
    num_nodes = 1
    while fringe:

        # get the first element from queue
        current = fringe.popleft()
        num_nodes += 1

        if current == goal:
            print('\nSUCCESS')
            return True, num_nodes

        else:
            neighbors = get_valid_neighbors(_maze, current, visited)
            visited.update(neighbors)
            fringe.extend(neighbors)

    print('\nFAILED')
    return False, num_nodes

def h(a, b):
    """
    Euclidean distance metric that determines the distance between two points
    :param a: position a
    :param b: position b
    :return: distance between a and b
    """
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def a_star(_maze, start, goal):
    # populate g_score with infinity
    g_score = []
    for row in range(dim):
        g_score.append([])
        for col in range(dim):
            g_score[row].append({(row, col): np.inf})

    # print(g_score)

    num_nodes = 1

    parent = {}
    visited = set(start)

    # cost of path from start to n
    g_score[start[0]][start[1]] = {start:0}

    # our guess of the cheapest path from start to goal
    f_score = {start:h(start, goal)}
    fringe = []

    heappush(fringe, (f_score[start], start))
    
    # while fringe is not empty
    while fringe:
        # get element with the lowest f_score
        current = heappop(fringe)[1]
        visited.add(current)

        num_nodes += 1

        if current == goal:
            # print('\nShortest path:')
            # print([goal] + get_s_path(parent, current))
            # color_s_path(current, [goal] + get_s_path(parent, current))

            print('\nSUCCESS')
            return True, num_nodes

        neighbors = get_valid_neighbors(_maze, current, visited)
        for neighbor in neighbors:
            # f(n) = path estimate from start to goal
            # g(n) = cost of path from start to n
            # h(n) = heuristic that estimates cost of shortest path from n to goal

            # start -> neighbor through current
            tentative_gScore = g_score[current[0]][current[1]].get(current) + h(current, neighbor)

            # check for better path
            if tentative_gScore < g_score[neighbor[0]][neighbor[1]].get(neighbor):
                parent[neighbor] = current
                g_score[neighbor[0]][neighbor[1]] = {neighbor:tentative_gScore}

                f_score[neighbor] = tentative_gScore + h(neighbor, goal)
                visited.add(neighbor)
                heappush(fringe, (f_score[neighbor], neighbor))

    print('\nFAILED')
    return False, num_nodes

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
            prob *= 100
            # mark cell on fire given probability
            if random.randint(0, 100) <= prob:
                maze_copy[neighbor[0]][neighbor[1]] = 2
            # color cell red for fire
            # color_fire(maze_copy, (neighbor[0], neighbor[1]))
    # print("\nfire locations: ")
    # print(fire_locs)
    return maze_copy

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
            print("\nFAILED")
            return False
        
        # recompute shortest path from current node to goal
        shortest_path = get_shortest_path_3(_maze, current, goal)

    print('\nSUCCESS')
    return True

def maze_valid(_maze, start, goal):
    """
    Check if goal is reachable is reachable from start as well as fire is reachable aswell
    :param _maze: generated maze
    :param start: starting positiion
    :param goal: ending position
    :return: false if no path, true otherwise
    """
    visited = set(start)
    fringe = collections.deque([start])
    reach_goal = False
    reach_fire = False
    while fringe:

        # get the first element from queue
        current = fringe.popleft()

        # check if a fire block can be reached
        if _maze[current[0]][current[1]] == 2:
            reach_fire = True

        # check if goal can be reach, and goal and fire are both true then maze is valid
        if current == goal:
            reach_goal = True
            if reach_goal == True and reach_fire == True:
                return True
            else:
                return False

        else:
            neighbors = get_neighbors_1(_maze, current, visited)
            for n in neighbors:
                if _maze[n[0]][n[1]] == 2:
                    reach_fire = True
            visited.update(neighbors)
            fringe.extend(neighbors)

    return False

def fire_get_neighbors(_maze, current):
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


def simulate_dfs(n, p):
    success = 0
    for i in range(n):
        maze = get_maze(p)
        if dfs(maze, (0, 0), (dim - 1, dim - 1)):
            success += 1
    return success/n

def plot_dfs():
    # y data
    p_success = []
    sims = 100
    p_success.append(simulate_dfs(sims, 0.1))
    p_success.append(simulate_dfs(sims, 0.125))
    p_success.append(simulate_dfs(sims, 0.150))
    p_success.append(simulate_dfs(sims, 0.175))
    p_success.append(simulate_dfs(sims, 0.2))
    p_success.append(simulate_dfs(sims, 0.225))
    p_success.append(simulate_dfs(sims, 0.250))
    p_success.append(simulate_dfs(sims, 0.275))
    p_success.append(simulate_dfs(sims, 0.3))
    p_success.append(simulate_dfs(sims, 0.325))
    p_success.append(simulate_dfs(sims, 0.350))
    p_success.append(simulate_dfs(sims, 0.375))
    p_success.append(simulate_dfs(sims, 0.4))
    p_success.append(simulate_dfs(sims, 0.425))
    p_success.append(simulate_dfs(sims, 0.450))
    p_success.append(simulate_dfs(sims, 0.475))
    p_success.append(simulate_dfs(sims, 0.5))
    
    # x axis
    x_axis = [0.1, 0.125, 0.150, 0.175, 
              0.2, 0.225, 0.250, 0.275, 
              0.3, 0.325, 0.350, 0.375, 
              0.4, 0.425, 0.450, 0.475,
              0.5]

    plt.title('DFS')
    plt.xlabel('Obstacle density p')
    plt.ylabel('Probability of success')
    plt.xlim([0.1, 0.5])
    plt.ylim([0.0, 1.0])
    plt.xticks(np.arange(0.1, 0.525, 0.025))
    plt.plot(x_axis, p_success)
    plt.show()

def simulate_bfs_astar(n, p):
    num_nodes_bfs = 0
    num_nodes_astar = 0
    for i in range(n):
        maze = get_maze(p)
        num_nodes_bfs += bfs(maze, (0, 0), (dim - 1, dim - 1))[1]
        num_nodes_astar += a_star(maze, (0, 0), (dim - 1, dim - 1))[1]
    # return avg
    return num_nodes_bfs/n, num_nodes_astar/n

def plot_bfs_astar():
    # y data
    avg_nodes = []
    avg_nodes_bfs = []
    avg_nodes_astar = []
    sims = 150
    avg_nodes.append(simulate_bfs_astar(sims, 0.1))
    avg_nodes.append(simulate_bfs_astar(sims, 0.125))
    avg_nodes.append(simulate_bfs_astar(sims, 0.150))
    avg_nodes.append(simulate_bfs_astar(sims, 0.175))
    avg_nodes.append(simulate_bfs_astar(sims, 0.2))
    avg_nodes.append(simulate_bfs_astar(sims, 0.225))
    avg_nodes.append(simulate_bfs_astar(sims, 0.250))
    avg_nodes.append(simulate_bfs_astar(sims, 0.275))
    avg_nodes.append(simulate_bfs_astar(sims, 0.3))
    avg_nodes.append(simulate_bfs_astar(sims, 0.325))
    avg_nodes.append(simulate_bfs_astar(sims, 0.350))
    avg_nodes.append(simulate_bfs_astar(sims, 0.375))
    avg_nodes.append(simulate_bfs_astar(sims, 0.4))
    avg_nodes.append(simulate_bfs_astar(sims, 0.425))
    avg_nodes.append(simulate_bfs_astar(sims, 0.450))
    avg_nodes.append(simulate_bfs_astar(sims, 0.475))
    avg_nodes.append(simulate_bfs_astar(sims, 0.5))

    avg_nodes_bfs = [i[0] for i in avg_nodes]
    avg_nodes_astar = [i[1] for i in avg_nodes]

    # x axis
    x_axis = [0.1, 0.125, 0.150, 0.175, 
              0.2, 0.225, 0.250, 0.275, 
              0.3, 0.325, 0.350, 0.375, 
              0.4, 0.425, 0.450, 0.475,
              0.5]

    plt.title('Number of nodes explored by BFS, A* vs Obstacle Density')
    plt.xlabel('Obstacle density p')
    plt.ylabel('Average Number Nodes')

    plt.xlim([0.1, 0.5])

    # set what ticks should be
    plt.xticks(np.arange(0.1, 0.525, 0.025))
    # plot bfs line
    plt.plot(x_axis, avg_nodes_bfs, label = "BFS")
    # plot a* line
    plt.plot(x_axis, avg_nodes_astar, label = "A*")
    plt.legend()
    plt.show()

def simulate_strats(n, p, q):
    success_1 = 0
    success_2 = 0
    success_3 = 0

    iterations = 0
    while iterations < n:
        maze = get_maze(p)
        fired = start_fire(maze)

        # if the maze is valid, then do strategies
        if maze_valid(maze, (0, 0), (dim - 1, dim - 1)):
            if bfs_1(maze, q):
                success_1 += 1
            iterations += 1
    print("\nSuccess 1")
    print(success_1)
    # iterations = 0
    # while iterations < n:
    #     maze = get_maze(p)
    #     fired = start_fire(maze)

    #     # if the maze is valid, then do strategies
    #     if maze_valid(maze, (0, 0), (dim - 1, dim - 1)):
    #         if bfs_2(maze, q):
    #             success_2 += 1
    #         iterations += 1
    # print("\nSuccess 2")
    # print(success_2)
    # iterations = 0
    # while iterations < n:
    #     maze = get_maze(p)
    #     fired = start_fire(maze)

    #     # if the maze is valid, then do strategies
    #     if maze_valid(maze, (0, 0), (dim - 1, dim - 1)):
    #         if bfs_3(maze, q):
    #             success_3 += 1
    #         iterations += 1
    # print("\nSuccess 3")
    # print(success_3)
    # return probability
    print(success_1/n)
    return success_1/n

def plot_strats():
    p_success =[]
    p_success_1 = []
    p_success_2 = []
    p_success_3 = []

    sims = 50
    p_success.append(simulate_strats(sims, 0.3, 0.1))
    p_success.append(simulate_strats(sims, 0.3, 0.15))
    p_success.append(simulate_strats(sims, 0.3, 0.2))
    p_success.append(simulate_strats(sims, 0.3, 0.25))
    p_success.append(simulate_strats(sims, 0.3, 0.3))
    p_success.append(simulate_strats(sims, 0.3, 0.35))
    p_success.append(simulate_strats(sims, 0.3, 0.4))
    p_success.append(simulate_strats(sims, 0.3, 0.45))
    p_success.append(simulate_strats(sims, 0.3, 0.5))
    p_success.append(simulate_strats(sims, 0.3, 0.55))
    p_success.append(simulate_strats(sims, 0.3, 0.6))
    p_success.append(simulate_strats(sims, 0.3, 0.65))
    p_success.append(simulate_strats(sims, 0.3, 0.7))
    p_success.append(simulate_strats(sims, 0.3, 0.75))
    p_success.append(simulate_strats(sims, 0.3, 0.8))
    p_success.append(simulate_strats(sims, 0.3, 0.85))
    p_success.append(simulate_strats(sims, 0.3, 0.9))

    # p_success_1 = [i[0] for i in p_success]
    # p_success_2 = [i[1] for i in p_success]
    # p_success_3 = [i[2] for i in p_success]

    avg_success_1 = (sum(p_success)/len(p_success)) * 100
    # avg_success_2 = (sum(p_success_2)/len(p_success_2)) * 100
    # avg_success_3 = (sum(p_success_3)/len(p_success_3)) * 100

    print(avg_success_1)

    # x axis
    x_axis = [0.1, 0.15, 
              0.2, 0.25, 
              0.3, 0.35, 
              0.4, 0.45, 
              0.5, 0.55,
              0.6, 0.65, 
              0.7, 0.75, 
              0.8, 0.85, 
              0.9]

    plt.title('Probability of success vs Flammability q')
    plt.xlabel('Flammability q')
    plt.ylabel('Probability of success')

    plt.xlim([0.1, 0.9])

    # set what ticks should be
    # plot strat 1 line
    plt.plot(x_axis, p_success, label = "Strategy 1")
    # # plot strat 2 line
    # plt.plot(x_axis, p_success_2, label = "Strategy 2")
    # # plot strat 3 line
    # plt.plot(x_axis, p_success_3, label = "Strategy 3")
    plt.legend()
    plt.show()
# maze = get_maze(0.3)
# start_time = time.time()
# a_star(maze, (0, 0), (dim - 1, dim - 1))
# print("%s seconds" % (time.time() - start_time))

# plot_bfs_astar()
# plot_dfs()

# plot_strats()

maze = get_maze(0.3)
start_time = time.time()
bfs_2(maze, 0.1)
print("%s seconds" % (time.time() - start_time))