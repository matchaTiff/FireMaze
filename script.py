import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np

import random
import sys
import time
import collections

dim = 20

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

    while fringe:

        # get the first element from queue
        current = fringe.popleft()

        if current == goal:
            print('\nSUCCESS')
            return True

        else:
            neighbors = get_valid_neighbors(_maze, current, visited)
            visited.update(neighbors)
            fringe.extend(neighbors)

    print('\nFAILED')
    return False

def get_distance(start, goal):
    """
    Returns the distance between two spots on a grid.
    :param start:
    :param goal:
    :return:
    """
    return (goal[0] - start[0]) + (goal[1] - start[1])

def get_distance_from_goal(pos):
    return get_distance(pos, (dim - 1, dim - 1))


class Node:
    def __init__(self, coords, parent, distance, weight):
        self.coords = coords
        self.parent = parent
        self.distance = distance
        self.weight = weight


def a_star(_maze, start, goal):
    fringe = collections.deque([Node(start, None, 0, get_distance_from_goal(start))])
    visited = set()
    while len(fringe) > 0:

        sorted(fringe, key=lambda x: x.weight)  # Sort by weight
        current = fringe.popleft()

        if current.coords == goal and (len(fringe) == 0 or fringe[0].weight <= current.weight):
            print('\nSUCCESS')
            return True
        else:
            neighbors = set()
            row = current.coords[0]
            col = current.coords[1]

            # left
            if row > 0 and (row - 1, col) not in visited and _maze[row - 1][col] != 1 and _maze[row - 1][col] != 2:
                left_n = (row - 1, col)
                neighbors.add(left_n)
                node = Node(left_n, current, current.distance + 1, current.distance + 1 + get_distance_from_goal((left_n[0], left_n[1])))
                fringe.append(node)
            # right
            if row + 1 < dim and (row + 1, col) not in visited and _maze[row + 1][col] != 1 and _maze[row + 1][col] != 2:
                right_n = (row + 1, col)
                neighbors.add(right_n)
                node = Node(right_n, current, current.distance + 1, current.distance + 1 + get_distance_from_goal((right_n[0], right_n[1])))
                fringe.append(node)
            # down
            if col > 0 and (row, col - 1) not in visited and _maze[row][col - 1] != 1 and _maze[row][col - 1] != 2:
                down_n = (row, col - 1)
                neighbors.add(down_n)
                node = Node(down_n, current, current.distance + 1, current.distance + 1 + get_distance_from_goal((down_n[0], down_n[1])))
                fringe.append(node)
            # up
            if col + 1 < dim and (row, col + 1) not in visited and _maze[row][col + 1] != 1 and _maze[row][col + 1] != 2:
                up_n = (row, col + 1)
                neighbors.add(up_n)
                node = Node(up_n, current, current.distance + 1, current.distance + 1 + get_distance_from_goal((up_n[0], up_n[1])))
                fringe.append(node)

            visited.add(current.coords)

    print('\nFAILED')
    return False


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

def simulate_bfs(n, p):
    num_nodes = 0
    for i in range(n):
        maze = get_maze(p)
        if bfs(maze, (0, 0), (dim - 1, dim - 1)):
            num_nodes += incoming
    # return avg
    return num_nodes/n

def plot_bfs_astar():
    # y data
    avg_nodes_bfs = []
    sims = 100
    avg_nodes_bfs.append(simulate_bfs(sims, 0.1))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.125))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.150))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.175))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.2))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.225))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.250))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.275))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.3))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.325))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.350))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.375))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.4))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.425))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.450))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.475))
    avg_nodes_bfs.append(simulate_bfs(sims, 0.5))

    avg_nodes_astar = []

    
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

    # set what ticks should be
    plt.xticks(np.arange(0.1, 0.525, 0.025))
    # plot bfs line
    plt.plot(x_axis, avg_nodes_bfs, label = "bfs")
    # plot a* line
    plt.plot(x_axis, avg_nodes_astar, label = "A*")
    plt.show()

maze = get_maze(0.3)
start_time = time.time()
a_star(maze, (0, 0), (dim - 1, dim - 1))
print("%s seconds" % (time.time() - start_time))

# plot_dfs()