from fire import bfs_1, bfs_2
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np

import random
import sys
import time
import collections
import numpy as np
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

def simulate_strats():
    bfs_1
    bfs_2
    bfs_3
# maze = get_maze(0.3)
# start_time = time.time()
# a_star(maze, (0, 0), (dim - 1, dim - 1))
# print("%s seconds" % (time.time() - start_time))



plot_bfs_astar()

# plot_dfs()