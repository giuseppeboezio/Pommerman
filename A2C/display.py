from main import World
from model import DisplayAgent
import torch
from matplotlib import pyplot as plt
import time
from gym import spaces
from pommerman import constants
from collections import defaultdict
import queue
import random
import numpy as np
from pommerman import utility


def plot_step(dis, val, act):
    names = act
    values = dis
    plt.subplot(211)
    plt.title("P(a|s)")
    plt.bar(names, values)
    plt.subplot(212)
    plt.plot(["V(s)"], [val], 'r+')
    plt.show()


def convert_bombs(bomb_map):
    '''Flatten outs the bomb array'''
    ret = []
    locations = np.where(bomb_map > 0)
    for r, c in zip(locations[0], locations[1]):
        ret.append({
            'position': (r, c),
            'blast_strength': int(bomb_map[(r, c)])
        })
    return ret


def dijkstra(board, my_position, bombs, enemies, depth=None, exclude=None):
    assert (depth is not None)

    if exclude is None:
        exclude = [
            constants.Item.Fog, constants.Item.Rigid, constants.Item.Flames
        ]

    def out_of_range(p_1, p_2):
        '''Determines if two points are out of rang of each other'''
        x_1, y_1 = p_1
        x_2, y_2 = p_2
        return abs(y_2 - y_1) + abs(x_2 - x_1) > depth

    items = defaultdict(list)
    dist = {}
    prev = {}
    Q = queue.Queue()

    my_x, my_y = my_position
    for r in range(max(0, my_x - depth), min(len(board), my_x + depth)):
        for c in range(max(0, my_y - depth), min(len(board), my_y + depth)):
            position = (r, c)
            if any([
                out_of_range(my_position, position),
                utility.position_in_items(board, position, exclude),
            ]):
                continue

            prev[position] = None
            item = constants.Item(board[position])
            items[item].append(position)

            if position == my_position:
                Q.put(position)
                dist[position] = 0
            else:
                dist[position] = np.inf

    for bomb in bombs:
        if bomb['position'] == my_position:
            items[constants.Item.Bomb].append(my_position)

    while not Q.empty():
        position = Q.get()

        if utility.position_is_passable(board, position, enemies):
            x, y = position
            val = dist[(x, y)] + 1
            for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (row + x, col + y)
                if new_position not in dist:
                    continue

                if val < dist[new_position]:
                    dist[new_position] = val
                    prev[new_position] = position
                    Q.put(new_position)
                elif (val == dist[new_position] and random.random() < .5):
                    dist[new_position] = val
                    prev[new_position] = position

    return items, dist, prev

def main():
    # number of episodes that I want to visualize
    num_episodes = 10
    name_acts = ["Stop", "Up", "Down", "Left", "Right", "Bomb"]
    world = World()
    env = world.env
    model = world.model
    model.load_state_dict(torch.load("convrnn-s.weights", map_location='cpu'))
    display_agent = DisplayAgent(model)

    for i in range(num_episodes):

        display_agent.clear()
        ded, state, _ = False, env.reset(), world.leif.clear()
        print("Episode {}".format(i+1))
        done = False
        j = 0
        while not done:
            print("Step {}".format(j))
            actions = env.act(state)

            # showing behaviour of our agent (first agent in the list)
            obs = state[0]

            # code of Simple Agent
            my_position = tuple(obs['position'])
            board = np.array(obs['board'])
            print(board)
            bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
            enemies = [constants.Item(e) for e in obs['enemies']]
            print("Item \t Distance \t Predecessor")
            items, dist, prev = dijkstra(
                board, my_position, bombs, enemies, depth=10)
            for it, d, p in zip(items, dist, prev):
                print(f'{it} \t {d} \t {p}')

            # action, distribution, val = display_agent.act(obs, spaces.Discrete(len(name_acts)))

            # preprocessing of distribution and val
            # dist = distribution.tolist()[:][0]
            # value = val.tolist()[0][0]

            env.render()
            time.sleep(0.5)
            # plot_step(dist, value, name_acts)
            state, reward, done, info = env.step(actions)
            j += 1


main()