from main import World
import torch
from matplotlib import pyplot as plt
import time
from gym import spaces


def plot_step(dis, val, act, title):
    names = act
    values = dis
    plt.title(title)
    plt.bar(names, values)
    plt.show()


def main():
    # number of episodes that I want to visualize
    num_episodes = 10
    name_acts = ["STOP", "UP", "DOWN", "LEFT", "RIGHT", "BOMB"]
    title = "P(a|s) and V(s)"
    world = World(display=True)
    env = world.env
    model = world.model
    model.load_state_dict(torch.load("convrnn-s.weights", map_location='cpu'))
    ded, state, _ = False, world.env.reset(), world.leif.clear()
    # our player is the first agent
    obs = state[0]

    for i in range(num_episodes):
        print("Episode {}".format(i+1))
        done = False
        j = 0
        while not done:
            j += 1
            print("Step {}".format(j))
            action, distribution, val = world.leif.act(obs, spaces.Discrete(len(name_acts)))

            # preprocessing of distribution and val
            dist = distribution.tolist()
            value = val.tolist()[0][0]

            env.render()
            time.sleep(2)
            plot_step(dist, value, name_acts, title)
            state, reward, done, info = env.step(action)
main()