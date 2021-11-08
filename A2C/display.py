from main import World
import torch
from matplotlib import pyplot as plt
import time


def plot_step(dis, val, act, title):
    names = act + ["VALUE"]
    values = dis + [val]
    plt.title(title)
    plt.bar(names, values)
    plt.show()


def main():
    # number of episodes that I want to visualize
    num_episodes = 10
    actions = ["STOP", "UP", "DOWN", "LEFT", "RIGHT", "BOMB"]
    title = "P(a|s) and V(s)"
    world = World()  # TODO add as parameter the type of agent
    env = world.env
    model = world.model
    model.load_state_dict(torch.load("convrnn-s.weights", map_location='cpu'))
    ded, state, _ = False, world.env.reset(), world.leif.clear()

    for i in range(num_episodes):
        print("Episode {}".format(i+1))
        done = False
        j = 0
        while not done:
            j += 1
            print("Step {}".format(j))
            action, distribution, val = env.act(state)
            env.render()
            time.sleep(0.2)
            plot_step(distribution, val, actions, title)
            state, reward, done, info = env.step(action)
main()