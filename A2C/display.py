from main import World
from model import DisplayAgent
import torch
from matplotlib import pyplot as plt
import time
from heuristic_modules import module_1 as m1


def plot_step(dis, val, act):
    names = act
    values = dis
    plt.subplot(211)
    plt.title("P(a|s)")
    plt.bar(names, values)
    plt.subplot(212)
    plt.plot(["V(s)"], [val], 'r+')
    plt.show()


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

            print("Board")
            print(obs['board'])

            print("Blast strength")
            print(obs['bomb_life'])



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