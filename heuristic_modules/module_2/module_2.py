import numpy as np
from pommerman import constants, make, agents
from pommerman.constants import Item
import random
from sklearn.model_selection import train_test_split


def preprocess_state_avoidance(board, position):
    """Create the input for the NN
    position: tuple containing the position to evaluate
    passage: 0
    bombs: 3
    my agent's position: 10
    marked position: 14
    other objects: 15"""

    input_state = np.array(board)
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            # not relevant objects
            if 0 < input_state[i, j] < 3 or 3 < input_state[i, j] < 10 or input_state[i, j] > 10:
                input_state[i, j] = 15
            # marked position
            if i == position[0] and j == position[1]:
                input_state[i, j] = 14
    return input_state


def get_object_positions(state, id_object):
    """Return positions (tuple) of the objects with id_object"""

    positions = []
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            # there is a bomb in position (i,j)
            if state['board'] == id_object:
                positions.append((i,j))
    return positions


def get_samples(state, position):

    samples = []
    board = state['board']
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            # clean the position where the agent can die and move it to any feasible position
            if board[i,j] == Item.Passage:
                board[i,j] = Item.Agent0
                board[position[0],position[1]] = Item.Passage
                # preprocess the input for the NN
                input_board = preprocess_state_avoidance(board, position)
                samples.append(input_board)
    return samples


def collect_data(num):
    """Create dataset to use for the NN
    num is the number of samples in each class"""

    config = "PommeFFACompetition-v0"
    game_state_file = None

    my_agents = [agents.RandomAgent(), agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()]

    env = make(config, my_agents, game_state_file)

    # number of data collected for each class
    num_pos = 0
    num_neg = 0
    pos_dataset = []
    neg_dataset = []

    state_history = []

    # collecting positive samples
    # In this case if my agent dies, I consider the last position (where the agent should not be) and
    # produce samples considering all positions where to put the agent except for the one where it could be killed
    print("Collecting negative samples\n----------------------------------------")

    while num_neg < num:

        state = env.reset
        # flag for episode termination
        done = False

        while not done:
            last_state = state[0]
            last_agent_position = get_object_positions(last_state, Item.Agent0)
            # action of each agent
            agent_actions = env.act(state)
            # optional rending
            env.render()
            # new state after the actions
            state, reward, done, _ = env.step(agent_actions)
            new_state = state[0]
            cur_agent_position = get_object_positions(new_state, Item.Agent0)

            # my agent has been killed
            if not cur_agent_position:

                samples = get_samples(last_state, last_agent_position[0])
                neg_dataset += samples
                num_neg += len(samples)

                print(f"Number of negative collected samples: {num_neg} / {num} - {num_neg/num} %")

                break

    print("Collecting positive samples\n----------------------------------------")

    while num_pos < num:

        state = env.reset
        # flag for episode termination
        done = False

        while not done:
            last_state = state[0]
            last_agent_position = get_object_positions(last_state, Item.Agent0)
            # action of each agent
            agent_actions = env.act(state)
            # optional rending
            env.render()
            # new state after the actions
            state, reward, done, _ = env.step(agent_actions)
            new_state = state[0]
            cur_agent_position = get_object_positions(new_state, Item.Agent0)

            # my agent has not been killed
            if cur_agent_position:

                samples = get_samples(last_state, last_agent_position[0])
                pos_dataset += samples
                num_pos += len(samples)

                print(f"Number of positive collected samples: {num_pos} / {num} - {num_pos/num} %")

                break

    # creating the dataset
    neg_labels = np.full(num, 0)
    pos_labels = np.full(num, 1)
    labels = np.concatenate((neg_labels,pos_labels))

    input_list = neg_dataset + pos_dataset

    return input_list, labels


def main():

    # number of samples to collect
    num = 2000

    input_list, labels = collect_data(num)


main()