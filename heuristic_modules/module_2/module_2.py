import numpy as np
from pommerman import constants, make, agents
from pommerman.constants import Item
from heuristic_modules.collection_samples import convert_to_images
from heuristic_modules.network import train_and_test
from sklearn.model_selection import train_test_split
import time


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
            if state['board'][i,j] == id_object:
                positions.append((i,j))
    return positions


def get_samples(state, position):

    samples = []
    board = state['board']
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            # clean the position where the agent can die and move it to any feasible position
            if board[i,j] == Item.Passage.value:
                board[i,j] = Item.Agent0.value
                board[position[0],position[1]] = Item.Passage.value
                # preprocess the input for the NN
                input_board = preprocess_state_avoidance(board, position)
                samples.append(input_board)
    return samples


def is_alive(state, agent_id):

    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if state['board'][i,j] == agent_id:
                return True
    return False


def collect_data(num):
    """Create dataset to use for the NN
    num is the number of samples in each class"""

    config = "PommeFFACompetition-v0"
    game_state_file = None

    my_agents = [agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()]

    env = make(config, my_agents, game_state_file)

    # number of data collected for each class
    num_pos = 0
    num_neg = 0
    pos_dataset = []
    neg_dataset = []

    num_episode = 1

    # collecting negative samples
    # In this case if my agent dies, I consider the last position (where the agent should not be) and
    # produce samples considering all positions where to put the agent except for the one where it could be killed
    print("Collecting negative samples\n---------------------------")

    while num_neg < num:

        num_step = 0

        state = env.reset()
        # flag for episode termination
        done = False

        while not done:

            last_state = state[0]
            last_agent_position = last_state['position']

            # action of each agent
            agent_actions = env.act(state)

            # optional rending
            # env.render()
            # time.sleep(0.5)
            # new state after the actions
            state, reward, done, info = env.step(agent_actions)
            new_state = state[0]

            # my agent has been killed
            if not is_alive(new_state, Item.Agent0.value):

                samples = get_samples(last_state, last_agent_position)
                neg_dataset += samples
                num_neg += len(samples)

                print(f"Number of negative collected samples: {num_neg} / {num} - {num_neg/num} %")

                break

            num_step += 1

        num_episode += 1

    add_sample_num = num_neg % num
    neg_dataset = neg_dataset[:-add_sample_num]
    print(f"Number of negative samples: {len(neg_dataset)}")

    print("Collecting positive samples\n------------------------")

    while num_pos < num:

        state = env.reset()
        # flag for episode termination
        done = False

        while not done:
            last_state = state[0]
            last_agent_position = last_state['position']
            # action of each agent
            agent_actions = env.act(state)
            # optional rending
            # env.render()
            # time.sleep(0.5)
            # new state after the actions
            state, reward, done, _ = env.step(agent_actions)
            new_state = state[0]

            # my agent has not been killed
            if is_alive(new_state, Item.Agent0.value):

                samples = get_samples(last_state, last_agent_position)
                pos_dataset += samples
                num_pos += len(samples)

                print(f"Number of positive collected samples: {num_pos} / {num} - {num_pos/num} %")

                break

    add_sample_num = num_pos % num
    pos_dataset = pos_dataset[:-add_sample_num]
    print(f"Number of positive samples: {len(neg_dataset)}")

    # creating the dataset
    neg_labels = np.full(num, 0)
    pos_labels = np.full(num, 1)
    labels = np.concatenate((neg_labels,pos_labels))

    input_list = neg_dataset + pos_dataset

    return input_list, labels


def main():

    # number of samples to collect for each class
    num = 200000

    input_list, labels = collect_data(num)

    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(input_list, labels, test_size=0.33, random_state=42, stratify=labels)

    # convert boards into images
    convert_to_images(X_train,y_train,"C:/Users/boezi/PycharmProjects/Pommerman/heuristic_modules/module_2/train", "C:/Users/boezi/PycharmProjects/Pommerman/heuristic_modules/module_2/train.csv")
    convert_to_images(X_test, y_test, "C:/Users/boezi/PycharmProjects/Pommerman/heuristic_modules/module_2/test", "C:/Users/boezi/PycharmProjects/Pommerman/heuristic_modules/module_2/test.csv")


def launch_training_and_test():

    train_csv = "train.csv"
    test_csv = "test.csv"
    dir_train = "train"
    dir_test = "test"
    model_path = "model_weights.pth"

    train_and_test(train_csv, test_csv, dir_train, dir_test, model_path)


if __name__ == '__main__':
    launch_training_and_test()
    # main()