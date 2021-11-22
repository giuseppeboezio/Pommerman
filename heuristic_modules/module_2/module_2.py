import numpy as np
from pommerman import constants, make, agents
from pommerman.constants import Item


def preprocess_state_avoidance(state, position):
    """Create the input for the NN
    position: tuple containing the position to evaluate
    passage: 0
    bombs: 3
    my agent's position: 10
    marked position: 14
    other objects: 15"""

    board = state['board']
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
    while num_pos < num:

        state = env.reset
        # flag for episode termination
        done = False

        while not done:
            # bomb positions
            last_bomb_positions = get_object_positions(state[0], Item.Bomb)
            last_agent_position = get_object_positions(state[0], Item.Agent0)
            # action of each agent
            agent_actions = env.act(state)
            # optional rending
            env.render()
            # new state after the actions
            state, reward, done, _ = env.step(agent_actions)
            cur_bomb_positions = get_object_positions(state[0], Item.Bomb)
            cur_agent_position = get_object_positions(state[0], Item.Agent0)

            # my agent has been killed
            if not cur_agent_position:

                killing_bomb = set(last_bomb_positions) - set(cur_bomb_positions)
                for elem in killing_bomb:
                    k_bomb_pos = elem
                # now I have the position of the agent in last_agent_position and the position of the bomb
                # in k_bomb_pos








    pass

