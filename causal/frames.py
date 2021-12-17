import numpy as np
from pommerman import constants, make, agents
import pandas as pd
import os

# class to manage the bomb


class Bomb:

    def __init__(self, pos_x, pos_y, blast):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.blast = blast
        self.life_time = 10

    def decrease_life(self):
        self.life_time -= 1

    def is_exploded(self):
        return self.life_time <= 0

    def get_blast(self):
        return self.blast

    def get_pos_x(self):
        return self.pos_x

    def get_pos_y(self):
        return self.pos_y

    def get_life(self):
        return self.life_time


def get_timestep_patch(board, bomb):
    """Patch obtained by the board with the bomb"""

    blast_strength = int(bomb.get_blast())
    pos_x = bomb.get_pos_x()
    pos_y = bomb.get_pos_y()

    right_bound = pos_y + blast_strength - 1
    left_bound = pos_y - (blast_strength - 1)
    up_bound = pos_x - (blast_strength - 1)
    low_bound = pos_x + blast_strength - 1

    # consider cases in which the patch is perfectly centered
    if right_bound < 11 and low_bound < 11 and left_bound >= 0 and up_bound >= 0:

        patch = np.array(board[up_bound:low_bound+1,left_bound:right_bound+1])

    # there are different cases of cropping the board
    # outside on the left
    elif left_bound < 0 and right_bound < 11 and low_bound < 11 and up_bound >= 0:

        patch = np.array(board[up_bound:low_bound+1,0:right_bound+1])

    # outside on the right
    elif right_bound >= 11 > low_bound and left_bound >= 0 and up_bound >= 0:

        patch = np.array(board[up_bound:low_bound+1, left_bound:constants.BOARD_SIZE])

    # outside on the top
    elif up_bound < 0 and right_bound < 11 and low_bound < 11 and left_bound >= 0:

        patch = np.array(board[0:low_bound+1,left_bound:right_bound+1])

    # outside on the bottom
    elif low_bound >= 11 and right_bound < 11 and left_bound >= 0 and up_bound >= 0:

        patch = np.array(board[up_bound:constants.BOARD_SIZE, left_bound:right_bound+1])

    # outside in the upper-left corner
    elif up_bound < 0 and left_bound < 0 and low_bound < 11 and right_bound < 11:

        patch = np.array(board[0:low_bound+1, 0:right_bound+1])

    # outside in the upper-right corner
    elif up_bound < 0 and right_bound >= 11 and low_bound < 11 and left_bound >= 0:

        patch = np.array(board[0:low_bound+1, left_bound:constants.BOARD_SIZE])

    # outside in the lower-left corner
    elif low_bound >= 11 and left_bound < 0 and up_bound >= 0 and right_bound < 11:

        patch = np.array(board[up_bound:constants.BOARD_SIZE, 0:right_bound+1])

    # outside in the lower-right corner
    else:
        patch = np.array(board[up_bound:constants.BOARD_SIZE, left_bound:constants.BOARD_SIZE])

    return patch


def bomb_on_board(board):

    """Find the position of a bomb on the board, if there is any"""

    pos_x = -1
    pos_y = -1
    found = False

    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if board[i,j] == 9:
                pos_x = i
                pos_y = j
                found = True
                break

    return found, pos_x, pos_y


def generate_point(patch):
    """patch is a numpy array having shape[2] == 2"""

    rows = patch.shape[0]
    cols = patch.shape[1]

    # initializing the vector which represents the patch
    # the structure is the following: [w,h, time_step1, time_step_2 (histograms of values)]
    num_board_val = 14
    len_and_height = 2

    point = np.zeros(num_board_val * 2 + len_and_height)

    # loading height and width
    point[0] = cols
    point[1] = rows

    # updating the histogram of the first and second timestep
    for i in range(rows):
        for j in range(cols):
            point[patch[i,j,0] + 2] += 1
            point[patch[i,j,1] + 16] += 1

    return point


def generate_patches():

    config = "PommeFFACompetition-v0"
    game_state_file = None

    my_agents = [agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()]

    env = make(config, my_agents, game_state_file)

    # number of episodes to run to collect patches
    num_ep = 1000

    # names of each dimension of the vector
    names = ['w', 'h']
    ts_1 = [f"{i}_ts1" for i in range(14)]
    ts_2 = [f"{i}_ts2" for i in range(14)]
    names = names + ts_1 + ts_2

    # number of sequences of patches obtained from the same bomb
    num = 0

    # path to save csv files
    path = "C:/Users/boezi/PycharmProjects/Pommerman/causal/patches"

    for ep in range(num_ep):

        print(f"Episode {ep}")

        state = env.reset()

        # flag to understand whether the bomb to use for the patch has been found or not
        bomb_found = False
        bomb = None

        # first_channel and second channel of the patch
        first_ch = None

        # list of patches
        list_patches = []

        # flag for episode termination
        done = False

        while not done:

            obs = state[0]

            # the bomb has not been found
            if not bomb_found:

                life_bombs = obs['bomb_life']
                bomb_found, pos_x, pos_y = bomb_on_board(life_bombs)

                if bomb_found:

                    blast = obs['bomb_blast_strength'][pos_x, pos_y]
                    bomb = Bomb(pos_x, pos_y, blast)
                    first_ch = get_timestep_patch(obs['board'], bomb)

                    # decrease life of the bomb
                    bomb.decrease_life()

            # there is a bomb to use for the patches
            else:
                # the bomb has not exploded
                if bomb.get_life() >= 0:

                    second_ch = get_timestep_patch(obs['board'], bomb)
                    # creation of the patch
                    patch = np.stack((first_ch, second_ch))
                    # generation of the vector
                    point = generate_point(patch)
                    # adding the point to the list
                    list_patches.append(point)

                    first_ch = np.array(second_ch)
                    bomb.decrease_life()

                # looking for another bomb
                else:
                    num += 1
                    # conversion of the list in a 2d numpy array
                    list_patches = np.array(list_patches)
                    # creation of the dataframe
                    df = pd.DataFrame(list_patches, columns=names)
                    # save data as csv file
                    df.to_csv(os.path.join(path,f"{num}.csv"), index=False)
                    # flush the list of patches
                    list_patches = []
                    # change the flag to look for another bomb
                    bomb_found = False

            # action of each agent
            agent_actions = env.act(state)
            # new state after the actions
            state, reward, done, info = env.step(agent_actions)

        '''# consider the situation in which the game ends but there are some not stored patches
        if len(list_patches) > 0:

            num += 1

            # conversion of the list in a 2d numpy array
            list_patches = np.array(list_patches)
            # creation of the dataframe
            df = pd.DataFrame(list_patches, columns=names)
            # save data as csv file
            df.to_csv(os.path.join(path, f"{num}.csv"), index=False)
        '''


def main():

    generate_patches()


if __name__ == '__main__':
    main()