from pommerman import constants, make, agents
import numpy as np
import colorama


def get_mask(board_1, board_2):
    """Produce a mask of the changes among states"""

    rows = board_1.shape[0]
    cols = board_1.shape[1]
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if board_1[i,j] != board_2[i,j]:
                mask[i,j] = 1
    return mask


def show_board_and_mask():

    config = "PommeFFACompetition-v0"
    game_state_file = None
    my_agents = [agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()]
    env = make(config, my_agents, game_state_file)

    num_ep = 10

    for ep in range(num_ep):

        print(f"Episode {ep+1}")

        state = env.reset()
        done = False
        num_step = 0
        old_board = np.empty((constants.BOARD_SIZE,constants.BOARD_SIZE))

        while not done:

            print(f"Step: {num_step}")

            obs = state[0]

            # at the initial step there is no variation of values on the board
            if num_step == 0:

                mask = np.zeros((constants.BOARD_SIZE,constants.BOARD_SIZE))
                comparison = np.concatenate((obs['board'],mask), axis=1)
                show_comparison(comparison)

            else:

                mask = get_mask(obs['board'], old_board)
                comparison = np.concatenate((obs['board'], mask), axis=1)
                show_comparison(comparison)

            old_board = np.array(obs['board'])
            # action of each agent
            agent_actions = env.act(state)
            # new state after the actions
            state, reward, done, info = env.step(agent_actions)
            num_step = num_step + 1


if __name__ == '__main__':
    show_board_and_mask()
