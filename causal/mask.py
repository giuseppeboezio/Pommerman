from pommerman import constants, make, agents
import numpy as np


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

            obs = state[0]

            # at the initial step there is no variation of values on the board
            if num_step == 0:

                mask = np.zeros((constants.BOARD_SIZE,constants.BOARD_SIZE))
                comparison = np.concatenate((obs['board'],mask), axis=1)
                print(comparison)

            else:

                mask = not obs['board'] == old_board
                comparison = np.concatenate((obs['board'], mask), axis=1)

            old_board = np.array(obs['board'])
            # action of each agent
            agent_actions = env.act(state)
            # new state after the actions
            state, reward, done, info = env.step(agent_actions)
            num_step = num_step + 1
