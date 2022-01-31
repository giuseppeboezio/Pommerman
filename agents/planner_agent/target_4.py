from pommerman.constants import Item
from pommerman import constants


def count_power_ups(board1, board2):
    items = {Item.ExtraBomb.value, Item.IncrRange.value, Item.Kick.value}
    pos_board1 = set()
    pos_board2 = set()
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if board1[i,j] in items:
                pos_board1.add((i,j))
            if board2[i,j] in items:
                pos_board2.add((i,j))
    # difference between the second set and the first set to add only positions where new power-ups have appeared
    new_pos_pow_ups = pos_board2 - pos_board1
    count = len(new_pos_pow_ups)
    return count
