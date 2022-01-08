from pommerman import constants
from pommerman.constants import Item


def get_dangerous_positions(board, blast_strength, position):
    # positions affected by the blast strength
    left = [(position[0], position[1] - j) for j in range(1, blast_strength + 1)]
    right = [(position[0], position[1] + j) for j in range(1, blast_strength + 1)]
    up = [(position[0] - j, position[1]) for j in range(1, blast_strength + 1)]
    down = [(position[0] + j, position[1]) for j in range(1, blast_strength + 1)]
    positions = left + right + up + down
    # removing positions outside the board
    for pos in positions:
        if pos[0] < 0 or pos[1] < 0 or pos[0] > constants.BOARD_SIZE - 1 or \
                pos[1] > constants.BOARD_SIZE - 1:
            positions.remove(pos)
    # removing positions not affected by the flames because there is a wooden or destructible walls
    # left direction
    left = [pos for pos in positions if pos[1] < position[1]]
    for elem in left:
        # checking whether the element on the position is a rigid wall or a wooden wall
        if board[elem[0], elem[1]] == Item.Rigid.value or board[elem[0], elem[1]] == Item.Wood.value:
            remove_list = [pos for pos in left if pos[1] < elem[1]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                left.remove(item)
    # right direction
    right = [pos for pos in positions if pos[1] > position[1]]
    for elem in right:
        # checking whether the element on the position is a rigid wall
        if board[elem[0], elem[1]] == Item.Rigid.value or board[elem[0], elem[1]] == Item.Wood.value:
            remove_list = [pos for pos in right if pos[1] > elem[1]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                right.remove(item)
    # up direction
    up = [pos for pos in positions if pos[0] < position[0]]
    for elem in up:
        # checking whether the element on the position is a rigid wall
        if board[elem[0], elem[1]] == Item.Rigid.value or board[elem[0], elem[1]] == Item.Wood.value:
            remove_list = [pos for pos in up if pos[0] < elem[0]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                up.remove(item)
    # down direction
    down = [pos for pos in positions if pos[0] > position[0]]
    for elem in down:
        # checking whether the element on the position is a rigid wall
        if board[elem[0], elem[1]] == 1 or board[elem[0], elem[1]] == 2:
            remove_list = [pos for pos in down if pos[0] > elem[0]]
            # removing positions beyond the rigid wall because they cannot be reached by the flames
            for item in remove_list:
                down.remove(item)
    # adding positions to the main list
    dang_pos = left + right + up + down

    return dang_pos