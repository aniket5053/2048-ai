import random
import constants as c

def new_game(n):
    matrix = [[0] * n for _ in range(n)]
    matrix = add_two(matrix)
    matrix = add_two(matrix)
    return matrix

def add_two(mat):
    a, b = random.randint(0, len(mat)-1), random.randint(0, len(mat)-1)
    while mat[a][b] != 0:
        a, b = random.randint(0, len(mat)-1), random.randint(0, len(mat)-1)
    mat[a][b] = 2
    return mat

def game_state(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 2048:
                return 'win'
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                return 'not over'
    for i in range(len(mat)-1):
        for j in range(len(mat[0])-1):
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return 'not over'
    for k in range(len(mat)-1):
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1):
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'

def reverse(mat):
    return [row[::-1] for row in mat]

def transpose(mat):
    return [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]

def cover_up(mat):
    new = [[0] * c.GRID_LEN for _ in range(c.GRID_LEN)]
    done = False
    for i in range(c.GRID_LEN):
        count = 0
        for j in range(c.GRID_LEN):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return new, done

def merge(mat, done):
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN-1):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j+1] = 0
                done = True
    return mat, done

def up(game):
    game = transpose(game)
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(game)
    return game, done

def down(game):
    game = reverse(transpose(game))
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(reverse(game))
    return game, done

def left(game):
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    return game, done

def right(game):
    game = reverse(game)
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = reverse(game)
    return game, done

def perform_action(matrix, action):
    if action == 'Up':
        return up(matrix)
    elif action == 'Down':
        return down(matrix)
    elif action == 'Left':
        return left(matrix)
    elif action == 'Right':
        return right(matrix)
    else:
        raise ValueError("Invalid action")