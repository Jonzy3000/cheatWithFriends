import numpy as np
from tile import Tile


class ScrabbleGrid:
    rows = 11
    cols = 11
    grid = []
    triple_letters_coords = [[0, 0], [0, 10], [2, 2], [2, 8], [3, 7], [
        3, 3], [7, 3], [7, 7], [8, 2], [8, 8], [10, 0], [10, 10]]

    triple_word_coords = [[0, 2], [0, 8], [2, 0], [
        2, 10], [8, 0], [8, 10], [10, 2], [10, 8]]

    double_word_coords = [[1, 1], [1, 5], [1, 9], [
        5, 1], [5, 5], [5, 9], [9, 1], [9, 5], [9, 9]]

    double_letter_coords = [[3, 4], [3, 6], [4, 2],
                            [4, 8], [6, 2], [6, 8], [8, 4], [8, 6]]

    def __init__(self):
        self.create_grid()

    def empty_grid(self):
        empty_tile = [None] * 121
        empty_grid = np.reshape(empty_tile, (self.rows, self.cols))
        return empty_grid

    def create_grid(self):
        self.grid = self.empty_grid()
        self.set_special_tiles(
            self.triple_letters_coords, Tile.TRIPLE_LETTER)
        self.set_special_tiles(self.triple_word_coords, Tile.TRIPLE_WORD)
        self.set_special_tiles(self.double_letter_coords, Tile.DOUBLE_LETTER)
        self.set_special_tiles(self.double_word_coords, Tile.DOUBLE_WORD)
        for (x,y), letter in np.ndenumerate(self.grid):
            if letter is None:
                self.grid[x][y] = Tile()

        self.pretty_print()

    def set_special_tiles(self, coords, init_fn):
        for xy in coords:
            self.grid[xy[0]][xy[1]] = init_fn()

    def get_grid(self):
        return self.grid

    def pretty_print(self):
        matrix = self.grid
        s = [[e.to_string() for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print '\n'.join(table)


    def add_letters(self, letter_grid):
        for (x,y), letter in np.ndenumerate(letter_grid):
            if len(letter) > 1 or letter == "0":
                continue
            self.grid[x][y].set_letter(letter)

        self.pretty_print()

            
