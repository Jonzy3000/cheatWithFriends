

class Tile:
    EMPTY = "0"

    def __init__(self):
        self.letter = self.EMPTY
        self.special_letter = self.EMPTY
        self.letter_score = 0
        self.word_score_fn = None
        self.letter_score_fn = None

    @classmethod
    def TRIPLE_WORD(self):
        def triple_word_score(word_score):
            return word_score * 3

        tile = Tile()
        tile.word_score_fn = triple_word_score
        tile.special_letter = "TW"
        return tile

    @classmethod
    def DOUBLE_WORD(self):
        def double_word_score(word_score):
            return word_score * 2

        tile = Tile()
        tile.word_score_fn = double_word_score
        tile.special_letter = "DW"
        return tile

    @classmethod
    def TRIPLE_LETTER(self):
        def triple_letter_score(self):
            return self.letter_score * 3

        tile = Tile()
        tile.letter_score_fn = triple_letter_score
        tile.special_letter = "TL"
        return tile

    @classmethod
    def DOUBLE_LETTER(self):
        def double_letter_score(self):
            return self.letter * 3

        tile = Tile()
        tile.letter_score_fn = double_letter_score
        tile.special_letter = "DL"
        return tile

    def to_string(self):
        return self.special_letter if self.letter is self.EMPTY else self.letter

    def set_letter(self, letter):
        self.letter = letter

    def get_letter(self):
        return self.letter
