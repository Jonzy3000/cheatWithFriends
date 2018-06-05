import sys
from imageProcessor import ImageProcessor
from gameSolver import GameSolver

def main():
    path = sys.argv[1]
    ip = ImageProcessor(path)
    scrabble_goard = ip.create_board()
    move, board = GameSolver(scrabble_goard, []).calculate_optimal_move()
    print "\n"
    board.pretty_print()

# imagepath = "images/image.jpg"
# imgage = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
# cv2.imshow("image", imgage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
