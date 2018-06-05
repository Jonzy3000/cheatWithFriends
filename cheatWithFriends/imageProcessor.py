import cv2
import numpy as np
from matplotlib import pyplot as plt
from tile import Tile
from letterDetection import LetterDetection
from scrabbleGrid import ScrabbleGrid


class ImageProcessor:
    rows = 11
    cols = 11

    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(self.path)
        self.scrabble_grid = ScrabbleGrid()

    def create_empty_tile_grid(self):
        empty_tile = [Tile()]
        test = np.tile(empty_tile, (self.rows, self.cols))
        return test

    def create_grid_of_boxes(self):
        img = self.image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 3, 2)

        cv2.imshow("thresh", thresh)

        _, contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

        # kernel = np.ones((5,5),np.uint8)
        # opening =   cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        wanted_boxes = self.get_bounding_boxes_of_tiles(contours)
        print len(wanted_boxes)

        cv2.drawContours(img, contours, -1, (100, 20, 100), 1)

        [cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
         for x, y, w, h in wanted_boxes]

        cv2.imshow("contours", img)

        return wanted_boxes

    def get_grid_rois(self, wanted_boxes):
        wanted_boxes = sorted(wanted_boxes, key=lambda box: box[0])
        wanted_boxes = sorted(wanted_boxes, key=lambda box: box[1])

        min_x, min_y = (10000000, 1000000)
        max_x, max_y = (-1, -1)
        avg_box = np.mean(wanted_boxes, axis=0)
        avg_w = avg_box[2]
        avg_h = avg_box[3]
        for box in wanted_boxes:
            x, y, w, h = box
            if (x < min_x):
                min_x = x
            if (y < min_y):
                min_y = y

            if (x + w > max_x):
                max_x = x + w
            if (y + h > max_y):
                max_y = y + h

        x_gap = ((max_x - min_x) - avg_w * self.cols) / (self.cols - 1)
        y_gap = ((max_y - min_y) - avg_h * self.rows - 1) / (self.rows - 1)

        img = self.image.copy()
        rectangles = []
        for i in range(self.cols):
            for j in range(self.rows):
                x = int(min_x + i * avg_w + i * x_gap)
                y = int(min_y + j * avg_h + j * y_gap)
                w = int(avg_w)
                h = int(avg_h)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                rectangles.append((x, y, w, h))

        cv2.imshow("slide", img)
        return rectangles

    def create_board(self):
        boxes = self.create_grid_of_boxes()
        rois = self.get_grid_rois(boxes)

        ld = LetterDetection(self.image, rois)
        letters = ld.detect()
        print "\n"
        print "\n"
        self.scrabble_grid.add_letters(letters)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return self.scrabble_grid

    def remove_small_areas(self, boxes):
        return filter(lambda box: box[2] * box[3] > 500, boxes)

    def filter_on_size_boxes(self, contours):
        def area_fn(box): return box[2] * box[3]

        boxes = map(cv2.boundingRect, contours)
        boxes = self.remove_small_areas(boxes)
        boxes = sorted(boxes, key=area_fn)
        median_area = area_fn(boxes[int(len(boxes) / 2)])
        return filter(lambda box: area_fn(box) > median_area / 1.2 and area_fn(box) < median_area * 1.12, boxes)

    def filter_on_pos(self, boxes):
        def looks_part_of_grid(box):
            height, _, _ = self.image.shape
            height = float(height)
            return box[1] > height * 0.1 and box[1] < height * 0.8

        return filter(looks_part_of_grid, boxes)

    def get_bounding_boxes_of_tiles(self, contours):
        boxes = self.filter_on_size_boxes(contours)
        return self.filter_on_pos(boxes)

    def get_letters_to_use(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(gray, (3, 3), 0)

        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
