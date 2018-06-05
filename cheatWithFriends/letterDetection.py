import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

IMAGE_SIZE = 1800
BINARY_THREHOLD = 180


class LetterDetection:
    def __init__(self, image, rois):
        self.image = image
        self.rois = rois

    def detect(self):
        img = self.image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        board = [0] * 121
        cv2.imshow("img", img)
        index = 0
        for index, roi in enumerate(self.rois):
            x, y, w, h = roi
            sub_img = img[y:y + h, x:x + w]
            if self.filter_grey_tiles(sub_img):
                continue

            roi = self.enhance_sub_image(sub_img)
            if roi is None:
                continue

            text = self.image_to_text(roi)
            board[index] = text

        x = np.array(board).reshape(11, 11)
        t = np.transpose(x)
        self.pretty_print(t)
        return t

    def pretty_print(self, matrix):
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print '\n'.join(table)

    def image_to_text(self, img):
        text = pytesseract.image_to_string(
            img, lang='eng', boxes=False, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ load_system_dawg=0 load_freq_dog=0')
        return text.encode('ascii', 'ignore')

    def reduce_colors(self, img):
        Z = img.reshape((-1, 3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, label, center = cv2.kmeans(
            Z, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        return res2

    def tighter_text_bounding_box(self, bw):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        # using RETR_EXTERNAL instead of RETR_CCOMP
        _, contours, hierarchy = cv2.findContours(
            connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        sub_x, sub_y = bw.shape
        mask = np.zeros(bw.shape, dtype=np.uint8)
        roi = None
        max_area = -1
        cv2.imshow("bw", connected)
        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            mask[y:y + h, x:x + w] = 0
            cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
            r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

            if r > 0.2 and w > 8 and h > sub_y * 0.2 and x is not 0:
                roi_area = h * w
                if roi_area > max_area:
                    max_area = roi_area
                    roi = (x - 5, y - 5, w + 10, h + 10)

        return roi

    def filter_grey_tiles(self, sub_img):
        img = sub_img.copy()
        one_color = self.reduce_colors(img)
        b, g, r = one_color[0][0]
        return b > 200 and g > 200 and r > 200

    def clean_image(self, sub_grey):
        sub_grey = cv2.GaussianBlur(sub_grey, (3, 3), 0)
        _, white = cv2.threshold(sub_grey, 200, 255, cv2.THRESH_BINARY)
        _, black = cv2.threshold(sub_grey, 50, 255, cv2.THRESH_BINARY_INV)
        return black + white

    def enhance_sub_image(self, sub_img):
        img = sub_img.copy()
        img = cv2.resize(img, None, fx=3.0,
                         fy=3.0, interpolation=cv2.INTER_CUBIC)

        sub_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = self.clean_image(sub_grey)
        roi = self.tighter_text_bounding_box(binary)
        if (roi is None):
            return None

        x, y, w, h = roi
        sub = binary[y:y + h, x:x + w]
        return sub
