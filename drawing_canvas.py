# file: drawing_canvas.py

import cv2
import numpy as np


class DrawingCanvas:
    def __init__(self):
        self.canvas = None
        self.prev_points = {}

    def initialize_canvas(self, frame):
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

    def reset_previous_point(self, hand_key):
        self.prev_points[hand_key] = None

    def reset_all_previous_points(self):
        self.prev_points = {}

    def draw_line(self, hand_key, x, y, color, thickness):
        if self.canvas is None:
            return

        prev_point = self.prev_points.get(hand_key)

        if prev_point is None:
            self.prev_points[hand_key] = (x, y)
            return

        prev_x, prev_y = prev_point

        cv2.line(
            self.canvas,
            (prev_x, prev_y),
            (x, y),
            color,
            thickness
        )

        self.prev_points[hand_key] = (x, y)

    def clear_canvas(self):
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)

        self.reset_all_previous_points()

    def merge_with_frame(self, frame):
        if self.canvas is None:
            return frame

        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)

        _, inverse_canvas = cv2.threshold(
            gray_canvas,
            20,
            255,
            cv2.THRESH_BINARY_INV
        )

        inverse_canvas = cv2.cvtColor(inverse_canvas, cv2.COLOR_GRAY2BGR)

        frame_without_drawing_area = cv2.bitwise_and(frame, inverse_canvas)
        final_frame = cv2.bitwise_or(frame_without_drawing_area, self.canvas)

        return final_frame

    def save_canvas(self, path):
        if self.canvas is not None:
            cv2.imwrite(path, self.canvas)
            return True

        return False