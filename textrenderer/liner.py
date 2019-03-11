import random
import cv2
import numpy as np
import math

class LineState(object):
    tableline_x_offsets = range(8, 32)
    tableline_y_offsets = range(2, 8)
    tableline_thickness = [1, 2, 2, 2, 2, 2, 2, 2, 2]

    # 0/1/2/3: 仅单边（左上右下）
    # 4/5/6/7: 两边都有线（左上，右上，右下，左下）
    tableline_options = [0, 2, 3, 3, 0, 2, 0, 2, 10, 11, 10, 11, 10, 11, 9, 9, 9, 1, 3, 1, 3, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 7, 7]
    # tableline_options = range(0, 8)

    middleline_thickness = [1, 2]
    middleline_thickness_p = [0.2, 0.8]


class Liner(object):
    def __init__(self, cfg):
        self.linestate = LineState()
        self.cfg = cfg

    def apply(self, word_img, text_box_pnts, word_color):
        """
        :param word_img:  word image with big background
        :param text_box_pnts: left-top, right-top, right-bottom, left-bottom of text word
        :return:
        """
        line_p = []
        funcs = []

        if self.cfg.line.under_line.enable:
            line_p.append(self.cfg.line.under_line.fraction)
            funcs.append(self.apply_under_line)

        if self.cfg.line.table_line.enable:
            line_p.append(self.cfg.line.table_line.fraction)
            funcs.append(self.apply_table_line)

        if self.cfg.line.middle_line.enable:
            line_p.append(self.cfg.line.middle_line.fraction)
            funcs.append(self.apply_middle_line)

        if len(line_p) == 0:
            return word_img, text_box_pnts

        line_effect_func = np.random.choice(funcs, p=line_p)

        return line_effect_func(word_img, text_box_pnts, word_color)

    def apply_under_line(self, word_img, text_box_pnts, word_color):
        y_offset = random.choice([-1, -1, 0, 1, 0, 1, 0, 1, 0, 1, 3, 3, 3, 4, 5, 2, 2])

        text_box_pnts[2][1] += y_offset
        text_box_pnts[3][1] += y_offset

        line_color = word_color + random.randint(0, 10)

        dst = cv2.line(word_img,
                       (text_box_pnts[2][0], text_box_pnts[2][1]),
                       (text_box_pnts[3][0], text_box_pnts[3][1]),
                       color=line_color,
                       thickness=1,
                       lineType=cv2.LINE_AA)
        self.apply_table_line(dst, text_box_pnts, word_color)
        return dst, text_box_pnts

    def apply_table_line(self, word_img, text_box_pnts, word_color):
        """
        共有 8 种可能的画法，横线横穿整张 word_img
        0/1/2/3: 仅单边（左上右下）
        4/5/6/7: 两边都有线（左上，右上，右下，左下）
        """

        t_img_h = word_img.shape[0]
        t_img_w = word_img.shape[1]

        t_box_h = text_box_pnts[2][1] - text_box_pnts[0][1]
        t_box_w = text_box_pnts[2][0] - text_box_pnts[0][0]

        s_space_h = t_img_h - t_box_h
        s_space_w = t_box_w - t_box_w 

        left_top_x = text_box_pnts[0][0]
        left_top_y = text_box_pnts[0][1]
        right_top_x = text_box_pnts[1][0]
        right_top_y = text_box_pnts[1][1]
        right_bottom_x = text_box_pnts[2][0]
        right_bottom_y = text_box_pnts[2][1]
        left_bottom_x = text_box_pnts[3][0]
        left_bottom_y = text_box_pnts[3][1]

        dst = word_img

        option = random.choice(self.linestate.tableline_options)
        thickness = random.choice(self.linestate.tableline_thickness)
        line_color = word_color + random.randint(0, 10)

        # 这是解决嵌入的问题
        top_y_offset = random.randint(-7, math.floor((s_space_h - left_top_y) * 0.382))
        bottom_y_offset = random.randint(- math.floor((left_bottom_y - t_box_h) * 0.382), 7)
        left_x_offset = random.randint(math.floor((t_box_h) * (0.382 ** 2)), math.floor((t_box_h) * 1.618))
        right_x_offset = random.randint(- math.floor((t_box_h) * 1.618), - math.floor((t_box_h) * (0.382 ** 2)))
        # 这是解决距离的问题
        t_top_y_offset = random.choice(self.linestate.tableline_y_offsets) + random.randint(-2, 2)
        t_bottom_y_offset = random.choice(self.linestate.tableline_y_offsets) + random.randint(-1, 1)
        t_left_x_offset = random.choice(self.linestate.tableline_x_offsets)
        t_right_x_offset = random.choice(self.linestate.tableline_x_offsets)        
        def is_top():
            return option in [1, 4, 5]

        def is_bottom():
            return option in [3, 6, 7]

        def is_left():
            return option in [0, 4, 7]

        def is_right():
            return option in [2, 5, 6]
        
        def is_vertical():
            return option in [9]

        def is_left_padding():
            return option in [10]

        def is_right_padding():
            return option in [11]

        if is_left_padding():
            left_top_x -= min(abs(t_left_x_offset - random.randint(1, 5)), abs(left_x_offset))
            left_bottom_x -= min(abs(t_left_x_offset - random.randint(1, 5)), abs(left_x_offset))

            text_box_pnts[0][0] -= t_left_x_offset
            text_box_pnts[3][0] -= t_left_x_offset

        if is_right_padding():
            right_top_x += min(abs(t_right_x_offset - random.randint(1, 5)), abs(right_x_offset))
            right_bottom_x += min(abs(t_right_x_offset - random.randint(1, 5)), abs(right_x_offset))

            text_box_pnts[1][0] += t_right_x_offset
            text_box_pnts[2][0] += t_right_x_offset      
        
        if is_vertical():
            left_top_x -= min(abs(t_left_x_offset - random.randint(1, 5)), abs(left_x_offset))
            left_bottom_x -= min(abs(t_left_x_offset - random.randint(1, 5)), abs(left_x_offset))

            text_box_pnts[0][0] -= t_left_x_offset
            text_box_pnts[3][0] -= t_left_x_offset

            right_top_x += min(abs(t_right_x_offset - random.randint(1, 5)), abs(right_x_offset))
            right_bottom_x += min(abs(t_right_x_offset - random.randint(1, 5)), abs(right_x_offset))

            text_box_pnts[1][0] += t_right_x_offset
            text_box_pnts[2][0] += t_right_x_offset                            

        if is_top():
            left_top_y += top_y_offset
            right_top_y += top_y_offset

            text_box_pnts[0][1] -= t_top_y_offset
            text_box_pnts[1][1] -= t_top_y_offset

        if is_bottom():
            right_bottom_y += bottom_y_offset
            left_bottom_y += bottom_y_offset

            text_box_pnts[2][1] += t_bottom_y_offset
            text_box_pnts[3][1] += t_bottom_y_offset            

        if is_left():
            left_top_x += left_x_offset
            left_bottom_x += left_x_offset

            text_box_pnts[0][0] -= t_left_x_offset
            text_box_pnts[3][0] -= t_left_x_offset            

        if is_right():
            right_top_x += right_x_offset
            right_bottom_x += right_x_offset

            text_box_pnts[1][0] += t_right_x_offset
            text_box_pnts[2][0] += t_right_x_offset            

        if is_bottom():
            dst = cv2.line(dst,
                           (0, right_bottom_y),
                           (word_img.shape[1], left_bottom_y),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_top():
            dst = cv2.line(dst,
                           (0, left_top_y),
                           (word_img.shape[1], right_top_y),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_left() or is_left_padding():
            dst = cv2.line(dst,
                           (left_top_x, 0),
                           (left_bottom_x, word_img.shape[0]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_right() or is_right_padding():
            dst = cv2.line(dst,
                           (right_top_x, 0),
                           (right_bottom_x, word_img.shape[0]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_vertical():
            dst = cv2.line(dst,
                           (left_top_x, 0),
                           (left_bottom_x, word_img.shape[0]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)     

            dst = cv2.line(dst,
                           (right_top_x, 0),
                           (right_bottom_x, word_img.shape[0]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        return dst, text_box_pnts

    def apply_middle_line(self, word_img, text_box_pnts, word_color):
        y_center = int((text_box_pnts[0][1] + text_box_pnts[3][1]) / 2)

        img_mean = int(np.mean(word_img))
        thickness = np.random.choice(self.linestate.middleline_thickness, p=self.linestate.middleline_thickness_p)

        dst = cv2.line(word_img,
                       (text_box_pnts[0][0], y_center),
                       (text_box_pnts[1][0], y_center),
                       color=img_mean,
                       thickness=thickness,
                       lineType=cv2.LINE_AA)
        self.apply_table_line(dst, text_box_pnts, word_color)
        return dst, text_box_pnts
