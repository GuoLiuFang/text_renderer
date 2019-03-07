import random
import cv2
import numpy as np
import math

class LineState(object):
    tableline_x_offsets = range(-32, 8)
    tableline_y_offsets = range(-4, 8)
    tableline_thickness = [1, 2, 2, 2, 2, 2, 2, 2, 2, 3]

    # 0/1/2/3: 仅单边（左上右下）
    # 4/5/6/7: 两边都有线（左上，右上，右下，左下）
    tableline_options = [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2]
    # tableline_options = range(0, 8)

    middleline_thickness = [1, 2, 3]
    middleline_thickness_p = [0.2, 0.7, 0.1]


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
        y_offset = random.choice([-1, 0, 1])

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
        left_top_x = text_box_pnts[0][0]
        left_top_y = text_box_pnts[0][1]
        right_top_x = text_box_pnts[1][0]
        right_top_y = text_box_pnts[1][1]
        right_bottom_x = text_box_pnts[2][0]
        right_bottom_y = text_box_pnts[2][1]
        left_bottom_x = text_box_pnts[3][0]
        left_bottom_y = text_box_pnts[3][1]
        dst = word_img
        # word_img是image对象,所以，size[0]是宽，size[1]是高。通过高来判断字符的大小问题。。
        # 划线区域，集中在，0.618 ** 5次方这个区间内。。
        t_img_w = word_img.shape[1]
        t_img_h = word_img.shape[0]
        option = random.choice(self.linestate.tableline_options)
        thickness = random.choice(self.linestate.tableline_thickness)
        line_color = word_color + random.randint(0, 10)
        ## 偏移长度。。在5%和95%之间进行。。进入为负数，远离为正数。。
        ## 进入到0.95的程度，远离到0.382的程度。。。这是水平。。
        ## 在垂直方向控制进入和远离各10%的水平
        # top_y_offset = random.choice(self.linestate.tableline_y_offsets)
        if np.random.uniform(0, 1) < 1.0 - 0.618 ** 3:
            top_y_offset = random.randint(0 - math.floor(t_img_h * (0.618 ** 4)), 0 - math.floor(t_img_h * (0.618 ** 6)))
        else:
            top_y_offset = random.randint(0 , math.floor(t_img_h * (0.618 ** 6)))
        if np.random.uniform(0, 1) < 1.0 - 0.618 ** 3:
            bottom_y_offset = random.randint(0 - math.floor(t_img_h * (0.618 ** 4)), 0 - math.floor(t_img_h * (0.618 ** 6)))
        else:
            bottom_y_offset = random.randint(0, math.floor(t_img_h * (0.618 ** 6)))
        # bottom_y_offset = random.choice(self.linestate.tableline_y_offsets)
        if np.random.uniform(0, 1) < 1.0 - 0.618 ** 4:
            left_x_offset = random.randint(0 - math.floor(t_img_h * (0.618 ** 1)), 0 - math.floor(t_img_h * (0.618 ** 5)))
        else:
            left_x_offset = random.randint(0, math.floor(t_img_h * (0.618 ** 7)))
        # left_x_offset = random.choice(self.linestate.tableline_x_offsets)
        if np.random.uniform(0, 1) < 1.0 - 0.618 ** 4:
            right_x_offset = random.randint(0 - math.floor(t_img_h * (0.618 ** 1)), 0 - math.floor(t_img_h * (0.618 ** 5)))
        else:
            right_x_offset = random.randint(0, math.floor(t_img_h * (0.618 ** 7)))
        # right_x_offset = random.choice(self.linestate.tableline_x_offsets)

        def is_top():
            return option in [1, 4, 5]

        def is_bottom():
            return option in [3, 6, 7]

        def is_left():
            return option in [0, 4, 7]

        def is_right():
            return option in [2, 5, 6]

        if is_top():
            left_top_y -= top_y_offset
            right_top_y -= top_y_offset

        if is_bottom():
            right_bottom_y += bottom_y_offset
            left_bottom_y += bottom_y_offset

        if is_left():
            left_top_x -= left_x_offset
            left_bottom_x -= left_x_offset

        if is_right():
            right_top_x += right_x_offset
            right_bottom_x += right_x_offset

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

        if is_left():
            dst = cv2.line(dst,
                           (left_top_x, 0),
                           (left_bottom_x, word_img.shape[0]),
                           color=line_color,
                           thickness=thickness,
                           lineType=cv2.LINE_AA)

        if is_right():
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
