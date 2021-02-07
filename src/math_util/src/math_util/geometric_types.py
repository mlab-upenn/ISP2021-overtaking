

import numpy as np

class line:
    #represents line in form of y= mx+b
    def __init__(self, m, b):
        self.m = m
        self.b = b

    @staticmethod
    def get_line(self, pt1, pt2):
        if(pt1[0] == pt2[0]):
            m = np.infty
            #in this case b is the x intercept, not the y intercept
            b = pt1[0]
        else:
            m = (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
            b = pt1[1]-m*pt1[0]
        return line(m,b)

class line_segment:
    #represents line in form y= mx+b with two endpoints
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2
        self.line = line.get_line(self.pt1, self.pt2)

    def in_segment(self, pt):
        return np.dot(pt-self.pt1, pt-self.pt2)<0


class circle:
    def __init__(self, pt, r):
        self.pt = pt
        self.r = np.abs(r)







