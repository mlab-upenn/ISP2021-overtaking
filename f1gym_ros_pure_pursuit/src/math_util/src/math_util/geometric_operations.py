
import numpy as np

def importTest():
    print('import successful for geometric_operations')

def intersection_of_lines(line1, line2):
    ret = []
    if(line1.m == line2.m):
        #parallel lines, return empty as no intersection is chosen
        pass
    elif(line1.m == np.infty):
        x = line1.b
        ret.append(np.array([x, line2.m*x+line2.b]))

    elif(line2.m == np.infty):
        x = line2.b
        ret.append(np.array([x,line1.m*x+line1.b]))
    else:
        x = (line2.b - line1.b)/(line1.m - line2.m)
        y = line1.m*x + line1.b
        ret.append(np.array([x,y]))

    return ret


def intersection_of_segment_and_line(line, segment):

    points = intersection_of_lines(line,segment.line)
    if(len(points) > 0):
        if(segment.in_segment(points[0])):
            return points

    return []

def intersection_of_segments(segment1, segment2):
    points = intersection_of_lines(segment1.line, segment2.line)
    if(len(points)>0):
        if(segment1.in_segment(points[0]) and segment2.in_segment(points[0])):
            return points
    return []

def intersection_of_circle_and_line(circle, line):
    ret = []
    xc = circle.pt[0]
    yc = circle.pt[1]
    if(line.m == np.infty):
        # case of vertical line
        x = line.b
        det = circle.r**2 - (x - xc)**2
        if(det > 0):
            y1 = np.sqrt(det) + yc
            y2 = -np.sqrt(det) + yc
            ret.append(np.array([x,y1]))
            ret.append(np.array([x,y2]))
        elif(det == 0):
            y = yc
            ret.append(np.array([x,y]))
        else:
            #no intersection
            pass

    else:

        a = (1+line.m**2)
        b = 2*line.m*line.b - 2*xc - 2*line.m*yc
        c = xc**2 - 2*yc*line.b + yc**2 + line.b**2 - circle.r**2

        det = b**2 - 4*a*c

        if(det < 0):
            # no intersection
            pass
        elif(det == 0):
            x = -b/(2*a)

        else:
            x1 = (-b+np.sqrt(det))/(2*a)
            x2 = (-b-np.sqrt(det))/(2*a)

            y1 = line.m*x1 + line.b
            y2 = line.m*x2 + line.b

            ret.append(np.array([x1,y1]))
            ret.append(np.array([x2,y2]))

    return ret

def intersection_of_circle_and_segment(circle, segment):
    ret = []
    points = intersection_of_circle_and_line(circle, segment.line)
    if (len(points) > 0):
        for p in points:
            if(segment.in_segment(p)):
                ret.append(p)

    return ret