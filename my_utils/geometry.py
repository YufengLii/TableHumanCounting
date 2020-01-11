# -*- coding: utf-8 -*-

import numpy as np
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import sin, cos, atan2, pi, fabs
import math


# 计算两点之间的距离
def linelength(pt1, pt2):
    return pow(pow((pt2[1] - pt1[1]), 2) + pow((pt2[0] - pt1[0]), 2), 0.5)


# 判断点是否在椭圆桌内


def isInEllipseTable(x, y, xp, yp, d, D, angle):
    #print(x,y,d,D,angle)
    # [xp, yp]       待检测点坐标
    # x,             椭圆中心点X坐标
    # y,             椭圆中心点Y坐标
    # d,             椭圆长轴
    # D,             椭圆短轴
    # angle          椭圆逆时针旋转角度
    # 调用实例
    # isInEllipseTable(ellipse_x_center, ellipse_y_center, check_point[0],
    #                  check_point[1], long_axis, short_axis,
    #                  180 - math.degrees(theta))

    cosa = math.cos(angle)
    sina = math.sin(angle)
    dd = d * d
    DD = D * D

    a = math.pow(cosa * (xp - x) + sina * (yp - y), 2)
    b = math.pow(sina * (xp - x) - cosa * (yp - y), 2)
    ellipse = (a / dd) + (b / DD)

    if ellipse <= 1:
        return True
    else:
        return False


def ellipe_tan_dot(rx, ry, px, py, theta):
    '''Dot product of the equation of the line formed by the point
    with another point on the ellipse's boundary and the tangent of the ellipse
    at that point on the boundary.
    '''
    return ((rx**2 - ry**2) * cos(theta) * sin(theta) - px * rx * sin(theta) +
            py * ry * cos(theta))


def ellipe_tan_dot_derivative(rx, ry, px, py, theta):
    '''The derivative of ellipe_tan_dot.
    '''
    return ((rx**2 - ry**2) * (cos(theta)**2 - sin(theta)**2) -
            px * rx * cos(theta) - py * ry * sin(theta))


# 计算点到椭圆的最短距离，以及离该点最近的椭圆边界上点的坐标，


def estimateDistance(x, y, rx, ry, x0=0, y0=0, angle=0, error=1e-5):
    # x0,           椭圆中心点X坐标
    # y0,           椭圆中心点Y坐标
    # rx,           椭圆长轴
    # ry,           椭圆短轴
    # angle         椭圆逆时针旋转角度
    # x, y          待计算点坐标
    # 调用实例
    # distance, px, py = estimate_distance(check_point[0], check_point[1],
    #                                      long_axis, short_axis, x_center,
    #                                      y_center, 180 - math.degrees(theta))
    x -= x0
    y -= y0
    if angle:
        # rotate the points onto an ellipse whose rx, and ry lay on the x, y
        # axis
        angle = -pi / 180. * angle
        x, y = x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle)

    theta = atan2(rx * y, ry * x)
    while fabs(ellipe_tan_dot(rx, ry, x, y, theta)) > error:
        theta -= ellipe_tan_dot(
            rx, ry, x, y, theta) / \
            ellipe_tan_dot_derivative(rx, ry, x, y, theta)

    px, py = rx * cos(theta), ry * sin(theta)
    #cv2.circle(image, (px, py), 10, (0, 0, 255), -1)
    intersection_x = px * cos(-angle) - py * sin(-angle) + x0
    intersection_y = px * sin(-angle) + py * cos(-angle) + y0
    return ((x - px)**2 + (y - py)**2)**.5, intersection_x, intersection_y


# 计算两向量之间的夹角


def getAngle(p0, p1=np.array([0, 0]), p2=None):

    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return abs(np.degrees(angle))


# 判断点是否在四边形内


def isInSquareTable(table_coor, point):

    # table_coor,      按照顺时针方向的桌子顶点
    # point,           待计算点坐标 例，Point(Neck[0], Neck[1])
    # 调用实例
    # table_back = [(ellipse_cali_info['back_top_left'][0],
    # ellipse_cali_info['back_top_left'][1]),
    # (ellipse_cali_info['back_top_right'][0],
    # ellipse_cali_info['back_top_right'][1]),
    # check_1 = isInSquareTable(table_back, Point(Neck[0], Neck[1]))

    polygon = Polygon(table_coor)
    inout = polygon.contains(point)
    if inout:
        return True
    else:
        return False


def isBetween2value(x, min_threshold, max_threshold):
    if x > min_threshold and x < max_threshold:
        return True
    else:
        return False


def isbetween2ellipse(ellipse_small, ellipse_large, check_point):
    judge_1 = is_in_ellipse_table(ellipse_small['x_center'],
                                  ellipse_small['y_center'], check_point[0],
                                  check_point[1], ellipse_small['long_axis'],
                                  ellipse_small['short_axis'],
                                  ellipse_small['theta'])
    judge_2 = is_in_ellipse_table(ellipse_large['x_center'],
                                  ellipse_large['y_center'], check_point[0],
                                  check_point[1], ellipse_large['long_axis'],
                                  ellipse_large['short_axis'],
                                  ellipse_large['theta'])
    if judge_1 is False and judge_2 is True:
        return True
    else:
        return False


def distanceThreshold(pt1, pt2, curr_thresh):

    distance = pow(pow((pt2[1] - pt1[1]), 2) + pow((pt2[0] - pt1[0]), 2), 0.5)
    if distance <= curr_thresh:
        return True
    else:
        return False


def KeypointPositionCheck(ellipse_parameter, humans_position, threshold_center,
                          threshold_edge):

    if is_in_ellipse_table(ellipse_parameter['x_center'],
                           ellipse_parameter['y_center'], humans_position[0],
                           humans_position[1], ellipse_parameter['long_axis'],
                           ellipse_parameter['short_axis'],
                           ellipse_parameter['theta']) is False:
        distancetotable, px, py = estimate_distance(
            humans_position[0], humans_position[1],
            ellipse_parameter['long_axis'], ellipse_parameter['short_axis'],
            ellipse_parameter['x_center'], ellipse_parameter['y_center'],
            180 - math.degrees(ellipse_parameter['theta']))
        #print("Distance to Table", ellipse_cali_info['tableID'],  " is: ", distancetotable)
        #print(ellipse_cali_info['short_axis'])
        if distancetotable <= threshold_edge:
            # cv2.line(image,
            #          ((int)(humans_position[0]), (int)(humans_position[1])),
            #          ((int)(ellipse_cali_info['x_center']),
            #           (int)(ellipse_cali_info['y_center'])), (255, 0, 0), 4, 4)
            return True
        else:
            return False
    else:
        distancetotable = pow(
            pow((ellipse_parameter['y_center'] - humans_position[1]), 2) + pow(
                (ellipse_parameter['x_center'] - humans_position[0]), 2), 0.5)
        if distancetotable > threshold_center:
            # cv2.line(image,
            #          ((int)(humans_position[0]), (int)(humans_position[1])),
            #          ((int)(ellipse_cali_info['x_center']),
            #           (int)(ellipse_cali_info['y_center'])), (255, 0, 0), 4, 4)
            return True
        else:
            return False
