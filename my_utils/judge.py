# -*- coding: utf-8 -*-

from my_utils import geometry
import math
import numpy as np
from shapely.geometry import Point
import time
from pprint import pprint
import cv2
import numpy as np
import time
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from my_utils import cvdraw
from shapely.geometry.polygon import Polygon


def HumanMainPostionExtract(TableHumanKPTimeList):

    TrackingMainPostionNP = []
    TrackingWholePostionNP = []

    for FrameKP in TableHumanKPTimeList:
        for humanKP in FrameKP:
            TrackingMainPostionNP.append([humanKP[3][0], humanKP[3][1]])
            TrackingWholePostionNP.append(humanKP)
            # cv2.circle(frame, (int(humanKP[3][0]), int(humanKP[3][1])), 10, (0, 255, 0), -1)
    return np.array(TrackingMainPostionNP), np.array(TrackingWholePostionNP)


def DBscanCluster(TrackingMainPostionNP):

    db = DBSCAN(eps=70, min_samples=10).fit(TrackingMainPostionNP)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # pprint(labels)

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    return labels, n_clusters_


def FinalPosition(labels, TrackingWholePostionNP):
    FinalHumanKpCoorList = []
    for i in range(labels.max() + 1):

        cluster_index = labels[labels == i]
        FinalHumanKpCoor = TrackingWholePostionNP[cluster_index[-1]]
        FinalHumanKpCoorList.append(FinalHumanKpCoor)

    return np.array(FinalHumanKpCoorList)


def HeightWidthThreshhold(tableHumanKpCoorListOverlap, mainaxis, subaxis):

    IntableHumanKpCoorList = []
    for FinalHumanKpCoor in tableHumanKpCoorListOverlap:
        RShoulderFinalCoor = FinalHumanKpCoor[4]
        LShoulderFinalCoor = FinalHumanKpCoor[5]
        RHipFinalCoor = FinalHumanKpCoor[1]

        p1 = RShoulderFinalCoor[0:2]
        p2 = RHipFinalCoor[0:2]
        p3 = p2 - p1
        Height = math.hypot(p3[0], p3[1])

        p1 = RShoulderFinalCoor[0:2]
        p2 = LShoulderFinalCoor[0:2]
        p3 = p2 - p1
        Width = math.hypot(p3[0], p3[1])

        if Width < subaxis and Height < 1.3 * mainaxis:
            IntableHumanKpCoorList.append(FinalHumanKpCoor)
        else:
            print('Length of MainRectangle too big, Human not siting.')

    return np.array(IntableHumanKpCoorList)


def OverLapPercentage(TableCali, FinalHumanKpCoorList, overlapthreshhold):

    main_ax = TableCali['long_axis']
    sub_ax = TableCali['short_axis']
    el_x = TableCali['x_center']
    el_y = TableCali['y_center']

    IntableHumanKpCoorList = []

    OverLapPercentageList = []
    for FinalHumanKpCoor in FinalHumanKpCoorList:

        RShoulderFinalCoor = FinalHumanKpCoor[4]
        LShoulderFinalCoor = FinalHumanKpCoor[5]
        MidShoulderFinalCoor = (RShoulderFinalCoor + LShoulderFinalCoor) / 2
        MidFinalCoor = FinalHumanKpCoor[3]

        XYAddRight = RShoulderFinalCoor - MidShoulderFinalCoor
        XYAddLeft = LShoulderFinalCoor - MidShoulderFinalCoor

        LHipDraw = MidFinalCoor + XYAddLeft
        RHipDraw = MidFinalCoor + XYAddRight

        min_x = min([
            RShoulderFinalCoor[0], LShoulderFinalCoor[0], LHipDraw[0],
            RHipDraw[0]
        ])
        max_x = max([
            RShoulderFinalCoor[0], LShoulderFinalCoor[0], LHipDraw[0],
            RHipDraw[0]
        ])
        min_y = min([
            RShoulderFinalCoor[1], LShoulderFinalCoor[1], LHipDraw[1],
            RHipDraw[1]
        ])
        max_y = max([
            RShoulderFinalCoor[1], LShoulderFinalCoor[1], LHipDraw[1],
            RHipDraw[1]
        ])

        jumpstepx = (max_x + 1 - min_x) / 10
        jumpstepy = (max_y + 1 - min_y) / 10

        if jumpstepx < 1:
            jumpstepx = 1

        if jumpstepy < 1:
            jumpstepy = 1

        polygon = Polygon([(LShoulderFinalCoor[0], LShoulderFinalCoor[1]),
                           (RShoulderFinalCoor[0], RShoulderFinalCoor[1]),
                           (RHipDraw[0], RHipDraw[1]),
                           (LHipDraw[0], LHipDraw[1])])

        pointstotalnum = 0
        pointsinellipsenum = 0

        for xindex in range(int(min_x), int(max_x + 1), int(jumpstepx)):
            for yindex in range(int(min_y), int(max_y + 1), int(jumpstepy)):
                if polygon.contains(Point(xindex, yindex)) is True:
                    pointstotalnum = pointstotalnum + 1
                    if geometry.isInEllipseTable(el_x, el_y, xindex, yindex,
                                                 main_ax, sub_ax,
                                                 TableCali['theta']) is True:
                        pointsinellipsenum = pointsinellipsenum + 1

        if pointstotalnum == 0:
            OverLapPercentageList.append(0.0)

        else:
            OverLapPercentageList.append(100 * pointsinellipsenum /
                                         pointstotalnum)
        if OverLapPercentageList[-1] >= overlapthreshhold:
            print('Overlap is ' + str(OverLapPercentageList[-1]) + '% Human not siting.')
        else:
            IntableHumanKpCoorList.append(FinalHumanKpCoor)
    return OverLapPercentageList, np.array(IntableHumanKpCoorList)


def IsInEllipseTableArround(TableID, TableCali, UsedKeyPointsList):

    Neck = UsedKeyPointsList[0]
    RHip = UsedKeyPointsList[1]
    LHip = UsedKeyPointsList[2]
    RShoulder = UsedKeyPointsList[4]
    LShoulder = UsedKeyPointsList[5]

    main_ax = TableCali['long_axis']
    sub_ax = TableCali['short_axis']
    el_x = TableCali['x_center']
    el_y = TableCali['y_center']

    check_3 = True
    if TableID == 7:
        check_1 = geometry.isInEllipseTable(el_x + 20, el_y, RShoulder[0],
                                            RShoulder[1], 2.5 * main_ax,
                                            3.8 * sub_ax, TableCali['theta'])
        check_2 = geometry.isInEllipseTable(el_x + 20, el_y, LShoulder[0],
                                            LShoulder[1], 2.5 * main_ax,
                                            3.8 * sub_ax, TableCali['theta'])
    elif TableID == 10:
        check_1 = geometry.isInEllipseTable(el_x, el_y, RShoulder[0],
                                            RShoulder[1], 3.2 * main_ax,
                                            4.2 * sub_ax, TableCali['theta'])
        check_2 = geometry.isInEllipseTable(el_x, el_y, LShoulder[0],
                                            LShoulder[1], 3.2 * main_ax,
                                            4.2 * sub_ax, TableCali['theta'])
        check_3 = geometry.isInEllipseTable(el_x, el_y, Neck[0], Neck[1],
                                            3.2 * main_ax, 4.2 * sub_ax,
                                            TableCali['theta'])
    elif TableID == 11:
        check_1 = geometry.isInEllipseTable(el_x, el_y - 23, RShoulder[0],
                                            RShoulder[1], 2.7 * main_ax,
                                            4.5 * sub_ax, TableCali['theta'])
        check_2 = geometry.isInEllipseTable(el_x, el_y - 23, LShoulder[0],
                                            LShoulder[1], 2.7 * main_ax,
                                            4.5 * sub_ax, TableCali['theta'])
    elif TableID == 12:
        check_1 = geometry.isInEllipseTable(el_x + 25, el_y + 120, RHip[0],
                                            RHip[1], 2.2 * main_ax,
                                            2.2 * sub_ax, TableCali['theta'])
        check_2 = geometry.isInEllipseTable(el_x + 25, el_y + 120, LHip[0],
                                            LHip[1], 2.2 * main_ax,
                                            2.2 * sub_ax, TableCali['theta'])
    elif TableID == 13:
        check_1 = geometry.isInEllipseTable(el_x + 60, el_y + 135, RHip[0],
                                            RHip[1], 2.1 * main_ax, 3 * sub_ax,
                                            TableCali['theta'])
        check_2 = geometry.isInEllipseTable(el_x + 60, el_y + 135, LHip[0],
                                            LHip[1], 2.1 * main_ax, 3 * sub_ax,
                                            TableCali['theta'])

    return check_1 and check_2 and check_3
