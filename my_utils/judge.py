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
import json



def UpdateTableHumanKPTimeList(TableHumanKPTimeList, KeypointListMaxLength,
                               CurrTableHumanKP, STableCaliInfo):

    FrontReadyAnalyze = False
    BackReadyAnalyze = False

    if STableCaliInfo['front_top_left'] != [0, 0]:
        if len(CurrTableHumanKP['front']) == 0:
            TableHumanKPTimeList[str(
                STableCaliInfo['tableID'])]['front'] = []
            # print('no human Table',
            #     str(STableCaliInfo['tableID']),
            #     'Front, Table List Reset',
            #     sep=' ')
        else:
            TableHumanKPTimeList[str(
                STableCaliInfo['tableID'])]['front'].append(
                CurrTableHumanKP['front'])


    if len(TableHumanKPTimeList[str(STableCaliInfo['tableID'])]
            ['front']) == KeypointListMaxLength:
        print('桌子区域前方有人驻留，进入聚类')
        # print('Table',
        #         str(STableCaliInfo['tableID']),
        #         'Front List full Judge Ready!!!',
        #         sep=' ')
        FrontReadyAnalyze = True
    # else:
    #     print('Table',
    #             str(STableCaliInfo['tableID']),
    #             'Front List not full Judge not Ready!!!',
    #             sep=' ')


    if STableCaliInfo['back_top_left'] != [0, 0]:

        if len(CurrTableHumanKP['back']) == 0:
            TableHumanKPTimeList[str(
                STableCaliInfo['tableID'])]['back'] = []
            # print('no human Table',
            #     str(STableCaliInfo['tableID']),
            #     'Back, Table List Reset',
            #     sep=' ')
        else:
            TableHumanKPTimeList[str(
                STableCaliInfo['tableID'])]['back'].append(
                    CurrTableHumanKP['back'])


    if len(TableHumanKPTimeList[str(STableCaliInfo['tableID'])]
            ['back']) == KeypointListMaxLength:
        print('桌子区域后方有人驻留，进入聚类')
        # print('Table',
        #         str(STableCaliInfo['tableID']),
        #         'Back List full Judge Ready!!!',
        #         sep=' ')
        BackReadyAnalyze = True

    # else:
    #     print('Table',
    #             str(STableCaliInfo['tableID']),
    #             'Back List not full Judge not Ready!!!',
    #             sep=' ')

    return TableHumanKPTimeList, FrontReadyAnalyze, BackReadyAnalyze


def IsInSquareTF(STableCaliInfo, usedKeyPointsList):
    iNTable = False

    RShoulder = usedKeyPointsList[0]
    LShoulder = usedKeyPointsList[1]

    table_front = [(STableCaliInfo['front_top_left'][0],
                    STableCaliInfo['front_top_left'][1]),
                   (STableCaliInfo['front_top_right'][0],
                    STableCaliInfo['front_top_right'][1]),
                   (STableCaliInfo['front_bottom_right'][0],
                    STableCaliInfo['front_bottom_right'][1]),
                   (STableCaliInfo['front_bottom_left'][0],
                    STableCaliInfo['front_bottom_left'][1])]

    if table_front[0] != (0, 0):
        check_1 = geometry.isInSquareTable(table_front,
                                           Point(RShoulder[0], RShoulder[1]))
        check_2 = geometry.isInSquareTable(table_front,
                                           Point(LShoulder[0], LShoulder[1]))
        iNTable = check_1 and check_2

    return iNTable


def IsInSquareTB(STableCaliInfo, usedKeyPointsList):
    iNTable = False

    RShoulder = usedKeyPointsList[0]
    LShoulder = usedKeyPointsList[1]

    table_back = [(STableCaliInfo['back_top_left'][0],
                   STableCaliInfo['back_top_left'][1]),
                  (STableCaliInfo['back_top_right'][0],
                   STableCaliInfo['back_top_right'][1]),
                  (STableCaliInfo['back_bottom_right'][0],
                   STableCaliInfo['back_bottom_right'][1]),
                  (STableCaliInfo['back_bottom_left'][0],
                   STableCaliInfo['back_bottom_left'][1])]

    if table_back[0] != (0, 0):
        check_1 = geometry.isInSquareTable(table_back,
                                           Point(RShoulder[0], RShoulder[1]))
        check_2 = geometry.isInSquareTable(table_back,
                                           Point(LShoulder[0], LShoulder[1]))
        iNTable = check_1 and check_2

    return iNTable


def HumanMainPostionExtract(TableHumanKPTimeList):

    TrackingMainPostionNP = []
    TrackingWholePostionNP = []

    for FrameKP in TableHumanKPTimeList:
        for humanKP in FrameKP:
            TrackingMainPostionNP.append([humanKP[3][0], humanKP[3][1]])
            TrackingWholePostionNP.append(humanKP)
    return np.array(TrackingMainPostionNP), np.array(TrackingWholePostionNP)


def SquareTrackingKpExtract(TableHumanKPTimeList):

    TrackingMainPostionNP = []
    TrackingWholePostionNP = []

    for FrameKP in TableHumanKPTimeList:
        for humanKP in FrameKP:

            RShoulder = np.array(humanKP[0][0:2])
            LShoulder = np.array(humanKP[1][0:2])

            TrackingMainPostionNP.append([(RShoulder[0] + LShoulder[0]) / 2,
                                          (RShoulder[1] + LShoulder[1]) / 2])
            TrackingWholePostionNP.append(humanKP)
    return np.array(TrackingMainPostionNP), np.array(TrackingWholePostionNP)


def DBscanCluster(TrackingMainPostionNP, ClusterEps, ClusterMinSample):

    db = DBSCAN(eps=ClusterEps,
                min_samples=ClusterMinSample).fit(TrackingMainPostionNP)
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


def KeenJudge(tableHumanKpCoorListHWthresh, TableAround, ETableCaliInfo):

    IntableHumanKpCoorList = []

    KneeCoorList = []

    for FinalHumanKpCoor in tableHumanKpCoorListHWthresh:

        RKnee = FinalHumanKpCoor[6]
        LKnee = FinalHumanKpCoor[7]
        if RKnee[2] > LKnee[2]:
            Knee = RKnee
        else:
            Knee = LKnee

        KneeCoorList.append(Knee)

        checkKnee = geometry.isInEllipseTable(
            ETableCaliInfo['x_center'] + TableAround[0],
            ETableCaliInfo['y_center'] + TableAround[1], Knee[0], Knee[1],
            ETableCaliInfo['long_axis'] * TableAround[2],
            TableAround[3] * ETableCaliInfo['short_axis'],
            ETableCaliInfo['theta'])
        if checkKnee is True:
            IntableHumanKpCoorList.append(FinalHumanKpCoor)
        else:
            print('knee not in Table Around, Human in this table.')

    return np.array(IntableHumanKpCoorList), np.array(KneeCoorList)


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
            print('Overlap is ' + str(OverLapPercentageList[-1]) +
                  '% Human not siting.')
        else:
            IntableHumanKpCoorList.append(FinalHumanKpCoor)
    return OverLapPercentageList, np.array(IntableHumanKpCoorList)


def OverLapPercentage_2(TableCali, TableAround, FinalHumanKpCoorList,
                        overlapthreshhold, BetweenElipseThreshold):

    main_ax = TableCali['long_axis']
    sub_ax = TableCali['short_axis']
    el_x = TableCali['x_center']
    el_y = TableCali['y_center']

    IntableHumanKpCoorList = []

    OverLapPercentageList = []
    BetweenElipPercentageList = []

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
        pointsbetweenellipsenum = 0

        for xindex in range(int(min_x), int(max_x + 1), int(jumpstepx)):
            for yindex in range(int(min_y), int(max_y + 1), int(jumpstepy)):
                if polygon.contains(Point(xindex, yindex)) is True:
                    pointstotalnum = pointstotalnum + 1

                    InElipsetable = geometry.isInEllipseTable(
                        el_x, el_y, xindex, yindex, main_ax, sub_ax,
                        TableCali['theta'])
                    if InElipsetable is True:
                        pointsinellipsenum = pointsinellipsenum + 1
                    else:

                        InAroundEllipse = geometry.isInEllipseTable(
                            TableAround[0] + el_x, TableAround[1] + el_y,
                            xindex, yindex, main_ax * TableAround[2],
                            TableAround[3] * sub_ax, TableCali['theta'])
                        if InAroundEllipse is True:
                            pointsbetweenellipsenum = pointsbetweenellipsenum + 1

        if pointstotalnum == 0:
            OverLapPercentageList.append(0.0)
        else:
            OverLapPercentageList.append(100 * pointsinellipsenum /
                                         pointstotalnum)

        if pointsbetweenellipsenum == 0:
            BetweenElipPercentageList.append(0.0)
        else:
            BetweenElipPercentageList.append(100 * pointsbetweenellipsenum /
                                             pointstotalnum)

        if OverLapPercentageList[-1] >= overlapthreshhold:
            print('Overlap is: ' + str(OverLapPercentageList[-1]) +
                  '%, too large. Human not siting.')
        elif BetweenElipPercentageList[-1] <= BetweenElipseThreshold:
            print('Between two Eliipse Percentage is: ' +
                  str(BetweenElipPercentageList[-1]) +
                  '%, too small. Human not siting.')
        else:
            IntableHumanKpCoorList.append(FinalHumanKpCoor)
    return OverLapPercentageList, BetweenElipPercentageList, np.array(
        IntableHumanKpCoorList)


def updateHumanNumTimeList(TableID, CurrHumannum, TableHumanNumTimeList,
                           JudgeLength):
    if len(TableHumanNumTimeList[str(TableID)]) != JudgeLength:
        TableHumanNumTimeList[str(TableID)].append(CurrHumannum)
    else:
        del TableHumanNumTimeList[str(TableID)][0]
        TableHumanNumTimeList[str(TableID)].append(CurrHumannum)
    return TableHumanNumTimeList


def updateDataBase(mysql_q, TableHumanNumTimeList, JudgeLength):
    for key, value in TableHumanNumTimeList.items():

        if len(value) == JudgeLength:
            if value[0] != value[1]:
                for i in range(1, JudgeLength):
                    if value[i] != value[-1]:
                        break
                msg_str = {
                    'areaID': key,
                    'timestamp': int(time.time() * 1000),
                    'num': value[-1],
                }
                mysql_q.publish(json.dumps(msg_str))
                print('DB updated')
                print(msg_str)

            else:
                continue
        else:
            continue


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
        check_1 = geometry.isInEllipseTable(el_x, el_y + 100, RHip[0], RHip[1],
                                            2.2 * main_ax, 3.5 * sub_ax,
                                            TableCali['theta'])
        check_2 = geometry.isInEllipseTable(el_x, el_y + 100, LHip[0], LHip[1],
                                            2.2 * main_ax, 3.5 * sub_ax,
                                            TableCali['theta'])
    elif TableID == 11:
        check_1 = geometry.isInEllipseTable(el_x, el_y + 100, RHip[0], RHip[1],
                                            2.2 * main_ax, 3 * sub_ax,
                                            TableCali['theta'])
        check_2 = geometry.isInEllipseTable(el_x, el_y + 100, LHip[0], LHip[1],
                                            2.2 * main_ax, 3 * sub_ax,
                                            TableCali['theta'])
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
