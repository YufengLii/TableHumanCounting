import cv2
import math
from modules.dbutil import MySQLPlugin
import json


def get_cali_data_from_file(working_cameraID):

    with open('./data/EllipseCali.txt', "r") as f:
        cameras_calibration_data = json.load(f)
        print('read done')

    working_cameras_cpture_info = []
    working_cameras = []
    for camera in cameras_calibration_data:
        if camera['cameraID'] not in working_cameraID:
            continue
        else:
            working_cameras_cpture_info.append({
                'cameraID':
                camera['cameraID'],
                'url':
                camera['url'],
                'calibration_keypoint':
                camera['calibration_keypoint']
            })
            working_cameras.append(camera)

    camera_info_lists = {}
    for data_cali in working_cameras_cpture_info:
        dict_cali = {
            str(data_cali['cameraID']): data_cali['calibration_keypoint']
        }
        camera_info_lists.update(dict_cali)
    url_lists = {}
    for data_cali in working_cameras_cpture_info:
        url_dict = {str(data_cali['cameraID']): data_cali['url']}
        url_lists.update(url_dict)

    return camera_info_lists, url_lists


def get_cali_data(working_cameraID):
    db = MySQLPlugin()
    cameras_calibration_data = db.query_camera_keypoints()

    # with open('./cali.txt', "w") as f:
    #     json.dump(cameras_calibration_data, f)
    #     print('write done')

    working_cameras_cpture_info = []
    working_cameras = []
    for camera in cameras_calibration_data:
        if camera['cameraID'] not in working_cameraID:
            continue
        else:
            working_cameras_cpture_info.append({
                'cameraID':
                camera['cameraID'],
                'url':
                camera['url'],
                'calibration_keypoint':
                camera['calibration_keypoint']
            })
            working_cameras.append(camera)

    camera_info_lists = {}
    for data_cali in working_cameras_cpture_info:
        dict_cali = {
            str(data_cali['cameraID']): data_cali['calibration_keypoint']
        }
        camera_info_lists.update(dict_cali)
    url_lists = {}
    for data_cali in working_cameras_cpture_info:
        url_dict = {str(data_cali['cameraID']): data_cali['url']}
        url_lists.update(url_dict)

    return camera_info_lists, url_lists


def drawSquareTableDebug(image, TrackingKp, lables):

    track_colors = [(0, 0, 255), (0, 255, 0), (255, 127, 255), (255, 0, 0),
                    (255, 255, 0), (127, 127, 255), (255, 0, 255),
                    (127, 0, 255), (127, 0, 127), (127, 10, 255),
                    (0, 255, 127)]

    for i in range(TrackingKp.shape[0]):
        # print(TrackingMainPostionNP[i])
        # pprint(lables[i])
        # 画出用于聚类的时间序列关键点
        cv2.circle(image, (int(
            TrackingKp[i][0]), int(TrackingKp[i][1])), 5,
                   track_colors[lables[i] + 1], -1)
        # 画出聚类分类结果
        cv2.putText(image, str(lables[i]),
                    (int(TrackingKp[i][0] + 30),
                     int(TrackingKp[i][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, track_colors[lables[i] + 1],
                    4)
    return image




def FinalMainBodyRect(UseUserColor, image, FinalHumanKpCoorList):

    for FinalHumanKpCoor in FinalHumanKpCoorList:
        RShoulderFinalCoor = FinalHumanKpCoor[4]
        LShoulderFinalCoor = FinalHumanKpCoor[5]
        MidShoulderFinalCoor = (RShoulderFinalCoor + LShoulderFinalCoor) / 2
        MidFinalCoor = FinalHumanKpCoor[3]
        XYAddRight = RShoulderFinalCoor - MidShoulderFinalCoor
        XYAddLeft = LShoulderFinalCoor - MidShoulderFinalCoor

        LHipDraw = MidFinalCoor + XYAddLeft
        RHipDraw = MidFinalCoor + XYAddRight

        drawRectUseUserColor(
            image, UseUserColor,
            tuple([int(RShoulderFinalCoor[0]),
                   int(RShoulderFinalCoor[1])]),
            tuple([int(LShoulderFinalCoor[0]),
                   int(LShoulderFinalCoor[1])]),
            tuple([int(LHipDraw[0]), int(LHipDraw[1])]),
            tuple([int(RHipDraw[0]), int(RHipDraw[1])]))

    return image


def drawEllipseCalibtarion(ETableCaliInfo, TableAround, CurrFrame):
    cv2.ellipse(
        CurrFrame,
        ((int)(ETableCaliInfo['x_center']), (int)(ETableCaliInfo['y_center'])),
        (int(ETableCaliInfo['long_axis']), int(ETableCaliInfo['short_axis'])),
        math.degrees(ETableCaliInfo['theta']), 0, 360, (0, 255, 0), 3)

    cv2.ellipse(CurrFrame,
                ((int)(ETableCaliInfo['x_center'] +
                       TableAround[str(ETableCaliInfo['tableID'])][0]),
                 (int)(ETableCaliInfo['y_center'] +
                       TableAround[str(ETableCaliInfo['tableID'])][1])),
                (int(ETableCaliInfo['long_axis'] *
                     TableAround[str(ETableCaliInfo['tableID'])][2]),
                 int(ETableCaliInfo['short_axis'] *
                     TableAround[str(ETableCaliInfo['tableID'])][3])),
                math.degrees(ETableCaliInfo['theta']), 0, 360, (0, 255, 0), 3)
    cv2.putText(CurrFrame, "Table " + (str)(ETableCaliInfo['tableID']),
                ((int)(ETableCaliInfo['x_center']),
                 (int)(ETableCaliInfo['y_center'])), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    return CurrFrame


def drawEllipseDebug(CurrFrame, lables, overlapthreshhold,
                     FinalHumanKpCoorList, OverLapPercentageList,
                     BetweenElipPercentageList, tableHumanKpCoorListHWthresh,
                     TrackingMainPostionNP, ETableCaliInfo, FinalKneeCoorList):

    track_colors = [(0, 0, 255), (0, 255, 0), (255, 127, 255), (255, 0, 0),
                    (255, 255, 0), (127, 127, 255), (255, 0, 255),
                    (127, 0, 255), (127, 0, 127), (127, 10, 255),
                    (0, 255, 127)]

    for i in range(TrackingMainPostionNP.shape[0]):
        # print(TrackingMainPostionNP[i])
        # pprint(lables[i])
        # 画出用于聚类的时间序列关键点
        cv2.circle(CurrFrame, (int(
            TrackingMainPostionNP[i][0]), int(TrackingMainPostionNP[i][1])), 5,
                   track_colors[lables[i] + 1], -1)
        # 画出聚类分类结果
        cv2.putText(CurrFrame, str(lables[i]),
                    (int(TrackingMainPostionNP[i][0] + 30),
                     int(TrackingMainPostionNP[i][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, track_colors[lables[i] + 1],
                    4)

    FinalMainBodyRect(track_colors[-1], CurrFrame, FinalHumanKpCoorList)

    for i in range(len(OverLapPercentageList)):

        RShoulderFinalCoor = FinalHumanKpCoorList[i][4]
        LShoulderFinalCoor = FinalHumanKpCoorList[i][5]
        MidShoulderFinalCoor = (RShoulderFinalCoor + LShoulderFinalCoor) / 2
        MidFinalCoor = FinalHumanKpCoorList[i][3]
        DrawPercentage = (MidShoulderFinalCoor + MidFinalCoor) / 2
        cv2.putText(CurrFrame,
                    str(int(OverLapPercentageList[i])) + "%",
                    ((int)(DrawPercentage[0]), (int)(DrawPercentage[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, track_colors[-4], 4)

    for i in range(len(BetweenElipPercentageList)):

        RShoulderFinalCoor = FinalHumanKpCoorList[i][4]
        LShoulderFinalCoor = FinalHumanKpCoorList[i][5]
        MidShoulderFinalCoor = (RShoulderFinalCoor + LShoulderFinalCoor) / 2
        MidFinalCoor = FinalHumanKpCoorList[i][3]
        DrawPercentage = (MidShoulderFinalCoor + MidFinalCoor) / 2
        cv2.putText(CurrFrame,
                    str(int(BetweenElipPercentageList[i])) + "%",
                    ((int)(DrawPercentage[0]), (int)(DrawPercentage[1] + 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, track_colors[-5], 4)

    FinalMainBodyRect(track_colors[-3], CurrFrame,
                      tableHumanKpCoorListHWthresh)

    for i in range(FinalKneeCoorList.shape[0]):
        cv2.circle(
            CurrFrame,
            (int(FinalKneeCoorList[i][0]), int(FinalKneeCoorList[i][1])), 10,
            track_colors[lables[i] + 1], 2)

    # if ETableCaliInfo['tableID'] == 12:
    #     cv2.ellipse(CurrFrame, ((int)(ETableCaliInfo['x_center'] + 25),
    #                             (int)(ETableCaliInfo['y_center'] + 120)),
    #                 (int(2.2 * ETableCaliInfo['long_axis']),
    #                  int(2.2 * ETableCaliInfo['short_axis'])),
    #                 math.degrees(ETableCaliInfo['theta']), 0, 360, (0, 255, 0),
    #                 3)

    cv2.putText(
        CurrFrame, "Table" + str(ETableCaliInfo['tableID']) +
        ' Human num is: ' + str(tableHumanKpCoorListHWthresh.shape[0]),
        ((int)(ETableCaliInfo['x_center']),
         (int)(ETableCaliInfo['y_center'] - 50)), cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0, 0, 255), 2)

    return CurrFrame


def draw_rect(img, pt1, pt2, pt3, pt4):
    point_color = (0, 255, 0)  # BGR
    thickness = 4
    lineType = 4
    pt1 = ((int)(pt1[0]), (int)(pt1[1]))
    pt2 = ((int)(pt2[0]), (int)(pt2[1]))
    pt3 = ((int)(pt3[0]), (int)(pt3[1]))
    pt4 = ((int)(pt4[0]), (int)(pt4[1]))

    cv2.line(img, pt1, pt2, point_color, thickness, lineType)
    cv2.line(img, pt2, pt3, point_color, thickness, lineType)
    cv2.line(img, pt3, pt4, point_color, thickness, lineType)
    cv2.line(img, pt4, pt1, point_color, thickness, lineType)
    return img


def drawRectUseUserColor(img, usercolor, pt1, pt2, pt3, pt4):
    # BGR
    thickness = 4
    lineType = 4
    pt1 = ((int)(pt1[0]), (int)(pt1[1]))
    pt2 = ((int)(pt2[0]), (int)(pt2[1]))
    pt3 = ((int)(pt3[0]), (int)(pt3[1]))
    pt4 = ((int)(pt4[0]), (int)(pt4[1]))

    cv2.line(img, pt1, pt2, usercolor, thickness, lineType)
    cv2.line(img, pt2, pt3, usercolor, thickness, lineType)
    cv2.line(img, pt3, pt4, usercolor, thickness, lineType)
    cv2.line(img, pt4, pt1, usercolor, thickness, lineType)
    return img


def draw_table_info(image, camera_info):

    camera_table_calibration_info = camera_info['calibrations']
    for table_info in camera_table_calibration_info:
        if table_info['shape'] is 'ellipses':

            # ellipse_table_y_edge_min = (int)(table_info['y_center'] -
            #                                  table_info['short_axis'])
            # ellipse_table_y_edge_max = (int)(table_info['y_center'] +
            #                                  table_info['short_axis'])
            # ellipse_table_x_edge_min = (int)(table_info['x_center'] -
            #                                  table_info['long_axis'])
            # ellipse_table_x_edge_max = (int)(table_info['x_center'] +
            #                                  table_info['long_axis'])

            # cv2.line(image, (1, ellipse_table_y_edge_min),
            #          (2559, ellipse_table_y_edge_min), (255, 0, 0), 4, 4)
            # cv2.line(image, (1, ellipse_table_y_edge_max),
            #          (2559, ellipse_table_y_edge_max), (255, 0, 0), 4, 4)
            # cv2.line(image, (ellipse_table_x_edge_min, 1),
            #          (ellipse_table_x_edge_min, 1439), (255, 0, 0), 4, 4)
            # cv2.line(image, (ellipse_table_x_edge_max, 1),
            #          (ellipse_table_x_edge_max, 1439), (255, 0, 0), 4, 4)
            # cv2.line(image, (200, 300),
            #          ((int)(200 + table_info['short_axis']), 300), (0, 255, 0),
            #          4, 4)
            # cv2.line(image, (200, 500),
            #          ((int)(200 + table_info['long_axis']), 500), (0, 255, 0),
            #          4, 4)

            # cv2.ellipse(image, ((int)(table_info['x_center']),
            #                     (int)(table_info['y_center'])),
            #             (int(0.8 * table_info['long_axis']),
            #              int(0.8 * table_info['short_axis'])),
            #             math.degrees(table_info['theta']), 0, 360, (0, 255, 0),
            #             3)
            cv2.ellipse(
                image,
                ((int)(table_info['x_center']), (int)(table_info['y_center'])),
                (int(table_info['long_axis']), int(table_info['short_axis'])),
                math.degrees(table_info['theta']), 0, 360, (0, 255, 0), 3)
            cv2.putText(image, "Table " + (str)(table_info['tableID']),
                        ((int)(table_info['x_center']),
                         (int)(table_info['y_center'])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            #print(table_info['center'][0])
            #print(table_info['center'][1])
            cv2.putText(image, "Table " + (str)(table_info['tableID']),
                        ((int)(table_info['center'][0]),
                         (int)(table_info['center'][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            draw_rect(image, tuple(table_info['top_left']),
                      tuple(table_info['top_right']),
                      tuple(table_info['bottom_right']),
                      tuple(table_info['bottom_left']))
            draw_rect(image, tuple(table_info['back_top_left']),
                      tuple(table_info['back_top_right']),
                      tuple(table_info['back_bottom_right']),
                      tuple(table_info['back_bottom_left']))
            draw_rect(image, tuple(table_info['front_top_left']),
                      tuple(table_info['front_top_right']),
                      tuple(table_info['front_bottom_right']),
                      tuple(table_info['front_bottom_left']))
            #draw_rect(image,)
            #print('draw rectangle')
    return image


def draw_calibration_info(image, curr_camera_calibration_info):

    for table_info in curr_camera_calibration_info:
        if table_info['shape'] == 'ellipses':

            cv2.ellipse(
                image,
                ((int)(table_info['x_center']), (int)(table_info['y_center'])),
                (int(table_info['long_axis']), int(table_info['short_axis'])),
                math.degrees(table_info['theta']), 0, 360, (0, 255, 0), 3)
            cv2.putText(image, "Table " + (str)(table_info['tableID']),
                        ((int)(table_info['x_center']),
                         (int)(table_info['y_center'])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:

            #print(table_info['center'][0])
            #print(table_info['center'][1])
            cv2.putText(image, "Table " + (str)(table_info['tableID']),
                        ((int)(table_info['center'][0]),
                         (int)(table_info['center'][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            draw_rect(image, tuple(table_info['top_left']),
                      tuple(table_info['top_right']),
                      tuple(table_info['bottom_right']),
                      tuple(table_info['bottom_left']))
            draw_rect(image, tuple(table_info['back_top_left']),
                      tuple(table_info['back_top_right']),
                      tuple(table_info['back_bottom_right']),
                      tuple(table_info['back_bottom_left']))
            draw_rect(image, tuple(table_info['front_top_left']),
                      tuple(table_info['front_top_right']),
                      tuple(table_info['front_bottom_right']),
                      tuple(table_info['front_bottom_left']))
            #draw_rect(image,)
            #print('draw rectangle')
    return image


def RotateClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    return new_img


def RotateAntiClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 0)
    return new_img


def image_preprocess(image, camera_info):
    #print(camera_info)
    cropped = image[camera_info[2]:camera_info[3],
                    camera_info[0]:camera_info[1]]
    if camera_info[4] == 90:
        new_image = RotateClockWise90(cropped)
    elif camera_info[4] == -90:
        new_image = RotateAntiClockWise90(cropped)
    else:
        new_image = cropped
    return new_image


def drawSquareTableInfo(image, STableCaliInfo):

    if STableCaliInfo['back_top_left'] != [0, 0]:
        draw_rect(image, tuple(STableCaliInfo['back_top_left']),
                  tuple(STableCaliInfo['back_top_right']),
                  tuple(STableCaliInfo['back_bottom_right']),
                  tuple(STableCaliInfo['back_bottom_left']))
    if STableCaliInfo['front_top_left'] != [0, 0]:
        draw_rect(image, tuple(STableCaliInfo['front_top_left']),
                  tuple(STableCaliInfo['front_top_right']),
                  tuple(STableCaliInfo['front_bottom_right']),
                  tuple(STableCaliInfo['front_bottom_left']))

    return image
