import time
import cv2
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import json

cameraIDList = [105, 106, 107, 108, 110, 112, 114, 119]

cut = {
    '105': [1000, 2440, 0, 1440, 90],
    '106': [0, 2560, 0, 1440, -90],
    '107': [900, 2340, 0, 1440, 90],
    '108': [80, 1520, 0, 1440, -90],
    '110': [0, 2560, 0, 1440, 0],
    '112': [0, 2560, 0, 1440, 0],
    '114': [800, 2240, 0, 1440, 0],
    '119': [400, 1840, 0, 1440, 0],
}

tableTamplate = {
    "center": [0, 0],
    "tableID": 0,
    "back_top_left": [0, 0],
    "back_top_right": [0, 0],
    "front_top_left": [0, 0],
    "front_top_right": [0, 0],
    "back_bottom_left": [0, 0],
    "back_bottom_right": [0, 0],
    "front_bottom_left": [0, 0],
    "front_bottom_right": [0, 0]
}

cameracalibration = {}


def RotateClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    return new_img


def RotateAntiClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 0)
    return new_img


def image_preprocess(image, cut_info):
    #print(camera_info)
    cropped = image[cut_info[2]:cut_info[3], cut_info[0]:cut_info[1]]
    if cut_info[4] == 90:
        new_image = RotateClockWise90(cropped)
    elif cut_info[4] == -90:
        new_image = RotateAntiClockWise90(cropped)
    else:
        new_image = cropped
    return new_image


if __name__ == '__main__':

    global inputX
    global inputY

    for cameraid in cameraIDList:

        imagepath = '../data/' + str(cameraid) + '.jpg'
        img = cv2.imread(imagepath)
        img_cut = image_preprocess(img, cut[str(cameraid)])
        cv2.imshow(str(cameraid), img_cut)
        cv2.waitKey(0)
        plt.imshow(img_cut, cmap=plt.get_cmap("gray"))

        print("input table num:")
        tableNum = input()

        cameracali = []
        for i in range(int(tableNum)):

            tableTamplate = {
                "center": [0, 0],
                "tableID": 0,
                "back_top_left": [0, 0],
                "back_top_right": [0, 0],
                "front_top_left": [0, 0],
                "front_top_right": [0, 0],
                "back_bottom_left": [0, 0],
                "back_bottom_right": [0, 0],
                "front_bottom_left": [0, 0],
                "front_bottom_right": [0, 0]
            }

            print("input cuur calibration tableID:")
            tableId = input()
            tableTamplate['tableID'] = int(tableId)


            print("center(y/n):")
            iscenter = input()
            if iscenter == 'y':
                print("select center on image:")
                pos = plt.ginput(1)
                print(pos)
                tableTamplate['center'] = [int(pos[0][0]), int(pos[0][1])]

            print("front(y/n):")
            isfront = input()
            if isfront == 'y':
                print("select front")
                pos = plt.ginput(4)
                print(pos)
                tableTamplate['front_top_left'] = [int(pos[0][0]), int(pos[0][1])]
                tableTamplate['front_top_right'] = [int(pos[1][0]), int(pos[1][1])]
                tableTamplate['front_bottom_right'] = [
                    int(pos[2][0]), int(pos[2][1])
                ]
                tableTamplate['front_bottom_left'] = [
                    int(pos[3][0]), int(pos[3][1])
                ]

            cameracali.append(tableTamplate)

        print(cameracali)
        cameracalibration[cameraid] = cameracali
        pprint(cameracalibration)
        cv2.destroyAllWindows()


    with open("record.json", "w") as f:
        json.dump(cameracalibration, f)
        print("ww...")


    for cameraid in cameraIDList:

        imagepath = './data/' + str(cameraid) + '.jpg'
        img = cv2.imread(imagepath)
        img_cut = image_preprocess(img, cut[str(cameraid)])
        cv2.imshow(str(cameraid), img_cut)
        cv2.waitKey(0)
        plt.imshow(img_cut, cmap=plt.get_cmap("gray"))

        print("input table num:")
        tableNum = input()

        cameracali = []
        for i in range(int(tableNum)):

            print("input cuur calibration tableID:")
            tableId = input()
            for j in range(int(tableNum)):
                if cameracalibration[cameraid][j]['tableID'] == int(tableId):
                    currindex = j


            print("center(y/n):")
            iscenter = input()
            if iscenter == 'y':
                print("select center on image:")
                pos = plt.ginput(1)
                print(pos)
                cameracalibration[cameraid][currindex]['center'] = [int(pos[0][0]), int(pos[0][1])]

            print("back(y/n):")
            isback = input()
            if isback == 'y':
                print("select back")
                pos = plt.ginput(4)
                print(pos)
                cameracalibration[cameraid][currindex]['back_top_left'] = [int(pos[0][0]), int(pos[0][1])]
                cameracalibration[cameraid][currindex]['back_top_right'] = [int(pos[1][0]), int(pos[1][1])]
                cameracalibration[cameraid][currindex]['back_bottom_right'] = [int(pos[2][0]), int(pos[2][1])]
                cameracalibration[cameraid][currindex]['back_bottom_left'] = [int(pos[3][0]), int(pos[3][1])]

        cv2.destroyAllWindows()
    pprint(cameracalibration)

    with open("record.json", "w") as f:
        json.dump(cameracalibration, f)
        print("ww...")
