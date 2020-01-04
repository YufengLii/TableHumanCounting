__author__ = 'werewolfLu'

import cv2
import json
import time
import base64
import requests
try:
    from modules.mlogger import mlogger as logging
except ImportError as e:
    import logging
from numpy import ndarray

HOST = 'http://192.168.31.161:8300'


def img_encodeb64(img):
    if isinstance(img, str) and os.path.isfile(img):
        return str(base64.b64encode(cv2.imencode('.jpg', cv2.imread(img))[1].tostring()), 'utf-8')
    elif isinstance(img, ndarray):
        return str(base64.b64encode(cv2.imencode('.jpg', img)[1].tostring()), 'utf-8')


json_dumps = json.dumps


class BAIDUFace(object):

    def __init__(self):
        super(BAIDUFace, self).__init__()

    @staticmethod
    def __request(url, data=None, json=None, timeout=3, retries=3):

        attempts = 1
        while attempts < retries:
            try:
                if data is None and json is None:
                    return requests.get(url, timeout=timeout).json()
                else:
                    return requests.post(url, data=data, json=json, timeout=timeout).json()
                # elif json is not None:
                    # return requests.post(url, data=json_dumps(json1))
            except Exception as e:
                logging.error("URL Request Error: %s", e)
                attempts += 1

    def _request_api(self, url, data=None, json=None, timeout=3, retries=3):
        content = self.__request(url, data=data, json=json, timeout=timeout, retries=retries)
        if content and isinstance(content, dict) and content['error_code'] == 0:
            # print(content['result'])
            return content['result']
        else:
            logging.error("[ERROR] %s: %s", url.rsplit('v3', 1)[-1], content)
            return -1

    def create_group(self, group, appid='default'):
        api = HOST + '/face-api/v3/group/add'

        data = {
            'appid': appid,
            'group_id': group
        }

        return self._request_api(api, json=data)

    def get_appid(self):
        api = HOST + '/face-api/v3/app/list'
        return self._request_api(api)

    # face detect
    # sample: face_detect(img_encodeb64(./test_image.jpg))
    # face_field: 'age,beauty,expression,face_shape,gender,glasses,landmark,landmark150,race,quality,eye_status,emotion,face_type'
    # face_type: 'LIVE' or 'IDCARD' or 'WATERMARK' or 'CERT'
    # liveness_control: 'NONE' or 'LOW' or 'NORMAL' or 'HIGH'
    def face_detect(
            self,
            img,
            appid='default',
            image_type='BASE64',
            face_field='face_probability,age,gender,glasses,eye_status,quality,expression,emotion,angel,feature',
            max_face_num=10,
            face_type='LIVE',
            liveness_control='NONE'):
        api = HOST + '/face-api/v3/face/detect'
        data = {
            'appid': appid,
            'image': img,
            'image_type': image_type,
            'face_field': face_field,
            'max_face_num': max_face_num,
            'face_type': face_type,
            'liveness_control': liveness_control
        }
        return self._request_api(api, json=data)

    def face_identity(
            self,
            img,
            appid='default',
            image_type='BASE64',
            group_id_list='member1',
            quality_control='NONE',
            liveness_control='NONE',
            user_id=None,
            max_user_num=1):
        api = HOST + '/face-api/v3/face/identify'
        data = {
            'appid': appid,
            'image': img,
            'image_type': image_type,
            'group_id_list': group_id_list,
            'quality_control': quality_control,
            'liveness_control': liveness_control,
            'user_id': user_id,
            'max_user_num': max_user_num
        }
        # 尝试6次，每次间隔0.2秒
        retries = 6
        attempt_i = 1
        sleep_time = 0.2
        while attempt_i <= retries:
            time.sleep(sleep_time)
            content = self.__request(api, json=data)
            if content and isinstance(content, dict) and content['error_code'] == 0:
                return content['result']
            elif isinstance(content, dict) and content['error_code'] == 222209:
                attempt_i += 1
            else:
                return -1
        return -1


    # add face to repository
    # sample: face_add(img_encodeb64('./何斌.jpg'), 'uid_1', '何斌')
    def add_user(
            self,
            image,
            user_id,
            user_name,
            appid='default',
            image_type='BASE64',
            group_id='member1',
            quality_control='NONE',
            liveness_control='NONE',
            action_type='APPEND'):
        api = HOST + '/face-api/v3/face/add'
        data = {
            'appid': appid,
            'image': image,
            'image_type': image_type,
            'group_id': group_id,
            'user_id': user_id,
            'user_info': user_name,
            'quality_control': quality_control,
            'liveness_control': liveness_control,
            'action_type': action_type}
        return self._request_api(api, json=data)

    def update_user(
            self,
            image,
            user_id,
            user_name,
            appid='default',
            image_type='BASE64',
            group_id='member1',
            quality_control='NONE',
            liveness_control='NONE',
            action_type='UPDATE'):
        api = HOST + '/face-api/v3/face/update'
        data = {
            'appid': appid,
            'image': image,
            'image_type': image_type,
            'group_id': group_id,
            'user_id': user_id,
            'user_info': user_name,
            'quality_control': quality_control,
            'liveness_control': liveness_control,
            'action_type': action_type
        }
        return self._request_api(api, json=data)

    def remove_user_face(self, user_id, face_token, group_id='member1', appid='default'):
        api = HOST + '/face-api/v3/face/delete'
        data = {
            'appid': appid,
            'user_id': user_id,
            'group_id': group_id,
            'face_token': face_token
        }
        return self._request_api(api, json=data)

    def copy_user(self, user_id, src_group_id, dst_group_id, appid='default'):
        api = HOST + '/face-api/v3/user/copy'
        data = {
            'appid': appid,
            'user_id': user_id,
            'src_group_id': src_group_id,
            'dst_group_id': dst_group_id
        }
        return self._request_api(api, json=data)

    def get_user_list(self, appid='default', group_id='member1', start=0, length=1000):
        api = HOST + '/face-api/v3/user/list'
        data = {
            'appid': appid,
            'group_id': group_id,
            'start': start,
            'length': length
        }
        return self._request_api(api, json=data)

    def get_user_info(self, user_id, group_id='member1', appid='default'):
        api = HOST + '/face-api/v3/user/get'
        data = {
            'appid': appid,
            'user_id': user_id,
            'group_id': group_id
        }
        return self._request_api(api, json=data)

    def del_user(self, user_id, group_id='member1', appid='default'):
        api = HOST + '/face-api/v3/user/delete'
        data = {
            'appid': appid,
            'user_id': user_id,
            'group_id': group_id
        }
        return self._request_api(api, json=data)

    def get_group_num(self, appid='default'):
        api = HOST + '/face-api/facehub/statapp'
        data = {
            'op_app_id_list': appid
        }
        return self._request_api(api, json=data)

    def get_group_stat(self, group_id_list, appid='default'):
        api = HOST + '/face-api/facehub/statgroup'

        data = {
            'op_app_id': appid,
            'group_id_list': group_id_list
        }
        return self._request_api(api, json=data)

    def del_group(self, group_id, appid='default'):
        api = HOST + '/face-api/v3/group/delete'
        data = {
            'appid': appid,
            'group_id': group_id
        }
        return self._request_api(api, json=data)

    def get_group_list(self, appid='default', start=0, length=100):
        api = HOST + '/face-api/v3/group/list'
        data = {
            'appid': appid,
            'start': start,
            'length': length
        }
        return self._request_api(api, json=data)


if __name__ == '__main__':

    import os
    from pprint import pprint

    baidu = BAIDUFace()
    print(baidu.face_identity('8b0e1e389ff83ea0016eba30563745e5', image_type='FACE_TOKEN'))
    # print(baidu.get_appid())
    #
    # # r = baidu.face_identity(img_encodeb64('../repository/李杰/李杰.png'))
    # # pprint(r)
    #
    # r = baidu.get_user_list()
    # pprint(r)
    #
    # r = baidu.get_user_info('798e2110e73011e99281e0d55eee5246')
    # pprint(r)
    #
    # r = baidu.get_group_num()
    # pprint(r)
    #
    # r = baidu.get_group_stat('member1')
    # pprint(r)
    #
    # print("end")
