"""

@author: JayLee

@contact: JayLee@tongji.edu.cn

@software: PyCharm

@file: db_api.py

@time: 2019/10/3 下午9:25

@desc:

"""
import re
import json
import pymysql.cursors

from modules.globals import DBCONF


DEFAULT = {
    'server': '192.168.31.243',
    'user': 'root',
    'pwd':  'root@1234',
    'db':   'cvtest1',
}


class MySQLPlugin(object):
    KEEP_PERIOD = 60 * 60 * 24 * 365

    def __init__(self, **kwargs):
        super(MySQLPlugin, self).__init__()
        self.MySQL_SERVER = kwargs.get('host') or DEFAULT['server']
        self.MySQL_PORT = kwargs.get('port')
        self.db = kwargs.get('db') or DEFAULT['db']
        self.user = kwargs.get('user') or DEFAULT['user']
        self.password = kwargs.get('password') or DEFAULT['pwd']
        self.location = kwargs.get('location')

        self.protocol = "MySQL"
        self.connection = None
        self.__iconnected = False

    def connect(self):
        attempts, max_attempts = 0, 5
        if self.__iconnected: return
        while attempts < max_attempts:
            try:
                attempts += 1
                self.connection = pymysql.connect(
                    host=self.MySQL_SERVER,
                    user=self.user,
                    password=self.password,
                    db=self.db,
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor,
                    autocommit=True)
                self.__iconnected = True
                return True
            except Exception as e:
                print(e)
                print("[MYSQL] CONNECT FAILED!", e)
                self.close()
        return False

    def close(self):
        if self.__iconnected is False: return
        try:
            self.connection.close()
            self.__iconnected = False
        except Exception as e:
            print(e)

    def __query(self, sql, params=None, num=None):

        if not self.__iconnected: self.connect()

        self.connection.ping(reconnect=True)

        with self.connection.cursor() as cursor:
            try:
                if params is not None:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)

                if num == 1:
                    return cursor.fetchone()
                elif num == 'all' or num is None:
                    return cursor.fetchall()
                elif isinstance(num, int):
                    return cursor.fetchmany(num)
                else:
                    print("[DB] Query: wrong argument", num, sql)
                    return None

            except pymysql.err.ProgrammingError as e:
                print("[DB] QUERY ProgrammingError", e, sql)
            except pymysql.err.IntegrityError as e:
                print("[DB] QUERY IntegrityError", e, sql)
            except Exception as e:
                print("[DB] QUERY Exception", e, sql)
            cursor.close()

    def __execute(self, sql, params=None):
        if not self.__iconnected: self.connect()
        self.connection.ping(reconnect=True)
        errno = 0
        with self.connection.cursor() as cursor:
            try:
                if params is not None:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                return self.connection.commit()
            except pymysql.err.ProgrammingError as e:
                print("[DB] COMMIT ProgrammingError", e, sql)
                self.connection.rollback()
                # errno = re.search(r'\((\d+),\s*[\"\'\.\\\s\w]+\)', str(e)).groups()[0]
                errno = str(e).split(',', 1)[0].replace('(', '')
            except pymysql.err.IntegrityError as e:
                print("[DB] COMMIT IntegrityError", e, sql)
                self.connection.rollback()
            except Exception as e:
                print("[DB] COMMIT Exception", e, sql)
                self.connection.rollback()
            finally:
                cursor.close()
                return errno

    def add_user(self, user_id, user_info, group_id, app_id):
        sql = "INSERT INTO Member (memberID, name, faceDB, faceGroup, faceID) VALUES (%(user_id)s, %(user_info)s, " \
              "%(app_id)s, %(group_id)s, %(user_id)s);"
        self.__execute(sql, dict(user_id=user_id, user_info=user_info, group_id=group_id, app_id=app_id))

    def del_user(self, user_id):
        sql = "DELETE FROM Member WHERE memberID=%s;"
        params = (user_id,)
        self.__execute(sql, params)

    def query_user(self, memberID=None, name=None):
        sql = "SELECT * FROM Member;"
        if memberID is not None:
            sql = sql[:-1] + ' WHERE memberID="{}";'.format(memberID)
        elif name is not None:
            sql = sql[:-1] + ' WHERE name="{}";'.format(name)
        return self.__query(sql)

    def query_camera(self, cameraID=None):
        sql = "SELECT cameraID, name, stream as url, spot, resolution, frame_rate, calibration FROM Camera;"

        if cameraID is not None:
            sql = sql[:-1] + ' WHERE cameraID="{};"'.format(cameraID)

        rt = self.__query(sql)
        if not rt: return []
        for item in rt:
            if item.get('calibration'):
                item['calibration'] = json.loads(item['calibration'])
        return rt

    def query_camera_keypoints(self, cameraID=None):
        sql = "SELECT cameraID, name, stream as url, spot, resolution, frame_rate, calibration_keypoint FROM Camera;"

        if cameraID is not None:
            sql = sql[:-1] + ' WHERE cameraID="{};"'.format(cameraID)

        rt = self.__query(sql)
        if not rt: return []
        for item in rt:
            if item.get('calibration_keypoint'):
                item['calibration_keypoint'] = json.loads(item['calibration_keypoint'])
        return rt


    def query_customer(self, customerID=None):
        sql = 'SELECT * FROM Customer;'
        if customerID is not None:
            sql = sql[:-1] + ' WHERE customerID="{}";'.format(customerID)
        return self.__query(sql)

    def query_customer_within_memberinfo(self, customerID=None, memberID=None):
        sql = 'SELECT Customer.*, Member.* FROM Customer join Member on Customer.faceID=Member.faceID;'
        if customerID is not None:
            sql = sql[:-1] + ' WHERE customerID="{}";'.format(customerID)
        elif memberID is not None:
            sql = sql[:-1] + ' WHERE memberID="{}";'.format(memberID)
        return self.__query(sql)

    def update_headcount(self, targetID, headcount, timestamp):
        sql = 'INSERT INTO {table} (timestamp, headcount) VALUES ( %(timestamp)s, %(headcount)s )'\
            .format(table="TargetArea{}Occupancy".format(targetID))
        e = self.__execute(sql, dict(timestamp=timestamp, headcount=headcount))
        if e and e == 1146 or e == '1146':
            self.__execute("CREATE TABLE IF NOT EXISTS {} LIKE TargetAreaXOccupancy;"\
                           .format("TargetArea{}Occupancy".format(targetID)))
            self.__execute(sql, dict(timestamp=timestamp, headcount=headcount))

    def query_track_tbl(self, cameraID=None, trackID=None):
        sql = 'SHOW tables;'
        if cameraID is not None and trackID is not None:
            sql = sql[:-1] + " LIKE 'Camera{cid}_Track{tid}';".format(cid=cameraID, tid=trackID)
        elif cameraID is not None:
            sql = sql[:-1] + " LIKE 'Camera{cid}_Track%';".format(cid=cameraID)
        elif trackID is not None:
            sql = sql[:-1] + " LIKE 'Camera%_Track{tid}';".format(tid=trackID)
        else:
            sql = sql[:-1] + " LIKE 'Camera%_Track%';"
            # sql = sql[:-1] + " LIKE 'TargetArea%';"

        l = self.__query(sql)
        return [list(d.values())[0] for d in l]

    def get_camera_tracks(self, cameraID=None):
        l = self.query_track_tbl(cameraID=cameraID)
        patt = r'Camera([\w\d]+)_Track([\w\d]+)'
        tracklist = {}
        for item in l:
            r = re.match(patt, item)
            if r:
                cid, tid = r.groups()
                if cid not in tracklist:
                    tracklist[cid] = [tid]
                else:
                    tracklist[cid].append(tid)

        if cameraID: return {cameraID: tracklist.get(cameraID,[])}
        return tracklist

    def get_trajectory(self, cameraID, trackID, trange=None):
        sql = 'SELECT timestamp, st_astext(coordinates), st_astext(rect), area FROM Camera{cid}_Track{tid};'\
            .format(cid=cameraID, tid=trackID)
        if trange is not None:
            sql = sql[:-1] + ' WHERE timestamp BETWEEN {t1} and {t2}'.format(t1=trange[0], t2=trange[1])
        rt = self.__query(sql)
        for item in rt:
            if item.get('st_astext(coordinates)'):
                try:
                    r = re.match(r'POINT\(([\d\s,\.]+)\)', item['st_astext(coordinates)'])
                    p = re.split(r',|\s', r.group(1))
                    item['coordinates'] = [float(p[0]), float(p[1])]
                    del item['st_astext(coordinates)']
                except Exception as e:
                    print(item['astext(coordinates)'])
                    print('[Exception] get_trajectory POINT parse error', e)
            if item.get('st_astext(rect)'):
                try:
                    r = re.match(r'POLYGON\(\(([\d\s,\.]+)\)\)', item['st_astext(rect)'])
                    p = re.split(r',|\s', r.group(1))
                    # item['rect'] = list(map(lambda x: float(x), [p[0], p[1], p[4], p[5]]))
                    item['rect'] = [float(p[0]), float(p[1]), float(p[4]), float(p[5])]
                    del item['st_astext(rect)']
                except Exception as e:
                    print('[Exception] get_trajectory POLYGON parse error', e)
        return rt


db = None


def db_init():
    global db
    db = MySQLPlugin(host=DBCONF['server'], user=DBCONF['user'], password=DBCONF['pwd'], db=DBCONF["db"])
    return db


def db_add_user(user_id, user_info, group_id, app_id):
    db.connect()
    db.add_user(user_id, user_info, group_id, app_id)
    db.close()


def db_del_user(user_id):
    db.connect()
    db.del_user(user_id)
    db.close()


def db_query_username():
    db.connect()
    user_list = db.query_user()
    name_list = list(map(lambda user: user.get('name'), user_list))
    db.close()
    return name_list


if __name__ == "__main__":
    import time
    from pprint import pprint

    db = MySQLPlugin()

    # db_init()
    # db_query_username()
    # db_del_user('6133129ce67c11e99281e0d55eee5246')
    # user_id = str(uuid.uuid1()).replace('-', '')
    # db_add_user(user_id, '何斌', 'member1', 'default')
    # db.connect()

    # print(user_id)
    # db.add_user(user_id, '何斌', 'member1', 'default')
    # db.close()

    db.connect()
    r = db.query_camera()
    pprint(r)
    #
    # r = db.query_user()
    # pprint(r)


    # db.update_headcount('table1', 2, int(time.time()*1000))

    # r = db.query_customer_within_memberinfo()
    # pprint(r)

    r = db.query_track_tbl()
    print(r)

    r = db.get_camera_tracks()
    pprint(r)

    r = db.get_trajectory('channel1', '10', [time.time()*1000-60*60*1000, time.time()*1000])
    pprint(r)