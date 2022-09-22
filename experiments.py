import os
import random
from time import time
import csv
import psutil
from pyproj import CRS, Transformer

from common.persistence import PersistentDict
from index.vortree import VoRtreeIndex
from rknn import csd, slice, vr
from uuid import uuid1 as generate_uuid
import numpy as np

from shapely.geometry import Point

default_times = 30


class Statistics:
    def __init__(self):
        dioc = tuple(psutil.disk_io_counters())
        self._read_count = dioc[0]
        self._write_count = dioc[1]
        self._read_bytes = dioc[2]
        self._write_bytes = dioc[3]

    def reset(self):
        dioc = tuple(psutil.disk_io_counters())
        self._read_count = dioc[0]
        self._write_count = dioc[1]
        self._read_bytes = dioc[2]
        self._write_bytes = dioc[3]
        self._time = time()

    @property
    def read_count(self):
        dioc = tuple(psutil.disk_io_counters())
        return dioc[0] - self._read_count

    @property
    def write_count(self):
        dioc = tuple(psutil.disk_io_counters())
        return dioc[1] - self._write_count

    @property
    def io_count(self):
        dioc = tuple(psutil.disk_io_counters())
        _read_count = dioc[0] - self._read_count
        _write_count = dioc[1] - self._write_count
        return _read_count + _write_count

    @property
    def read_bytes(self):
        dioc = tuple(psutil.disk_io_counters())
        return dioc[2] - self._read_bytes

    @property
    def write_bytes(self):
        dioc = tuple(psutil.disk_io_counters())
        return dioc[3] - self._write_bytes

    @property
    def io_bytes(self):
        dioc = tuple(psutil.disk_io_counters())
        _read_bytes = dioc[2] - self._read_bytes
        _write_bytes = dioc[3] - self._write_bytes
        return _read_bytes + _write_bytes

    @property
    def time_elapse(self):
        return float(time() - self._time)


def get_points_from_csv(path):
    crs_WGS84 = CRS.from_epsg(4214)  # WGS84 Coordinate Reference System
    crs_WebMercator = CRS.from_epsg(2435)  # Web Mercator Coordinate Reference System
    transformer = Transformer.from_crs(crs_WGS84, crs_WebMercator, always_xy=True)
    data = list()
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'CSD-RkNN/' + path), mode='r',
              encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x, y = transformer.transform(float(row['XWGS']), float(row['YWGS']))
            data.append((row['ID'], Point(x, y)))
        return data


def generate_points(n, distribution, bounds=None):
    if distribution == 'Uniform':
        if bounds is None:
            bounds = (0, 0, 1.0, 1.0)
        p_set = set()
        data = []
        i = 0
        while len(p_set) < n:
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])
            if (x, y) not in p_set:
                p_set.add((x, y))
                data.append((str(i), Point(x, y)))
                i += 1
        return data
    elif distribution == 'Normal':
        if bounds is None:
            bounds = (0, 0, 1.0, 1.0)
        p_set = set()
        x_mu = (bounds[0] + bounds[2]) / 2
        x_sigma = abs(bounds[0] - bounds[2]) / 10
        y_mu = (bounds[1] + bounds[3]) / 2
        y_sigma = abs(bounds[1] - bounds[3]) / 10
        data = []
        i = 0
        while len(p_set) < n:
            x = np.random.normal(x_mu, x_sigma)
            y = np.random.normal(y_mu, y_sigma)
            if (x, y) not in p_set and bounds[0] < x < bounds[2] and bounds[1] < y < bounds[3]:
                p_set.add((x, y))
                data.append((str(i), Point(x, y)))
                i += 1
        return data


def get_points_from_txt(path):
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'CSD-RkNN/' + path)) as f:
        data = list()
        for line in f.readlines():
            line_str = line.strip().split()
            data.append((line_str[0], Point(float(line_str[1]), float(line_str[2]))))
        return data


def exist_index(name, distribution=None, n=None):
    path = name + '-' + distribution
    if n is not None:
        path += '-' + str(n)
    return os.path.exists(path)


def create_index(name, distribution=None, n=None):
    if distribution == 'Real':
        if name == 'User':
            data = get_points_from_txt('data/North America-1.txt')
        if name == 'Facility':
            data = get_points_from_txt('data/North America-2.txt')
    elif distribution == 'Uniform' or distribution == 'Normal':
        data = generate_points(n, distribution)
    elif distribution == 'Wuhan':
        if name == 'School':
            data = get_points_from_csv('data/school.csv')
        if name == 'Mall':
            data = get_points_from_csv('data/mall.csv')
        if name == 'Hospital':
            data = get_points_from_csv('data/hospital.csv')
        if name == 'Residence':
            data = get_points_from_csv('data/residence.csv')
        if name == 'Restaurant':
            data = get_points_from_csv('data/restaurant.csv')
    path = name + '-' + distribution
    if n is not None:
        path += '-' + str(n)
    index = VoRtreeIndex(path=path, data=data)
    index.close()
    print(path + ' Complete')


def load_index(name, distribution=None, n=None):
    path = name + '-' + distribution
    if n is not None:
        path += '-' + str(n)
    return VoRtreeIndex(path=path)


def create_temp_index_copy(name, distribution=None, n=None):
    if not exist_index(name, distribution, n):
        create_index(name, distribution, n)
    temp_name = name + '-' + str(generate_uuid())
    path = name + '-' + distribution
    temp_path = temp_name + '-' + distribution
    if n is not None:
        path += '-' + str(n)
        temp_path += '-' + str(n)
    PersistentDict.copy(path, temp_path)
    return load_index(temp_name, distribution, n)


def random_ids(n, name, distribution, size=None):
    if distribution == 'Uniform' or distribution == 'Normal':
        return [str(i) for i in random.sample(range(0, size), n)]
    elif distribution == 'Real':
        if name == 'User':
            return [i for i, p in random.sample(get_points_from_txt('data/North America-1.txt'), n)]
        if name == 'Facility':
            return [i for i, p in random.sample(get_points_from_txt('data/North America-2.txt'), n)]
    elif distribution == 'Wuhan':
        if name == 'School':
            return [i for i, p in random.sample(get_points_from_csv('data/school.csv'), n)]
        if name == 'Mall':
            return [i for i, p in random.sample(get_points_from_csv('data/mall.csv'), n)]
        if name == 'Hospital':
            return [i for i, p in random.sample(get_points_from_csv('data/hospital.csv'), n)]
        if name == 'Residence':
            return [i for i, p in random.sample(get_points_from_csv('data/residence.csv'), n)]
        if name == 'Restaurant':
            return [i for i, p in random.sample(get_points_from_csv('data/restaurant.csv'), n)]


def delete_index(name, distribution, n):
    path = name + '-' + distribution
    if n is not None:
        path += '-' + str(n)
    PersistentDict.drop(path)


class BenchmarkExperiments:
    @staticmethod
    def evaluate_effect_of_data_size_on_MonoRkNN(k, distribution=None, times=None):
        if distribution is None:
            uniform_time_v, uniform_io_v = BenchmarkExperiments.evaluate_effect_of_data_size_on_MonoRkNN(k, 'Uniform',
                                                                                                         times)
            normal_time_v, normal_io_v = BenchmarkExperiments.evaluate_effect_of_data_size_on_MonoRkNN(k, 'Normal',
                                                                                                       times)
            time_data = {'uniform': uniform_time_v['data'], 'normal': normal_time_v['data']}
            io_data = {'uniform': uniform_io_v['data'], 'normal': normal_io_v['data']}
            uniform_time_v['data'] = time_data
            uniform_io_v['data'] = io_data
            return uniform_time_v, uniform_io_v
        else:
            if times is None:
                times = default_times
            data_sizes = [50000, 100000, 150000, 200000]
            csd_time_cost = []
            vr_time_cost = []
            slice_time_cost = []
            csd_io_cost = []
            vr_io_cost = []
            slice_io_cost = []
            for data_size in data_sizes:
                q_ids = random_ids(times, 'Facility', distribution, data_size)
                # VR
                time_cost = []
                io_cost = []
                for i in q_ids:
                    facility_index = create_temp_index_copy('Facility', distribution, data_size)
                    st = Statistics()
                    st.reset()
                    list(vr.MonoRkNN(facility_index.nodes[i], k, facility_index))
                    time_cost.append(st.time_elapse)
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                vr_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                vr_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # SLICE
                time_cost = []
                io_cost = []
                for i in q_ids:
                    facility_index = create_temp_index_copy('Facility', distribution, data_size)
                    st = Statistics()
                    st.reset()
                    list(slice.MonoRkNN(facility_index.nodes[i], k, facility_index))
                    time_cost.append(st.time_elapse)
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                slice_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                slice_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # CSD
                time_cost = []
                io_cost = []
                for i in q_ids:
                    facility_index = create_temp_index_copy('Facility', distribution, data_size)
                    st = Statistics()
                    st.reset()
                    list(csd.MonoRkNN(facility_index.nodes[i], k, facility_index))
                    time_cost.append(st.time_elapse)
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                csd_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                csd_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            time_v = dict()
            time_v['data'] = {'VR': vr_time_cost, 'SLICE': slice_time_cost, 'CSD': csd_time_cost}
            time_v['x_tick_labels'] = data_sizes
            time_v['x_label'] = 'Data size'
            time_v['y_label'] = 'Time cost (in sec)'

            io_v = dict()
            io_v['data'] = {'VR': vr_io_cost, 'SLICE': slice_io_cost, 'CSD': csd_io_cost}
            io_v['x_tick_labels'] = data_sizes
            io_v['x_label'] = 'Data size'
            io_v['y_label'] = '# I/O'
            return time_v, io_v

    @staticmethod
    def evaluate_effect_of_data_size_on_BiRkNN(k, distribution=None, times=None):
        if distribution is None:
            uniform_time_v, uniform_io_v = BenchmarkExperiments.evaluate_effect_of_data_size_on_BiRkNN(k, 'Uniform',
                                                                                                       times)
            normal_time_v, normal_io_v = BenchmarkExperiments.evaluate_effect_of_data_size_on_BiRkNN(k, 'Normal', times)
            time_data = {'uniform': uniform_time_v['data'], 'normal': normal_time_v['data']}
            io_data = {'uniform': uniform_io_v['data'], 'normal': normal_io_v['data']}
            uniform_time_v['data'] = time_data
            uniform_io_v['data'] = io_data
            return uniform_time_v, uniform_io_v
        else:
            if times is None:
                times = default_times
            data_sizes = [50000, 100000, 150000, 200000]
            csd_time_cost = []
            vr_time_cost = []
            slice_time_cost = []
            csd_io_cost = []
            vr_io_cost = []
            slice_io_cost = []
            for data_size in data_sizes:
                q_ids = random_ids(times, 'Facility', distribution, data_size)
                # VR
                time_cost = []
                io_cost = []
                for i in q_ids:
                    user_index = create_temp_index_copy('User', distribution, data_size)
                    facility_index = create_temp_index_copy('Facility', distribution, data_size)
                    st = Statistics()
                    st.reset()
                    list(vr.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    user_index.close()
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                    user_index.drop_file()
                vr_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                vr_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # SLICE
                time_cost = []
                io_cost = []
                for i in q_ids:
                    user_index = create_temp_index_copy('User', distribution, data_size)
                    facility_index = create_temp_index_copy('Facility', distribution, data_size)
                    st = Statistics()
                    st.reset()
                    list(slice.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    user_index.close()
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                    user_index.drop_file()
                slice_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                slice_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # CSD
                time_cost = []
                io_cost = []
                for i in q_ids:
                    user_index = create_temp_index_copy('User', distribution, data_size)
                    facility_index = create_temp_index_copy('Facility', distribution, data_size)
                    st = Statistics()
                    st.reset()
                    list(csd.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    user_index.close()
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                    user_index.drop_file()
                csd_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                csd_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            time_v = dict()
            time_v['data'] = {'VR': vr_time_cost, 'SLICE': slice_time_cost, 'CSD': csd_time_cost}
            time_v['x_tick_labels'] = data_sizes
            time_v['x_label'] = 'Data size'
            time_v['y_label'] = 'Time cost (in sec)'
            io_v = dict()
            io_v['data'] = {'VR': vr_io_cost, 'SLICE': slice_io_cost, 'CSD': csd_io_cost}
            io_v['x_tick_labels'] = data_sizes
            io_v['x_label'] = 'Data size'
            io_v['y_label'] = '# I/O'
            return time_v, io_v

    @staticmethod
    def evaluate_effect_of_k_on_MonoRkNN(distribution, times=None):
        if distribution == 'Synthetic':
            uniform_time_v, uniform_io_v = BenchmarkExperiments.evaluate_effect_of_k_on_MonoRkNN('Uniform', times)
            normal_time_v, normal_io_v = BenchmarkExperiments.evaluate_effect_of_k_on_MonoRkNN('Normal', times)
            time_data = {'uniform': uniform_time_v['data'], 'normal': normal_time_v['data']}
            io_data = {'uniform': uniform_io_v['data'], 'normal': normal_io_v['data']}
            uniform_time_v['data'] = time_data
            uniform_io_v['data'] = io_data
            return uniform_time_v, uniform_io_v
        else:
            if times is None:
                times = default_times
            k_list = [1, 10, 100, 1000]
            if distribution == 'Real':
                data_size = None
            else:
                data_size = 100000
            csd_time_cost = []
            vr_time_cost = []
            slice_time_cost = []
            csd_io_cost = []
            vr_io_cost = []
            slice_io_cost = []
            for k in k_list:
                q_ids = random_ids(times, 'Facility', distribution, data_size)
                # VR
                time_cost = []
                io_cost = []
                for i in q_ids:
                    facility_index = create_temp_index_copy('Facility', distribution, data_size)
                    st = Statistics()
                    st.reset()
                    list(vr.MonoRkNN(facility_index.nodes[i], k, facility_index))
                    time_cost.append(st.time_elapse)
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                vr_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                vr_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # SLICE
                time_cost = []
                io_cost = []
                for i in q_ids:
                    facility_index = create_temp_index_copy('Facility', distribution, data_size)
                    st = Statistics()
                    st.reset()
                    list(slice.MonoRkNN(facility_index.nodes[i], k, facility_index))
                    time_cost.append(st.time_elapse)
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                slice_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                slice_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # CSD
                time_cost = []
                io_cost = []
                for i in q_ids:
                    facility_index = create_temp_index_copy('Facility', distribution, data_size)
                    st = Statistics()
                    st.reset()
                    list(csd.MonoRkNN(facility_index.nodes[i], k, facility_index))
                    time_cost.append(st.time_elapse)
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                csd_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                csd_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            time_v = dict()
            time_v['data'] = {'VR': vr_time_cost, 'SLICE': slice_time_cost, 'CSD': csd_time_cost}
            time_v['x_tick_labels'] = k_list
            time_v['x_label'] = '$k$'
            time_v['y_label'] = 'Time cost (in sec)'

            io_v = dict()
            io_v['data'] = {'VR': vr_io_cost, 'SLICE': slice_io_cost, 'CSD': csd_io_cost}
            io_v['x_tick_labels'] = k_list
            io_v['x_label'] = '$k$'
            io_v['y_label'] = '# I/O'
            return time_v, io_v

    @staticmethod
    def evaluate_effect_of_k_on_BiRkNN(distribution, times=None):
        if distribution == 'Synthetic':
            uniform_time_v, uniform_io_v = BenchmarkExperiments.evaluate_effect_of_k_on_BiRkNN('Uniform', times)
            normal_time_v, normal_io_v = BenchmarkExperiments.evaluate_effect_of_k_on_BiRkNN('Normal', times)
            time_data = {'uniform': uniform_time_v['data'], 'normal': normal_time_v['data']}
            io_data = {'uniform': uniform_io_v['data'], 'normal': normal_io_v['data']}
            uniform_time_v['data'] = time_data
            uniform_io_v['data'] = io_data
            return uniform_time_v, uniform_io_v
        else:
            if times is None:
                times = default_times
            k_list = [1, 10, 100, 1000]
            if distribution == 'Real':
                user_data_size = None
                facility_data_size = None
            else:
                user_data_size = 100000
                facility_data_size = 100000
            csd_time_cost = []
            vr_time_cost = []
            slice_time_cost = []
            csd_io_cost = []
            vr_io_cost = []
            slice_io_cost = []
            for k in k_list:
                q_ids = random_ids(times, 'Facility', distribution, facility_data_size)
                # VR
                time_cost = []
                io_cost = []
                for i in q_ids:
                    facility_index = create_temp_index_copy('Facility', distribution, facility_data_size)
                    user_index = create_temp_index_copy('User', distribution, user_data_size)
                    st = Statistics()
                    st.reset()
                    list(vr.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    facility_index.close()
                    user_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                    user_index.drop_file()
                vr_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                vr_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # SLICE
                time_cost = []
                io_cost = []
                for i in q_ids:
                    facility_index = create_temp_index_copy('Facility', distribution, facility_data_size)
                    user_index = create_temp_index_copy('User', distribution, user_data_size)
                    st = Statistics()
                    st.reset()
                    list(slice.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    facility_index.close()
                    user_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                    user_index.drop_file()
                slice_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                slice_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # CSD
                time_cost = []
                io_cost = []
                for i in q_ids:
                    facility_index = create_temp_index_copy('Facility', distribution, facility_data_size)
                    user_index = create_temp_index_copy('User', distribution, user_data_size)
                    st = Statistics()
                    st.reset()
                    list(csd.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    facility_index.close()
                    user_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                    user_index.drop_file()
                csd_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                csd_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            time_v = dict()
            time_v['data'] = {'VR': vr_time_cost, 'SLICE': slice_time_cost, 'CSD': csd_time_cost}
            time_v['x_tick_labels'] = k_list
            time_v['x_label'] = '$k$'
            time_v['y_label'] = 'Time cost (in sec)'
            io_v = dict()
            io_v['data'] = {'VR': vr_io_cost, 'SLICE': slice_io_cost, 'CSD': csd_io_cost}
            io_v['x_tick_labels'] = k_list
            io_v['x_label'] = '$k$'
            io_v['y_label'] = '# I/O'
            return time_v, io_v

    @staticmethod
    def evaluate_effect_of_data_distribution(k, times=None):
        if times is None:
            times = default_times
        distributions = ['Uniform', 'Real', 'Normal']
        csd_time_cost = []
        vr_time_cost = []
        slice_time_cost = []
        csd_io_cost = []
        vr_io_cost = []
        slice_io_cost = []
        for facility_distribution in distributions:
            if facility_distribution == 'Real':
                facility_data_size = None
            else:
                facility_data_size = 87902
            q_ids = random_ids(times, 'Facility', facility_distribution, facility_data_size)
            for user_distribution in distributions:
                if user_distribution == 'Real':
                    user_data_size = None
                else:
                    user_data_size = 87901
                # VR
                time_cost = []
                io_cost = []
                for i in q_ids:
                    user_index = create_temp_index_copy('User', user_distribution, user_data_size)
                    facility_index = create_temp_index_copy('Facility', facility_distribution, facility_data_size)
                    st = Statistics()
                    st.reset()
                    list(vr.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    user_index.close()
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                    user_index.drop_file()
                vr_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                vr_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # SLICE
                time_cost = []
                io_cost = []
                for i in q_ids:
                    user_index = create_temp_index_copy('User', user_distribution, user_data_size)
                    facility_index = create_temp_index_copy('Facility', facility_distribution, facility_data_size)
                    st = Statistics()
                    st.reset()
                    list(slice.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    user_index.close()
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                    user_index.drop_file()
                slice_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                slice_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # CSD
                time_cost = []
                io_cost = []
                for i in q_ids:
                    user_index = create_temp_index_copy('User', user_distribution, user_data_size)
                    facility_index = create_temp_index_copy('Facility', facility_distribution, facility_data_size)
                    st = Statistics()
                    st.reset()
                    list(csd.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    user_index.close()
                    facility_index.close()
                    io_cost.append(st.io_count)
                    facility_index.drop_file()
                    user_index.drop_file()
                csd_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                csd_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
        time_v = dict()
        time_v['data'] = {'VR': vr_time_cost, 'SLICE': slice_time_cost, 'CSD': csd_time_cost}
        time_v['x_tick_labels'] = ['(U,U)', '(U,R)', '(U,N)', '(R,U)', '(R,R)', '(R,N)', '(N,U)', '(N,R)', '(N,N)']
        time_v['x_label'] = 'Data distribution'
        time_v['y_label'] = 'Time cost (in sec)'
        io_v = dict()
        io_v['data'] = {'VR': vr_io_cost, 'SLICE': slice_io_cost, 'CSD': csd_io_cost}
        io_v['x_tick_labels'] = ['(U,U)', '(U,R)', '(U,N)', '(R,U)', '(R,R)', '(R,N)', '(N,U)', '(N,R)', '(N,N)']
        io_v['x_label'] = 'Data distribution'
        io_v['y_label'] = '# I/O'
        return time_v, io_v

    @staticmethod
    def evaluate_effect_of_user_num_relative_to_facility_num(k, distribution=None, times=None):
        if distribution is None:
            uniform_time_v, uniform_io_v = BenchmarkExperiments.evaluate_effect_of_user_num_relative_to_facility_num(k,
                                                                                                                     'Uniform',
                                                                                                                     times)
            normal_time_v, normal_io_v = BenchmarkExperiments.evaluate_effect_of_user_num_relative_to_facility_num(k,
                                                                                                                   'Normal',
                                                                                                                   times)
            time_data = {'uniform': uniform_time_v['data'], 'normal': normal_time_v['data']}
            io_data = {'uniform': uniform_io_v['data'], 'normal': normal_io_v['data']}
            uniform_time_v['data'] = time_data
            uniform_io_v['data'] = io_data
            return uniform_time_v, uniform_io_v
        else:
            if times is None:
                times = default_times
            facility_data_size = 100000
            csd_time_cost = []
            vr_time_cost = []
            slice_time_cost = []
            csd_io_cost = []
            vr_io_cost = []
            slice_io_cost = []
            scales = [0.25, 0.5, 1, 2, 4]
            q_ids = random_ids(times, 'Facility', distribution, facility_data_size)
            for s in scales:
                user_data_size = int(s * facility_data_size)
                # VR
                time_cost = []
                io_cost = []
                for i in q_ids:
                    user_index = create_temp_index_copy('User', distribution, user_data_size)
                    facility_index = create_temp_index_copy('Facility', distribution, facility_data_size)
                    st = Statistics()
                    st.reset()
                    list(vr.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    user_index.close()
                    facility_index.close()
                    io_cost.append(st.io_count)
                    user_index.drop_file()
                    facility_index.drop_file()
                vr_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                vr_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # SLICE
                time_cost = []
                io_cost = []
                for i in q_ids:
                    user_index = create_temp_index_copy('User', distribution, user_data_size)
                    facility_index = create_temp_index_copy('Facility', distribution, facility_data_size)
                    st = Statistics()
                    st.reset()
                    list(slice.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    user_index.close()
                    facility_index.close()
                    io_cost.append(st.io_count)
                    user_index.drop_file()
                    facility_index.drop_file()
                slice_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                slice_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
                # CSD
                time_cost = []
                io_cost = []
                for i in q_ids:
                    user_index = create_temp_index_copy('User', distribution, user_data_size)
                    facility_index = create_temp_index_copy('Facility', distribution, facility_data_size)
                    st = Statistics()
                    st.reset()
                    list(csd.BiRkNN(facility_index.nodes[i], k, facility_index, user_index))
                    time_cost.append(st.time_elapse)
                    user_index.close()
                    facility_index.close()
                    io_cost.append(st.io_count)
                    user_index.drop_file()
                    facility_index.drop_file()
                csd_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
                csd_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            time_v = dict()
            time_v['data'] = {'VR': vr_time_cost, 'SLICE': slice_time_cost, 'CSD': csd_time_cost}
            time_v['x_tick_labels'] = [str(int(s * 100)) + '%' for s in scales]
            time_v['x_label'] = '|$U$|/|$F$|'
            time_v['y_label'] = 'Time cost (in sec)'

            io_v = dict()
            io_v['data'] = {'VR': vr_io_cost, 'SLICE': slice_io_cost, 'CSD': csd_io_cost}
            io_v['x_tick_labels'] = [str(int(s * 100)) + '%' for s in scales]
            io_v['x_label'] = '|$U$|/|$F$|'
            io_v['y_label'] = '# I/O'
            return time_v, io_v


class CaseStudyExperiments:
    @staticmethod
    def evaluate_RkNN_for_school(times=None):
        if times is None:
            times = default_times
        k_list = [1, 4, 16, 64, 256]
        csd_time_cost = []
        vr_time_cost = []
        slice_time_cost = []
        csd_io_cost = []
        vr_io_cost = []
        slice_io_cost = []
        q_ids = random_ids(times, 'School', 'Wuhan')
        for k in k_list:
            # VR
            time_cost = []
            io_cost = []
            for i in q_ids:
                educational_institution_index = create_temp_index_copy('School', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(vr.BiRkNN(educational_institution_index.nodes[i], k, educational_institution_index,
                               residential_district_index))
                time_cost.append(st.time_elapse)
                educational_institution_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                educational_institution_index.drop_file()
                residential_district_index.drop_file()
            vr_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            vr_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            # SLICE
            time_cost = []
            io_cost = []
            for i in q_ids:
                educational_institution_index = create_temp_index_copy('School', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(slice.BiRkNN(educational_institution_index.nodes[i], k, educational_institution_index,
                                  residential_district_index))
                time_cost.append(st.time_elapse)
                educational_institution_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                educational_institution_index.drop_file()
                residential_district_index.drop_file()
            slice_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            slice_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            # CSD
            time_cost = []
            io_cost = []
            for i in q_ids:
                educational_institution_index = create_temp_index_copy('School', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(csd.BiRkNN(educational_institution_index.nodes[i], k, educational_institution_index,
                                residential_district_index))
                time_cost.append(st.time_elapse)
                educational_institution_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                educational_institution_index.drop_file()
                residential_district_index.drop_file()
            csd_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            csd_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
        time_v = dict()
        time_v['data'] = {'VR': vr_time_cost, 'SLICE': slice_time_cost, 'CSD': csd_time_cost}
        time_v['x_tick_labels'] = k_list
        time_v['x_label'] = '$k$'
        time_v['y_label'] = 'Time cost (in sec)'
        io_v = dict()
        io_v['data'] = {'VR': vr_io_cost, 'SLICE': slice_io_cost, 'CSD': csd_io_cost}
        io_v['x_tick_labels'] = k_list
        io_v['x_label'] = '$k$'
        io_v['y_label'] = '# I/O'
        return time_v, io_v

    @staticmethod
    def evaluate_RkNN_for_mall(times=None):
        if times is None:
            times = default_times
        k_list = [1, 4, 16, 64, 256]
        csd_time_cost = []
        vr_time_cost = []
        slice_time_cost = []
        csd_io_cost = []
        vr_io_cost = []
        slice_io_cost = []
        q_ids = random_ids(times, 'Mall', 'Wuhan')
        for k in k_list:
            # VR
            time_cost = []
            io_cost = []
            for i in q_ids:
                mall_index = create_temp_index_copy('Mall', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(vr.BiRkNN(mall_index.nodes[i], k, mall_index, residential_district_index))
                time_cost.append(st.time_elapse)
                mall_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                mall_index.drop_file()
                residential_district_index.drop_file()
            vr_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            vr_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            # SLICE
            time_cost = []
            io_cost = []
            for i in q_ids:
                mall_index = create_temp_index_copy('Mall', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(slice.BiRkNN(mall_index.nodes[i], k, mall_index, residential_district_index))
                time_cost.append(st.time_elapse)
                mall_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                mall_index.drop_file()
                residential_district_index.drop_file()
            slice_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            slice_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            # CSD
            time_cost = []
            io_cost = []
            for i in q_ids:
                mall_index = create_temp_index_copy('Mall', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(csd.BiRkNN(mall_index.nodes[i], k, mall_index, residential_district_index))
                time_cost.append(st.time_elapse)
                mall_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                mall_index.drop_file()
                residential_district_index.drop_file()
            csd_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            csd_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
        time_v = dict()
        time_v['data'] = {'VR': vr_time_cost, 'SLICE': slice_time_cost, 'CSD': csd_time_cost}
        time_v['x_tick_labels'] = k_list
        time_v['x_label'] = '$k$'
        time_v['y_label'] = 'Time cost (in sec)'
        io_v = dict()
        io_v['data'] = {'VR': vr_io_cost, 'SLICE': slice_io_cost, 'CSD': csd_io_cost}
        io_v['x_tick_labels'] = k_list
        io_v['x_label'] = '$k$'
        io_v['y_label'] = '# I/O'
        return time_v, io_v

    @staticmethod
    def evaluate_RkNN_for_hospital(times=None):
        if times is None:
            times = default_times
        k_list = [1, 4, 16, 64, 256]
        csd_time_cost = []
        vr_time_cost = []
        slice_time_cost = []
        csd_io_cost = []
        vr_io_cost = []
        slice_io_cost = []
        q_ids = random_ids(times, 'Hospital', 'Wuhan')
        for k in k_list:
            # VR
            time_cost = []
            io_cost = []
            for i in q_ids:
                medical_institution_index = create_temp_index_copy('Hospital', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(vr.BiRkNN(medical_institution_index.nodes[i], k, medical_institution_index,
                               residential_district_index))
                time_cost.append(st.time_elapse)
                medical_institution_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                medical_institution_index.drop_file()
                residential_district_index.drop_file()
            vr_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            vr_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            # SLICE
            time_cost = []
            io_cost = []
            for i in q_ids:
                medical_institution_index = create_temp_index_copy('Hospital', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(slice.BiRkNN(medical_institution_index.nodes[i], k, medical_institution_index,
                                  residential_district_index))
                time_cost.append(st.time_elapse)
                medical_institution_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                medical_institution_index.drop_file()
                residential_district_index.drop_file()
            slice_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            slice_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            # CSD
            time_cost = []
            io_cost = []
            for i in q_ids:
                medical_institution_index = create_temp_index_copy('Hospital', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(csd.BiRkNN(medical_institution_index.nodes[i], k, medical_institution_index,
                                residential_district_index))
                time_cost.append(st.time_elapse)
                medical_institution_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                medical_institution_index.drop_file()
                residential_district_index.drop_file()
            csd_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            csd_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
        time_v = dict()
        time_v['data'] = {'VR': vr_time_cost, 'SLICE': slice_time_cost, 'CSD': csd_time_cost}
        time_v['x_tick_labels'] = k_list
        time_v['x_label'] = '$k$'
        time_v['y_label'] = 'Time cost (in sec)'
        io_v = dict()
        io_v['data'] = {'VR': vr_io_cost, 'SLICE': slice_io_cost, 'CSD': csd_io_cost}
        io_v['x_tick_labels'] = k_list
        io_v['x_label'] = '$k$'
        io_v['y_label'] = '# I/O'
        return time_v, io_v

    @staticmethod
    def evaluate_RkNN_for_restaurant(times=None):
        if times is None:
            times = default_times
        k_list = [1, 4, 16, 64, 256]
        csd_time_cost = []
        vr_time_cost = []
        slice_time_cost = []
        csd_io_cost = []
        vr_io_cost = []
        slice_io_cost = []
        q_ids = random_ids(times, 'Restaurant', 'Wuhan')
        for k in k_list:
            # VR
            time_cost = []
            io_cost = []
            for i in q_ids:
                restaurant_index = create_temp_index_copy('Restaurant', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(vr.BiRkNN(restaurant_index.nodes[i], k, restaurant_index, residential_district_index))
                time_cost.append(st.time_elapse)
                restaurant_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                restaurant_index.drop_file()
                residential_district_index.drop_file()
            vr_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            vr_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            # SLICE
            time_cost = []
            io_cost = []
            for i in q_ids:
                restaurant_index = create_temp_index_copy('Restaurant', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(slice.BiRkNN(restaurant_index.nodes[i], k, restaurant_index, residential_district_index))
                time_cost.append(st.time_elapse)
                restaurant_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                restaurant_index.drop_file()
                residential_district_index.drop_file()
            slice_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            slice_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
            # CSD
            time_cost = []
            io_cost = []
            for i in q_ids:
                restaurant_index = create_temp_index_copy('Restaurant', 'Wuhan')
                residential_district_index = create_temp_index_copy('Residence', 'Wuhan')
                st = Statistics()
                st.reset()
                list(csd.BiRkNN(restaurant_index.nodes[i], k, restaurant_index, residential_district_index))
                time_cost.append(st.time_elapse)
                restaurant_index.close()
                residential_district_index.close()
                io_cost.append(st.io_count)
                restaurant_index.drop_file()
                residential_district_index.drop_file()
            csd_time_cost.append({'mean': round(round(np.mean(time_cost), 2), 2), 'median':round(np.median(time_cost), 2),'std':np.std(time_cost)})
            csd_io_cost.append({'mean': round(np.mean(io_cost), 1), 'median':round(np.median(io_cost), 1),'std':round(np.std(io_cost), 1)})
        time_v = dict()
        time_v['data'] = {'VR': vr_time_cost, 'SLICE': slice_time_cost, 'CSD': csd_time_cost}
        time_v['x_tick_labels'] = k_list
        time_v['x_label'] = '$k$'
        time_v['y_label'] = 'Time cost (in sec)'
        io_v = dict()
        io_v['data'] = {'VR': vr_io_cost, 'SLICE': slice_io_cost, 'CSD': csd_io_cost}
        io_v['x_tick_labels'] = k_list
        io_v['x_label'] = '$k$'
        io_v['y_label'] = '# I/O'
        return time_v, io_v
