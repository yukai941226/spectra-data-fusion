"""
This main module contains all the functions available in spectrapepper. Please
use the search function to look up specific functionalities and keywords.
"""

import math
import copy
import random
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import interpolate
from scipy.special import gamma
from scipy.stats import stats
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
from scipy.interpolate import splev, splrep
from scipy.sparse.linalg import spsolve
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools
from itertools import combinations
import statistics as sta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from copy import deepcopy
import linecache
import os.path
import os

# 多元散射校正

# 均值中心化
def mean_center(data):
    ''' 
    data --> np.array格式
    '''
    m = data.shape[0]
    n = data.shape[1]
    MEAN = np.mean(data,axis=0)
    data_CT = np.array([[(data[i][j] - MEAN[j])  for j in range(n)] for i in range(m)])
    return data_CT

# 最大最小值归一化
def MMS(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MinMaxScaler :(n_samples, n_features)
       """
    return MinMaxScaler().fit_transform(data)


# 标准化
def SS(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after StandScaler :(n_samples, n_features)
       """
    return StandardScaler().fit_transform(data)


# 均值中心化
def CT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MeanScaler :(n_samples, n_features)
       """
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data


# 标准正态变换
def SNV(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after SNV :(n_samples, n_features)
    """
    m = data.shape[0]
    n = data.shape[1]
    # 求标准差
    data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # SNV计算
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return  np.array(data_snv)



# 移动平均平滑
def MA(data, WSZ=11):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param WSZ: int
       :return: data after MA :(n_samples, n_features)
    """

    for i in range(data.shape[0]):
        out0 = np.convolve(data[i], np.ones(WSZ, dtype=int), 'valid') / WSZ # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(data[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(data[i, :-WSZ:-1])[::2] / r)[::-1]
        data[i] = np.concatenate((start, out0, stop))
    return data


# Savitzky-Golay平滑滤波
def SG(data, w=11, p=2):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param w: int
       :param p: int
       :return: data after SG :(n_samples, n_features)
    """
    return signal.savgol_filter(data, w, p)


# 一阶导数
def D1(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after First derivative :(n_samples, n_features)
    """
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di


# 二阶导数
def D2(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after second derivative :(n_samples, n_features)
    """
    data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    temp2 = (pd.DataFrame(data)).diff(axis=1)
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return spec_D2


# 趋势校正(DT)
def DT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after DT :(n_samples, n_features)
    """
    lenth = data.shape[1]
    x = np.asarray(range(lenth), dtype=np.float32)
    out = np.array(data)
    l = LinearRegression()
    for i in range(out.shape[0]):
        l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] - (j * k + b)

    return out


# 多元散射校正
def MSC(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MSC :(n_samples, n_features)
    """
    n, p = data.shape
    msc = np.ones((n, p))

    for j in range(n):
        mean = np.mean(data, axis=0)

    # 线性拟合
    for i in range(n):
        y = data[i, :]
        l = LinearRegression()
        l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        msc[i, :] = (y - b) / k
    return msc

# 小波变换
def wave(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after wave :(n_samples, n_features)
    """
    data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    def wave_(data):
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)
        threshold = 0.04
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'db8')
        return datarec

    tmp = None
    for i in range(data.shape[0]):
        if (i == 0):
            tmp = wave_(data[i])
        else:
            tmp = np.vstack((tmp, wave_(data[i])))

    return tmp
    
# 对每条光谱的最大值为1
def normtomax(y, to=1, zeromin=False):
    """
    Normalizes spectras to the maximum value of each, in other words, the
    maximum value of each spectras is set to 1.

    :type y: list[float]
    :param y: Single or multiple vectors to normalize.
    
    :type to: float
    :param to: value to which normalize to. Default is 1.
    
    :type zeromin: boolean
    :param zeromin: If `True`, the minimum value is traslated to 0. Default
        values is `False`
    
    :returns: Normalized data.
    :rtype: list[float]
    """
    y = copy.deepcopy(y)  # so it does not chamge the input list
    dims = len(np.array(y).shape)  # detect dimensions
    
    if dims == 1:
        y = [y]
    
    for i in range(len(y)):           
        if zeromin:
            min_data = min(y[i])
            y[i] = y[i] - min_data
        max_data = max(y[i])

        y[i] = to*np.array(y[i])/max_data

    if dims == 1:
        y = y[0]
    
    return y

##选择某个峰作为基线校正
def normtopeak(y, x, peak, shift=10):
    """
    Normalizes the spectras to a particular peak.

    :type y: list[float]
    :param y: Data to be normalized.

    :type x: list[float]
    :param x: x axis of the data

    :type peak: int
    :param peak: Peak position in x-axis values.

    :type shift: int
    :param shift: Range to look for the real peak. The default is 10.

    :returns: Normalized data.
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    dims = len(np.array(y).shape)
    shift = int(shift)
    pos = valtoind(peak, x)

    if dims == 1:
        y = [y]

    for j in range(len(y)):
        section = y[j][pos - shift:pos + shift]   
        y[j] = y[j] / max(section) 
    
    if dims == 1:
        y = y[0]
    
    return y
## 每条光谱等比例缩小val倍
def normtovalue(y, val):
    """
    Normalizes the spectras to a set value, in other words, the defined value
    will be reescaled to 1 in all the spectras.

    :type y: list[float]
    :param y: Single or multiple vectors to normalize.

    :type val: float
    :param val: Value to normalize to.

    :returns: Normalized data.
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    y = np.array(y)/val
    y = list(y)
    return y

##曲线下面积为1
def normsum(y, x=None, lims=None):
    """
    Normalizes the sum under the curve to 1, for single or multiple spectras.

    :type y: list[float]
    :param y: Single spectra or a list of them.

    :type lims: list[float, float]
    :param lims: Limits of the vector to nomrlize the sum. Default is `None`.

    :returns: Normalized data
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    dims = len(np.array(y).shape)
    if x is not None:
        pos = valtoind(lims, x)
    
    if dims == 1:
        y = [y]
    
    for i in range(len(y)):
        if lims is None:
            s = sum(y[i])
        else:
            s = sum(y[i][pos[0]:pos[1]])
        
        for j in range(len(y[i])):
            y[i][j] = y[i][j] / s
    
    if dims == 1:
        y = y[0]
    
    return y

##按照所有光谱最大值归一
def normtoglobalmax(y, globalmin=False):
    """
    Normalizes a list of spectras to the global max.

    :type y: list[float]
    :param y: List of spectras.

    :type globalmin: Bool
    :param globalmin: If `True`, the global minimum is reescaled to 0. Default
        is `False`.

    :returns: Normalized data
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    dims = len(np.array(y).shape)
    if dims > 1:
        maximum = -999999999  # safe start
        
        if globalmin==True:
            minimum = 999999999
        else:
            minimum = 0
            
        for i in range(len(y)):
            for j in range(len(y[0])):
                if y[i][j] > maximum:
                    maximum = y[i][j]
                if y[i][j] < minimum and globalmin==True:
                    minimum = y[i][j]
        
        if globalmin==True:
            for i in range(len(y)):
                for j in range(len(y[0])):
                    y[i][j] -= minimum
        
        y = normtovalue(y, (maximum-minimum))
    else:  # if s single vector, then is the same as nortomax (local)
        y = normtomax(y, zeromin=globalmin)
    return y

##读取数据，注意修改my_file的路径，出来的为列表数据
def load_spectras(sample=None):
    """
    Load sample specrtal data, axis included in the first line.
    
    :type sample: int, tuple
    :param sample: Index of the sample spectra wanted or a tuple with the range
        of the indeces of a groups of spectras.
    
    :returns: X axis and the sample spectral data selected.
    :rtype: list[float], list[float]
    """ 
    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, 'datasets', 'spectras.txt')
    data = load(my_file)
    x = data[0]
    y = [list(i) for i in data[1:]]
    
    dims = len(np.array(sample).shape)
    
    if dims == 1:
        y = y[sample[0]: sample[1]]
    if dims == 0 and sample is not None:
        y = y[sample]
    
    return x, y

def load_targets(flatten=True):
    """
    Load sample targets data for the spectras.
    
    :returns: Sample targets.
    :rtype: list[float]
    """
    
    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, 'datasets', 'targets.txt')
    data = load(my_file)
    
    if flatten:
        data = np.array(data).flatten()
    
    return data


def load_params(transpose=True):
    """
    Load sample parameters data for the spectras.
    
    :returns: Sample parameters.
    :rtype: list[float]
    """
    
    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, 'datasets', 'params.txt')
    data = load(my_file)
    
    if transpose:
        data = np.transpose(data)
    
    return data

## fromline从第几行开始读取,自动填充nan为-1
def load(file, fromline=0, transpose=False, dtype=float):
    """
    Load data from a standard text file obtained from LabSpec and other
    spectroscopy instruments. Normally, when single measurement these come in 
    columns with the first one being the x-axis. When it is a mapping, the
    first row is the x-axis and the following are the measurements.

    :type file: str
    :param file: Url of data file. Must not have headers and separated by 'spaces' (LabSpec).
    
    :type fromline: int
    :param fromline: Line of file from which to start loading data. The default is 0.
    
    :type transpose: boolean
    :param transpose: If True transposes the data. Default is False.
    
    :returns: List of the data.
    :rtype: list[float]
    """
    new_data = []
    raw_data = open(file, "r")
    i = 0
    for row in raw_data:
        if i >= fromline:
            # row = row.replace(",", ".")
            row = row.replace(";", " ")
            row = row.replace("NaN", "-1")
            row = row.replace("nan", "-1")
            row = row.replace("--", "-1")
            s_row = str.split(row)
            s_row = np.array(s_row, dtype=dtype)
            new_data.append(s_row)
        i += 1
    raw_data.close()

    if transpose:
        new_data = np.transpose(new_data)

    return new_data

def loadline(file, line=0, dtype='float', split=False):
    """
    Random access to file. Loads a specific line in a file. Useful when
    managing large data files in processes where time is important. It can
    load numbers as floats.

    :type file: str
    :param file: Url od the data file

    :type line: int
    :param line: Line number. Counts from 0.

    :type dtype: str
    :param dtype: Type of data. If its numeric then 'float', if text then 'string'.
        Default is 'float'.

    :type split: boolean
    :param split: True to make a list of strings when 'tp' is 'string', 
        separated by space. The default is False.

    :returns: Array with the desired line.
    :rtype: list[float]
    """
    line = int(line) + 1
    file = str(file)
    info = linecache.getline(file, line)
    
    if dtype == 'float':
        info = info.replace("NaN", "-1")
        info = info.replace("nan", "-1")
        info = info.replace("--", "-1")
        info = str.split(info)
        info = np.array(info, dtype=float)

    if dtype == 'string':
        if split:
                info = str.split(info)
        info = np.array(info, dtype=str)

    return info


def lowpass(y, cutoff=0.25, fs=30, order=2, nyq=0.75):
    """
    Butter low pass filter for a single or spectra or a list of them.
        
    :type y: list[float]
    :param y: List of vectors in line format (each line is a vector).
    
    :type cutoff: float
    :param cutoff: Desired cutoff frequency of the filter. The default is 0.25.
    
    :type fs: int
    :param fs: Sample rate in Hz. The default is 30.
    
    :type order: int
    :param order: Sin wave can be approx represented as quadratic. The default is 2.
    
    :type nyq: float
    :param nyq: Nyquist frequency, 0.75*fs is a good value to start. The default is 0.75*30.

    :returns: Filtered data
    :rtype: list[float]
    """
    y = copy.deepcopy(y)  # so it does not change the input list
    dims = len(np.array(y).shape)
    print(dims)
    if dims == 1:
        y = [y]
    
    normal_cutoff = cutoff / (nyq * fs)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    for i in range(len(y)):
        y[i] = filtfilt(b, a, y[i])

    if dims == 1:
        y = y[0]
    
    return y


def alsbaseline(y, lam=100, p=0.001, niter=10):
    """
    Calculation of the baseline using Asymmetric Least Squares Smoothing. This
    script only makes the calculation but it does not remove it. Original idea of
    this algorithm by P. Eilers and H. Boelens (2005):

    :type y: list[float]
    :param y: Spectra to calculate the baseline from.

    :type lam: int
    :param lam: Lambda, smoothness. The default is 100.

    :type p: float
    :param p: Asymmetry. The default is 0.001.

    :type niter: int
    :param niter: Niter. The default is 10.

    :returns: Returns the calculated baseline.
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    dims = len(np.array(y).shape)
    
    if dims == 1:
        y = [y]
    
    l = len(y[0])
    d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(l, l - 2))
    w = np.ones(l)
    for i in range(len(y)):
        for _ in range(niter):
            W = sparse.spdiags(w, 0, l, l)
            Z = W + lam * d.dot(d.transpose())       
            z = spsolve(Z, w * y[i])
            w = p * (y[i] > z) + (1 - p) * (y[i] < z)
        y[i] = y[i] - z  
                
    if dims == 1:
        y = y[0]
    
    return y


def bspbaseline(y, x, points, avg=5, remove=True, plot=False):
    """
    Calcuates the baseline using b-spline.

    :type y: list[float]
    :param y: Single or several spectras to remove the baseline from.

    :type x: list[float]
    :param x: x axis of the data, to interpolate the baseline function.

    :type points: list[float], list[[float, float]]
    :param points: Axis values of points to calculate the bspline. Axis ranges
        are also acepted. In this case, the `avg` value will be `0`.

    :type avg: int
    :param avg: Points to each side to make average. Default is 5. If `points`
        are axis ranges, then it is set to 0 and will not have any effect.

    :type remove: True
    :param remove: If `True`, calculates and returns `data - baseline`. If 
        `False`, then it returns the `baseline`.

    :type plot: bool
    :param plot: If True, calculates and returns (data - baseline).    

    :returns: The baseline.
    :rtype: list[float]
    """
    data = copy.deepcopy(y)
    x = list(x)
    points = list(points)
    pos = valtoind(points, x)
    avg = int(avg)
    dims = len(np.array(data).shape)

    if len(np.array(points).shape) == 2:
        avg = 0
    else:
        pos = [[i, i] for i in pos]
        points = [[i, i] for i in points]

    if dims < 2:
        data = [data]
    
    baseline = []
    result = []
    
    for j in range(len(data)):
        y_p = []
        for i in range(len(pos)):
            temp = np.mean(data[j][pos[i][0] - avg: pos[i][1] + avg + 1])
            y_p.append(temp)
        
        points = [np.mean(i) for i in points]
        spl = splrep(points, y_p)
        baseline.append(splev(x, spl))
        
        if len(np.array(baseline[j]).shape) == 2:
            #sometime this happends when doing preprocesing
            baseline[j] = [h[0] for h in baseline[j]]
        
        if remove:
            result.append(np.array(data[j]) - np.array(baseline[j]))
        else:
            result.append(baseline[j])

        if plot and j == 0:        
            plt.plot(x, data[0], label='Original')
            plt.plot(x, baseline[0], label='Baseline')
            plt.plot(points, y_p, 'o', color='red')
            plt.ylim(min(data[0]), max(data[0]))
            plt.legend()
            plt.show()
    
    if dims < 2:
        result = result[0]

    return result

##基线校正
def polybaseline(y, axis, points, deg=2, avg=5, remove=True, plot=False):
    """
    Calcuates the baseline using polynomial fit.

    :type y: list[float]
    :param y: Single or several spectras to remove the baseline from.

    :type axis: list[float]
    :param axis: x axis of the data, to interpolate the baseline function.

    :type points: list[int]
    :param points: positions in axis of points to calculate baseline.

    :type deg: int
    :param deg: Polynomial degree of the fit.

    :type avg: int
    :param avg: points to each side to make average.

    :type remove: True
    :param remove: if True, calculates and returns (y - baseline).

    :type plot: bool
    :param plot: if True, calculates and returns (y - baseline).    

    :returns: The baseline.
    :rtype: list[float]
    """
    y = copy.deepcopy(y)
    dims = len(np.array(y).shape)
    axis = list(axis)
    points = list(points)
    pos = valtoind(points, axis)
    avg = int(avg)
    
    if dims == 1:
        y = [y]
    
    baseline = []
    for j in range(len(y)):
        averages = []
        for i in range(len(pos)):
            averages.append(np.mean(y[j][pos[i] - avg: pos[i] + avg + 1]))
                    
        z = np.polyfit(points, averages, deg)  # polinomial fit
        f = np.poly1d(z)  # 1d polinomial
        fit = f(axis)
        if plot and j == 0:        
            plt.plot(axis, y[j])
            plt.plot(axis, fit)
            plt.plot(points, averages, 'o', color='red')
            plt.show()
    
        if remove:
            baseline.append(y[j] - fit)
        else:    
            baseline.append(fit)

    if dims == 1:
        baseline = baseline[0]

    return baseline

##返回波长索引值
def valtoind(vals, x):
    """
    To translate the value in an axis to its index in the axis, basically
    searches for the position of the value. It approximates to the closest.

    :type vals: list[float]
    :param vals: List of values to be searched and translated.

    :type x: list[float]
    :param x: Axis.

    :returns: Index, or position, in the axis of the values in vals
    :rtype: list[int], int
    """
    vals = copy.deepcopy(vals)
    shape = len(np.array(vals).shape)
    
    if shape > 1:
        pos = [[0 for _ in range(len(vals[0]))] for _ in range(len(vals))]  # i position of area limits
        for i in range(len(vals)):  # this loop takes the approx. x and takes its position
            for j in range(len(vals[0])):
                dif_temp = 999  # safe initial difference
                temp_pos = 0  # temporal best position
                for k in range(len(x)):  # search in axis
                    if abs(vals[i][j] - x[k]) < dif_temp:  # compare if better
                        temp_pos = k  # save best value
                        dif_temp = abs(vals[i][j] - x[k])  # calculate new diff
                vals[i][j] = x[temp_pos]  # save real value in axis
                pos[i][j] = temp_pos  # save the position
                
    if shape == 1:
        pos = [0 for _ in range(len(vals))]  # i position of area limits
        for i in range(len(vals)):  # this loop takes the approx. x and takes its position
            dif_temp = 999  # safe initial difference
            temp_pos = 0  # temporal best position
            for k in range(len(x)):  # search in axis
                if abs(vals[i] - x[k]) < dif_temp:  # compare if better
                    temp_pos = k  # save best value
                    dif_temp = abs(vals[i] - x[k])  # calculate new diff
            vals[i] = x[temp_pos]  # save real value in axis
            pos[i] = temp_pos  # save the position           
                
    if shape == 0:
        dif_temp = 9999999  # safe initial difference
        temp_pos = 0  # temporal best position
        for k in range(len(x)):
            if abs(vals - x[k]) < dif_temp:
                temp_pos = k
                dif_temp = abs(vals - x[k])
        vals = x[temp_pos]  # save real value in axis
        pos = temp_pos
    return pos


def peakfinder(y, x=None, between=False, ranges=None, look=10):
    """
    Finds the location of the peaks in a single vector.

    :type y: list[float]
    :param y: Data to find a peak in. Single spectra.

    :type x: list[float]
    :param x: X axis of the data. If no axis is passed then the axis goes 
        from 0 to N, where N is the length of the spectras. Default is `None`.

    :type between: list[float]
    :param between: Range in x values 

    :type ranges: list[[float, float]]
    :param ranges: Aproximate ranges of known peaks, if any. If no ranges are 
        known or defined, it will return all the peaks that comply with the 
        `look` criteria. If ranges are defined, it wont use the `look` criteria,
        but just for the absolute maximum within the range. Default is `None`.

    :type look: int
    :param look: Amount of position to each side to decide if it is a local 
        maximum. The default is 10.

    :returns: A list of the index of the peaks found.
    :rtype: list[int]
    """
    y = copy.deepcopy(y)
    peaks = []
    
    if len(np.array(x).shape) < 1:
        x = [i for i in range(len(y))]
        
    # if between:
    #     between = valtoind(between, x)

    if between:
        start, finish = between
    else:
        start, finish = 0, len(y)
    
    if not ranges:
        is_max = [0 for _ in y]
        is_min = [0 for _ in y]
        for i in range(start + look, finish - look):  # start at "look" to avoid o.o.r
            lower = 0  # negative if lower, positive if higher
            higher = 0
            for j in range(look):
                if y[i] <= y[i - look + j] and y[i] <= y[i + j]:  # search all range lower
                    lower += 1  # +1 if lower
                elif (y[i] >= y[i - look + j] and
                      y[i] >= y[i + j]):  # search all range higher
                    higher += 1  # +1 if higher
            if higher == look:  # if all higher then its local max
                is_max[i] = 1
                is_min[i] = 0
                peaks.append(int(i))
            elif lower == look:  # if all lower then its local min
                is_max[i] = 0
                is_min[i] = 1
            else:
                is_max[i] = 0
                is_min[i] = 0
                
    elif ranges:
        ranges = valtoind(ranges, x)
        if len(np.array(ranges).shape) > 1:
            for i in ranges:
                section = y[i[0]:i[1]]
                m = max(section)
                for j in range(i[0], i[1]):
                    if y[j] == m:
                        peaks.append(int(j))
        else:
            m = max(y[ranges[0]:ranges[1]])
            for j in range(ranges[0], ranges[1]):
                if y[j] == m:
                    peaks.append(int(j))            

    if len(peaks) == 0:
        print('No peak was detected using the defined criteria. Change the parameters and try again.') 
        peaks = []

    if len(peaks) == 1:
        peaks = int(peaks[0])
    
    return peaks


def fwhm(y, x, peaks, alpha=0.5, s=10):
    """
    Calculates the Full Width Half Maximum of specific peak or list of
    peaks for a single or multiple spectras.
    
    :type y: list
    :param y: spectrocopic data to calculate the fwhm from. Single vector or
        list of vectors.    

    :type x: list
    :param x: Axis of the data. If none, then the axis will be 0..N where N
        is the length of the spectra or spectras.
    
    :type peaks: float or list[float]
    :param peaks: Aproximate axis value of the position of the peak. If single
        peak then a float is needed. If many peaks are requierd then a list of
        them.
    
    :type alpha: float
    :param alpha: multiplier of maximum value to find width ratio. Default is 0.5
        which makes this a `full width half maximum`. If `alpha=0.25`, it would
        basically find the `full width quarter maximum`. `alpha` should be 
        ´0 < alpha < 1´. Default is 0.5.
    
    :type s: int
    :param s: Shift to sides to check real peak. The default is 10.   
        
    :type interpolate: boolean
    :param interpolate: If True, will interpolte according to `step` and `s`.   
    
    :returns: A list, or single float value, of the fwhm.
    :rtype: float or list[float]
    """
    dims = len(np.array(y).shape)
        
    if dims == 1:
        y = [y]
        
    ind = valtoind(peaks, x)
    dims_peaks = len(np.array(peaks).shape)
    if dims_peaks < 1:
        ind = [ind]
    
    r_fwhm = []
    for h in y:
        
        fwhm = []
        for j in range(len(ind)):
            for i in range(ind[j] - s, ind[j] + s):
                if h[i] > h[ind[j]]:
                    ind[j] = i
                       
            h_m = h[ind[j]]*alpha # half maximum 
            temp = 999999999
            left = 0
            for i in range(ind[j]):
                delta = abs(h[ind[j]-i] - h_m)
                if temp > delta:
                    temp = delta
                    left = ind[j]-i
            
            temp = 999999999
            right = 0
            for i in range(len(x)-ind[j]):
                delta = abs(h[ind[j]+i] - h_m)
                if temp > delta:
                    temp = delta
                    right = ind[j]+i
            
            if dims_peaks < 1:
                fwhm = x[right] - x[left]
            else:
                fwhm.append(x[right] - x[left])
        
        if dims > 1:
            r_fwhm.append(fwhm)
        else:
            r_fwhm = fwhm

    return r_fwhm   