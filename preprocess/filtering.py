import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter


def fftspectrum(thesignal, fs, num):  # add fs
    nsel = thesignal.size
    fsel = fs * (np.arange(0, nsel / 2) / nsel)  # add fs*
    ysel = np.fft.fft(thesignal)
    ysel = np.abs(ysel)
    ysel = ysel[0:len(fsel)]
    # ysel=20*np.log(ysel)
    index = np.argsort(ysel)[::-1][:num]
    # print(fsel[index])
    # # plt.figure()
    # plt.plot(fsel[0:], ysel)
    # plt.show()
    # print(fsel[index][np.argmax(fsel[index])])
    return fsel[index][np.argmax(fsel[index])]


# fs=200
# t=np.arange(0,1,1/fs)
# sig=np.cos(2*np.pi*40*t)
# plt.figure()    # function needs
# fftspectrum(sig,fs)
# plt.show() # function needs


def mysubplot(file):
    l = len(file)

    if l == 2:
        plt.figure()
        plt.subplot(211)
        plt.plot(file[0])
        plt.subplot(212)
        plt.plot(file[1])

    if l == 3:
        plt.figure()
        plt.subplot(311)
        plt.plot(file[0])
        plt.subplot(312)
        plt.plot(file[1])
        plt.subplot(313)
        plt.plot(file[2])

    if l == 4:
        plt.figure()
        plt.subplot(411)
        plt.plot(file[0])
        plt.subplot(412)
        plt.plot(file[1])
        plt.subplot(413)
        plt.plot(file[2])
        plt.subplot(414)
        plt.plot(file[3])

    if l == 6:
        plt.figure()
        plt.subplot(611)
        plt.plot(file[0])
        plt.subplot(612)
        plt.plot(file[1])
        plt.subplot(613)
        plt.plot(file[2])
        plt.subplot(614)
        plt.plot(file[3])
        plt.subplot(615)
        plt.plot(file[4])
        plt.subplot(616)
        plt.plot(file[5])

    if l == 8:
        plt.figure()
        plt.subplot(811)
        plt.plot(file[0])
        plt.subplot(812)
        plt.plot(file[1])
        plt.subplot(813)
        plt.plot(file[2])
        plt.subplot(814)
        plt.plot(file[3])
        plt.subplot(815)
        plt.plot(file[4])
        plt.subplot(816)
        plt.plot(file[5])
        plt.subplot(817)
        plt.plot(file[6])
        plt.subplot(818)
        plt.plot(file[7])

    plt.show()


def mysubplot_addx(file):
    l = len(file)

    if l == 2:
        plt.figure()
        plt.subplot(211)
        plt.plot(file[0][0], file[0][1])
        plt.subplot(212)
        plt.plot(file[1][0], file[1][1])

    if l == 3:
        plt.figure()
        plt.subplot(311)
        plt.plot(file[0][0], file[0][1])
        plt.subplot(312)
        plt.plot(file[1][0], file[1][1])
        plt.subplot(313)
        plt.plot(file[2][0], file[2][1])

    if l == 4:
        plt.figure()
        plt.subplot(411)
        plt.plot(file[0][0], file[0][1])
        plt.subplot(412)
        plt.plot(file[1][0], file[1][1])
        plt.subplot(413)
        plt.plot(file[2][0], file[2][1])
        plt.subplot(414)
        plt.plot(file[3][0], file[3][1])

    if l == 6:
        plt.figure()
        plt.subplot(611)
        plt.plot(file[0][0], file[0][1])
        plt.subplot(612)
        plt.plot(file[1][0], file[1][1])
        plt.subplot(613)
        plt.plot(file[2][0], file[2][1])
        plt.subplot(614)
        plt.plot(file[3][0], file[3][1])
        plt.subplot(615)
        plt.plot(file[4][0], file[4][1])
        plt.subplot(616)
        plt.plot(file[5][0], file[5][1])

    if l == 8:
        plt.figure()
        plt.subplot(811)
        plt.plot(file[0][0], file[0][1])
        plt.subplot(812)
        plt.plot(file[1][0], file[1][1])
        plt.subplot(813)
        plt.plot(file[2][0], file[2][1])
        plt.subplot(814)
        plt.plot(file[3][0], file[3][1])
        plt.subplot(815)
        plt.plot(file[4][0], file[4][1])
        plt.subplot(816)
        plt.plot(file[5][0], file[5][1])
        plt.subplot(817)
        plt.plot(file[6][0], file[6][1])
        plt.subplot(818)
        plt.plot(file[7][0], file[7][1])

    plt.show()


def fftspectrum_value(thesignal, fs):  # add fs
    nsel = thesignal.size
    fsel = fs * (np.arange(0, nsel / 2) / nsel)  # add fs*
    ysel = np.fft.fft(thesignal)
    ysel = np.abs(ysel)
    ysel = ysel[1:len(fsel)]
    # ysel=20*np.log(ysel)
    # plt.figure()
    # plt.plot(fsel,ysel)
    # plt.show()
    return fsel, ysel


def Implement_Notch_Filter(time, band, freq, order, filter_type, data):
    from scipy.signal import iirfilter
    fs = 1 / time
    nyq = fs / 2.0
    low = freq - band / 2.0
    high = freq + band / 2.0
    low = low / nyq
    high = high / nyq
    b, a = iirfilter(order, [low, high], btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data
