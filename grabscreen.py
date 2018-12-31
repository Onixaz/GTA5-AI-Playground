# Done by Frannecklp

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api


def grab_screen(region=None):

    #### for manual region grab ######

    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    #### for auto grab ####
    # hwin = win32gui.FindWindow(None, 'Grand Theft Auto V')
    # rect = win32gui.GetWindowRect(hwin)

    # x = rect[0]
    # y = rect[1]
    # left = 0
    # top = 40
    # height = rect[3] - y - top
    # width = rect[2] - x

    ###################
    #bmp.CreateCompatibleBitmap(srcdc, width, height)
    # win32ui.error: CreateCompatibleDC failed
    # if getting the error above, try the manual grab

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
