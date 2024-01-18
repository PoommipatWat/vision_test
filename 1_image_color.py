import cv2
import numpy as np

def rgb2hsv(r, g, b):               # r,g,b = [0,255]
    M = max([r, g, b])
    m = min([r, g, b])
    if m == M:
        h = 0
    elif M == r:
        h = (g-b)/(M-m)*60%360
    elif M == g:
        h = (b-r)/(M-m)*60+120
    elif M == b:
        h = (r-g)/(M-m)*60+240
    return [h, 1-(m/M), M/255]      #  h = [0,360]      # s, v = [0,1]

def hsv2rgb(h, s, v):               #  h = [0,360]      # s, v = [0,1]
    if s == 0:
        r = g = b = int(v * 255)
    else:
        h /= 60.0
        i = int(h)
        f = h - i
        p = int(255 * (v * (1.0 - s)))
        q = int(255 * (v * (1.0 - s * f)))
        t = int(255 * (v * (1.0 - s * (1.0 - f))))
        v = int(v * 255)
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
    return [r, g, b]                # r,g,b = [0,255]

def show_hsv(h, s, v):
    h = h % 360
    h_norm = h / 2
    s_norm = int(s * 255)
    v_norm = int(v * 255) 
    hsv_image = np.array([[[h_norm, s_norm, v_norm]]], dtype=np.uint8)
    rgb_color = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    color_display = np.zeros((100, 100, 3), dtype=np.uint8)
    color_display[:, :] = rgb_color
    cv2.imshow('Color from HSV', color_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_rgb(r, g, b):
    rgb_color = np.array([[[b, g, r]]], dtype=np.uint8)
    color_display = np.zeros((100, 100, 3), dtype=np.uint8)
    color_display[:, :] = rgb_color
    cv2.imshow('Color from RGB', color_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rgb2gray(r,g,b):                  # Average intensity
    return np.mean([r,g,b])

def rgb2gray_luminance(r,g,b):        # Luminance เพื่อให้ใกล้เคียงกับความรู้สึกมนุษย์
    return 0.299*r + 0.587*g + 0.114*b


