import cv2
import numpy as np


def safe_float(form, key, default=0.0):
    val = form.get(key, '').strip()
    try:
        return float(val) if val else default
    except:
        return default

def Quantization(img, bit_depth):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    levels = 2 ** bit_depth
    quantized = np.round(gray / 255 * (levels - 1)) * (255 // (levels - 1))
    return cv2.cvtColor(quantized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def RGBShift(img):
    b, g, r = cv2.split(img)
    return cv2.merge((r, b, g))

def RGBRotate(img):
    b, g, r = cv2.split(img)
    return cv2.merge((g, r, b))

def RotateImage(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h))

def PixelSortEdge(img, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    sorted_img = img.copy()
    for y in range(edges.shape[0]):
        edge_indices = np.where(edges[y, :] > 0)[0]
        if len(edge_indices) == 0:
            continue
        start = edge_indices[0]
        end = edge_indices[-1] + 1
        segment = sorted_img[y, start:end]
        sorted_segment = segment[np.argsort(np.sum(segment, axis=1))]
        sorted_img[y, start:end] = sorted_segment
    return sorted_img

def Zoom(img, zoom_factor):
    if zoom_factor == 1.0:
        return img
    h, w = img.shape[:2]
    resized = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
    zh, zw = resized.shape[:2]
    if zoom_factor > 1.0:
        startx = (zw - w) // 2
        starty = (zh - h) // 2
        return resized[starty:starty + h, startx:startx + w]
    else:
        top = (h - zh) // 2
        bottom = h - zh - top
        left = (w - zw) // 2
        right = w - zw - left
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT)

def posterize_image(img, levels):
    if levels < 1:
        levels = 1
    factor = 256 // levels
    posterized = (img // factor) * factor
    return posterized.astype(np.uint8)

def Shear(img, shear_x, shear_y):
    h, w = img.shape[:2]
    M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    new_w = int(w + abs(shear_x * h))
    new_h = int(h + abs(shear_y * w))
    return cv2.warpAffine(img, M, (new_w, new_h))

def PixelSortGray(img, orientation='horizontal'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sorted_img = img.copy()
    if orientation == 'horizontal':
        for y in range(gray.shape[0]):
            row_pixels = sorted_img[y, :]
            gray_row = gray[y, :]
            sorted_indices = np.argsort(gray_row)
            sorted_img[y, :] = row_pixels[sorted_indices]
    elif orientation == 'vertical':
        for x in range(gray.shape[1]):
            col_pixels = sorted_img[:, x]
            gray_col = gray[:, x]
            sorted_indices = np.argsort(gray_col)
            sorted_img[:, x] = col_pixels[sorted_indices]
    else:
        return PixelSortGray(img, 'horizontal')
    return sorted_img
