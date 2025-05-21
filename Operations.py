import cv2
import numpy as np

# This function adds static noise to the image.
def StaticNoise(img, intensity):
    noise = np.random.randint(0, 256, img.shape, dtype=np.uint8)
    mask = np.random.rand(*img.shape[:2]) < intensity
    for c in range(3): 
        img[:, :, c][mask] = noise[:, :, c][mask]
    return img

# This loop applies the noise to each of the 3 color channels (Red, Green, Blue).
# Only the pixels selected by the mask randomly gets updated with the random noise values.

# This function shifts the pixels in the horizontal direction by a random amount.
def SliceShift(img, block_height):
    output = img.copy()
    h, w = img.shape[:2]
    for y in range(0, h, block_height):
        shift = np.random.randint(-w // 10, w // 10)
        output[y:y+block_height] = np.roll(output[y:y+block_height], shift, axis=1)
    return output

# This function shifts the pixels in the horizontal direction by a random amount.
# The shift amount is between -w // 10 and w // 10.
# The shift is applied to each block of pixels defined by the block_height.

# This function quantizes the image to a specified bit depth.
def Quantization(img, bit_depth):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    levels = 2 ** bit_depth
    quantized = np.round(gray / 255 * (levels - 1)) * (255 // (levels - 1))
    return cv2.cvtColor(quantized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# This function quantizes the image to a specified bit depth.
# The image is first converted to grayscale, then quantized to the specified bit depth.
# The quantized image is then converted back to the BGR color space.

# This function shifts the RGB channels of the image.
def RGBShift(img):
    b, g, r = cv2.split(img)
    return cv2.merge((r, b, g))

# This function shifts the RGB channels of the image.
# The blue channel is shifted to the red channel, the green channel is shifted to the blue channel, and the red channel is shifted to the green channel.

# This function rotates the image by 90 degrees.
def RGBRotate(img):
    b, g, r = cv2.split(img)
    return cv2.merge((g, r, b))

# This function rotates the image by 90 degrees.
# The blue channel is shifted to the red channel, the green channel is shifted to the blue channel, and the red channel is shifted to the green channel.

# This function rotates the image by a specified angle.
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

# This function rotates the image by a specified angle.
# The image is rotated around its center.
# The new width and height of the image are calculated based on the rotation matrix.
# The image is then rotated using the cv2.warpAffine function.

# This function sorts the pixels in the image based on the edges of the image.
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

# This function sorts the pixels in the image based on the edges of the image.
# The image is first converted to grayscale, then the edges are detected using the cv2.Canny function.
# The pixels are then sorted based on the edges.

# This function zooms in or out on the image.
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

# This function zooms in or out on the image.
# The image is resized to the specified zoom factor.
# The image is then cropped to the specified zoom factor.

# This function posterizes the image.
def posterize_image(img, levels):
    if levels < 1:
        levels = 1
    factor = 256 // levels
    posterized = (img // factor) * factor
    return posterized.astype(np.uint8)

# This function posterizes the image.
# The image is divided into the specified number of levels.
# The image is then quantized to the specified number of levels.

# This function shears the image.
def Shear(img, shear_x, shear_y):
    h, w = img.shape[:2]
    M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    new_w = int(w + abs(shear_x * h))
    new_h = int(h + abs(shear_y * w))
    return cv2.warpAffine(img, M, (new_w, new_h))

# This function shears the image.
# The image is sheared by the specified shear_x and shear_y values.
# The image is then warped using the cv2.warpAffine function.

# This function sorts the pixels in the image based on the grayscale values of the image.
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

# This function sorts the pixels in the image based on the grayscale values of the image.
# The image is first converted to grayscale, then the pixels are sorted based on the grayscale values.
# The image is then sorted based on the grayscale values.
