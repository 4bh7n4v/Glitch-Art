import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploaded')
OUTPUT_FOLDER = os.path.join(app.root_path, 'static', 'output')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

UPLOAD_FILENAME = "original.png"  # fixed name for original upload

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    original_image = UPLOAD_FILENAME
    processed_image = None

    if request.method == 'POST':
        file = request.files.get('image1')
        operation = request.form.get("operations")

        # Save the uploaded file always with the fixed name
        if file and file.filename:
            upload_path = os.path.join(UPLOAD_FOLDER, UPLOAD_FILENAME)
            file.save(upload_path)
            original_image = UPLOAD_FILENAME
            img = cv2.imread(upload_path)
            processed_filename = "processed.png"  # fixed processed file name
        else:
            # No new upload, use last processed or original
            processed_filename = request.form.get('last_processed_image')
            if not processed_filename:
                processed_filename = "processed.png"
            img_path = os.path.join(OUTPUT_FOLDER, processed_filename)
            img = cv2.imread(img_path)
            if img is None:
                # fallback to original image if processed missing
                img = cv2.imread(os.path.join(UPLOAD_FOLDER, UPLOAD_FILENAME))
                processed_filename = None

        operation = request.form.get('operations')
        print("Operation Mode : ",operation)
        if operation == "reset":
    # Show original image, clear processed
            processed_image = UPLOAD_FILENAME  # 'original.png'
            print("Processed image being passed to template:", processed_image)
            return render_template('index.html',
                                original_image=original_image,  # "original.png"
                                processed_image=processed_image,
                                last_processed_image=None,
                                error=None)

        try:
            if operation == 'quantization':
                bit_depth = int(request.form.get('bit_depth', 2))
                img = Quantization(img, bit_depth)
            elif operation == 'rgb_shift':
                img = RGBShift(img)
            elif operation == 'rgb_rotate':
                img = RGBRotate(img)
            elif operation == 'rotate_angle':
                angle = safe_float(request.form, 'angle', 0.0)
                img = RotateImage(img, angle)
            elif operation == 'zoom':
                zoom_factor = safe_float(request.form, 'zoom_factor', 1.0)
                img = Zoom(img, zoom_factor)
            elif operation == 'shear':
                shear_x = safe_float(request.form, 'shear_x', 0.0)
                shear_y = safe_float(request.form, 'shear_y', 0.0)
                img = Shear(img, shear_x, shear_y)
            elif operation == 'pixel_sort_edge':
                img = PixelSortEdge(img)
            elif operation == 'pixel_sort_gray':
                orientation = request.form.get('orientation', 'horizontal')
                img = PixelSortGray(img, orientation)
            elif operation == "posterize":
                bit_depth = int(request.form.get('bit_depth', 2))
                img = posterize_image(img, bit_depth)
            elif operation == 'recovery':
                backup_path = os.path.join(OUTPUT_FOLDER, "processed_backup.png")
                if os.path.exists(backup_path):
                    img = cv2.imread(backup_path)
                    processed_filename = "processed_backup.png"
                else:
                    error = "No backup image to recover."
            else:
                error = "Invalid operation selected."
        except Exception as e:
            error = f"Processing error: {e}"

        # Save processed image
        if operation != 'reset' and img is not None:
            processed_path = os.path.join(OUTPUT_FOLDER, "processed.png")
            cv2.imwrite(processed_path, img)
            processed_image = "processed.png"

        return render_template('index.html',
                            original_image=original_image,
                            processed_image=processed_image,
                            last_processed_image=None,
                            error=None)


    # GET request, just show blank or last known
    return render_template('index.html', error=error)


if __name__ == '__main__':
    app.run(debug=True)


