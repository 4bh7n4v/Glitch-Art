import os
import shutil
from flask import Flask, render_template, request
from Operations import *
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploaded')
OUTPUT_FOLDER = os.path.join(app.root_path, 'static', 'output')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

UPLOAD_FILENAME = "original.jpg"  # fixed filename for original image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_float(form, key, default=0.0):
    val = form.get(key, '').strip()
    try:
        return float(val) if val else default
    except:
        return default

def reset_images():
    original_path = os.path.join(UPLOAD_FOLDER, UPLOAD_FILENAME)
    processed_path = os.path.join(OUTPUT_FOLDER, "processed.png")
    backup_path = os.path.join(OUTPUT_FOLDER, "processed_backup.png")
    demo_input_path = os.path.join(UPLOAD_FOLDER, "DemoInput.jpg")

    # Remove existing images if any
    for path in [original_path, processed_path, backup_path]:
        if os.path.exists(path):
            os.remove(path)

    # Copy DemoInput.jpg as original.jpg
    if os.path.exists(demo_input_path):
        shutil.copyfile(demo_input_path, original_path)
    else:
        print("Warning: DemoInput.jpg not found in uploaded folder!")

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    original_image = UPLOAD_FILENAME
    processed_image = None

    if request.method == 'POST':
        file = request.files.get('image1')
        operation = request.form.get('operations')

        if file and file.filename and allowed_file(file.filename):
            upload_path = os.path.join(UPLOAD_FOLDER, UPLOAD_FILENAME)
            file.save(upload_path)
            original_image = UPLOAD_FILENAME
            img = cv2.imread(upload_path)
        else:
            # Use processed.png if it exists, else fallback to original.jpg
            processed_path = os.path.join(OUTPUT_FOLDER, "processed.png")
            if os.path.exists(processed_path):
                img = cv2.imread(processed_path)
                original_image = UPLOAD_FILENAME  # still display the original on left
            else:
                original_path = os.path.join(UPLOAD_FOLDER, UPLOAD_FILENAME)
                img = cv2.imread(original_path)

#
        if img is None:
            error = f"Failed to load base image."
            return render_template('index.html', error=error)

        if operation == "reset":
            # On reset, just show original
            processed_image = None
            return render_template('index.html',
                                   original_image=original_image,
                                   processed_image=processed_image,
                                   last_processed_image=None,
                                   error=None)

        try:
            # Apply the requested filter
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
            elif operation == 'pixel_sort_gray':
                orientation = request.form.get('orientation', 'horizontal')
                img = PixelSortGray(img, orientation)
            elif operation == "posterize":
                bit_depth = int(request.form.get('bit_depth', 2))
                img = posterize_image(img, bit_depth)
            elif operation == 'Noise':
                noise_level = float(request.form.get('noise_level', 0.0))
                img = StaticNoise(img, noise_level)
            elif operation == 'SliceShift':
                block_height = int(request.form.get('block_height', 10))
                img = SliceShift(img, block_height)
            elif operation == 'recovery':
                backup_path = os.path.join(OUTPUT_FOLDER, "processed_backup.png")
                if os.path.exists(backup_path):
                    img = cv2.imread(backup_path)
                else:
                    error = "No backup image to recover."
                    return render_template('index.html', error=error)
            else:
                error = "Invalid operation selected."
                return render_template('index.html', error=error)
        except Exception as e:
            error = f"Processing error: {e}"
            return render_template('index.html', error=error)

        # Save processed image
        processed_path = os.path.join(OUTPUT_FOLDER, "processed.png")
        backup_path = os.path.join(OUTPUT_FOLDER, "processed_backup.png")

        if os.path.exists(processed_path):
            os.replace(processed_path, backup_path)

        cv2.imwrite(processed_path, img)
        processed_image = "processed.png"

        return render_template('index.html',
                               original_image=original_image,
                               processed_image=processed_image,
                               last_processed_image=None,
                               error=None)

    # GET request — show original image only, no processed
    return render_template('index.html',
                           original_image=original_image,
                           processed_image=None,
                           error=None)


if __name__ == '__main__':
    reset_images()
    app.run(debug=True)
