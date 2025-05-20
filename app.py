import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from Operations import *

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
            # elif operation == 'pixel_sort_edge':
            #     img = PixelSortEdge(img)
            elif operation == 'pixel_sort_gray':
                orientation = request.form.get('orientation', 'horizontal')
                img = PixelSortGray(img, orientation)
            elif operation == "posterize":
                bit_depth = int(request.form.get('bit_depth', 2))
                img = posterize_image(img, bit_depth)
            elif operation == 'Noise':
                noise_level = float(request.form['noise_level'])
                img = StaticNoise(img, noise_level)

            elif operation == 'SliceShift':
                block_height = int(request.form['block_height'])
                img = SliceShift(img, block_height)
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
            backup_path = os.path.join(OUTPUT_FOLDER, "processed_backup.png")

            # Backup the current processed image if it exists
            if os.path.exists(processed_path):
                os.replace(processed_path, backup_path)

            # Save the new processed image
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


