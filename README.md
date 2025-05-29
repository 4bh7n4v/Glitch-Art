## GLITCH ART IMAGE GENERATOR
# THE GAME OF PIXEL MANIPULATION

Glitch Art is a Multimedia Project for Manuplulating pixel of Image

# Understanding And Applying Project

1. **First, get a copy of the code from GitHub**

```bash
git clone https://github.com/4bh7n4v/Glitch-Art.git
cd Glitch-Art
```

2. **Install Requirements Using requirements.txt from the Directory**
```
pip install -r requirements.txt
```
3. **Understand the Folder Structure**
```
glitch-art-app/
├── app.py                    # Flask app logic
├── Operations.py             # All image processing functions
├── requirements.txt          
├── templates/
│   └── index.html            # HTML interface (Jinja2 templating)
├── static/
│   ├── uploaded/             # Stores uploaded images
│   └── output/               # Stores processed images
└── README.md

```
4. **Run the Flask App**

```
python app.py
```
5. **You can proceed to the localhost once you see**
```
Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
6. **Using the app**

- Upload an Image (PNG/JPEG)

- Choose a Filter (e.g., Quantization, RGB Shift)

- Input parameters if required (e.g., bit depth or angle)

- Click “Apply Filter”

- The result appears in the Processed Image section

**The http://127.0.0.1:5000/ is the User Interface which deals with the output and preview of the processed image.**
