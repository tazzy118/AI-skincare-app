from flask import Flask, render_template, request, url_for
import os
import cv2
from ultralytics import YOLO
import supervision as sv

#initialising flask application. its a lightweight framework for python applications too.
app = Flask(__name__)

#Loading the trained YOLO model for skin condition detection.
#https://www.datacamp.com/blog/yolo-object-detection-explained
#this line loads the pre-trained model last.pt, which has been trained to identify skin conditions.
model = YOLO("last.pt")

#set upload folder and allowed extensions.
#this also ensures the user can only upload images and not unsupported formats like .txt or .pdf.
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

#dictionary mapping detected skin conditions to their respective skincare routines. Ive only done it for acne and eczema(dry skin) since 
#it was diffcult trying to make it detect oily skin. The ai can detect differnt skin conditions too but my focus was acne,dry and oily.
#if YOLO detects a particular condition, the corresponding routine steps will be retrieved and displayed.
SKINCARE_ROUTINES = {
    "Acne": [
        {"step": "Cleanse", "image": "https://acnecide.co.uk/static/3b88e71e1c7110e39efc816818aade06/bc92f/Acnecide_Face_Wash_1_fd588b2c7e.avif", "description": "Acnecide Gel Wash: Use a gentle cleanser to clean acne bacteria on the face."},
        {"step": "Serum", "image": "https://www.skincarebydrv.com/cdn/shop/products/TrioBlemishExfoliatorbox_bottle.jpg?v=1651737739&width=1800", "description": "Trio Blemish Exfoliator: A gentle exfoliator, safe for skin of color, to help manage oily skin, improve acne, red marks, and brown marks."},
        {"step": "Moisturize", "image": "https://acnecide.co.uk/static/ca90fe05ec594c9374160462f340904d/f7f53/Purifide_Daily_Moisturiser_SPF_1_9d12d952f5.avif", "description": "Purifide Daily Moisturiser: A lightweight moisturizer, great for acne-prone skin."}
    ],
    "Eczema": [
        {"step": "Cleanse", "image": "https://www.skincarebydrv.com/cdn/shop/products/DrVanitaRattanMicellarGelWashboxandbottle_Darker.jpg?v=1668448357&width=720", "description": "DR V Micellar Gel Wash: A gentle, NAFE-safe cleanser, good for dry skin."},
        {"step": "Serum", "image": "https://www.cerave.com/-/media/project/loreal/brand-sites/cerave/americas/us/products/hydrating-hyaluronic-acid-serum/700x875/hydrating-hyaluronic-acid-serum-front-700x875-v1.jpg?rev=6f8323a8d08440db804e8c0824b6381d&w=900&hash=3D75534722EFD1805E456C6CBE67D3EC", "description": "Hyaluronic Acid Serum: Great for hydrating dry skin."},
        {"step": "Barrier Protection", "image": "https://www.cerave.com/-/media/project/loreal/brand-sites/cerave/americas/us/products-v4/moisturizing-cream/cerave_moisturizing_cream_16oz_jar_front-700x875-v4.jpg?rev=db6e3c22250e4928bc749dd2c207de5b&w=900&hash=E68F77D279494CD09396613CB8496EB7", "description": "CeraVe Moisturizing Cream: Use a cream to lock in moisture, good for dry skin."}
    ]
}


#this function is validate uploaded files (checks if the file is an accepted image type).
#this function ensures that users are only submitting allowed file types.
#https://stackoverflow.com/questions/46136478/flask-upload-how-to-get-file-name
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#this function is to process uploaded images and apply skin condition detection https://stackoverflow.com/questions/24564889/opencv-not-opening-images-with-imread
def process_image(input_image_path: str, output_image_path: str):
    #reads the uploaded image using OpenCV
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return [] #if the image cannot be loaded, return an empty list
    #resizes the image to 640x640 pixels which is needed for the mdoels compatibility.
    resized = cv2.resize(image, (640, 640))
    
    #runs the mdoels object detection on the resized image.
    #https://stackoverflow.com/questions/77723160/ultralytics-yolo-error-when-trying-to-extract-bounding-box-coordinates-from-res
    results = model(resized)[0]
    #extracts detected skin conditions from the model
    #https://stackoverflow.com/questions/75740000/application-keeps-crashing-and-is-very-laggy-indexerror-index-0-is-out-of-boun
    detected_classes = []
    if results.boxes is not None and results.boxes.cls is not None:
        class_indices = results.boxes.cls.cpu().numpy()
        detected_classes = [model.names[int(idx)] for idx in class_indices]  #correct name extraction

    #annotate image with bounding boxes and detects labels using uupervision
    #https://supervision.roboflow.com/annotators/
    detections = sv.Detections.from_ultralytics(results)
    annotated = sv.BoundingBoxAnnotator().annotate(scene=resized, detections=detections)
    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections)

    #saves the processed image with annotations
    cv2.imwrite(output_image_path, annotated)
    print(f"Processed and saved: {output_image_path}")
    
    return detected_classes  #this returns a list of detected skin conditions. for e.g if it says acne,acne,ecxema it would be acne since its 2/3.

#flask route to handle image uploads, process images, and display results
#https://stackoverflow.com/questions/55079926/do-i-need-to-use-methods-get-post-in-app-route
@app.route('/', methods=['GET', 'POST'])
def upload_files():
    processed_files = [] #stores paths of processed images
    detected_conditions = [] #the detected skin conditions stored here
    skincare_routines = []  #stores the skincare routines
    #got stuck with this but had help from cousin and this https://stackoverflow.com/questions/71158665/how-to-get-path-of-all-uploaded-files-in-flask
    if request.method == 'POST':
        files = request.files.getlist('files') #retrieves uploaded files.

        for file in files:
            if file and allowed_file(file.filename): #checks file type.
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)

                output_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'annotated_' + file.filename)
                detected_classes = process_image(filename, output_filename)  #when the image is processed it gets detected for conditions.
                
                processed_file_url = url_for('static', filename=f'outputs/annotated_{file.filename}')
                processed_files.append(processed_file_url)
                #https://stackoverflow.com/questions/40844719/initialize-a-set-as-argument-of-a-function
                detected_conditions.extend(detected_classes) #detected conditions
                #ensure skincare routines are added only once per detected condition.
                added_conditions = set()  #to track unique conditions beacuse before it would output the skincare routine twice.

                for condition in detected_classes:
                    if condition in SKINCARE_ROUTINES and condition not in added_conditions:
                        skincare_routines.extend(SKINCARE_ROUTINES[condition])  #adds only once
                        added_conditions.add(condition)  #mark condition as added

    #https://stackoverflow.com/questions/65318395/how-to-render-html-for-variables-in-flask-render-template
    #renders index.html with processed images, detected conditions & skincare routines.
    return render_template('index.html', processed_files=processed_files, detected_conditions=detected_conditions, skincare_routines=skincare_routines) #passes the corrected variable

#ensures required folders are in/exists and starts flask server
#https://stackoverflow.com/questions/273192/how-do-i-create-a-directory-and-any-missing-parent-directories
if __name__ == "__main__":
    # Create necessary folders if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    app.run(debug=True)

#references
# Skin Disease Dataset (Roboflow Universe)
# @misc{skin-disease-vrvtv_dataset,
#   title = { Skin Disease Dataset },
#   author = { Skin Disease },
#   publisher = { Roboflow },
#   year = { 2025 },
#   url = { https://universe.roboflow.com/skin-disease/skin-disease-vrvtv }
# }
#https://www.tutorialspoint.com/python/python_file_handling.htm