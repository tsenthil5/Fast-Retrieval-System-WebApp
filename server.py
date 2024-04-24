from flask import Flask, request, redirect, url_for, render_template, session
from PIL import Image
import requests
from transformers import AutoModel, AutoProcessor
import torch
import pandas as pd
import faiss
import ast
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, SiglipTextModel
import mysql.connector as mysql
import time
import os
from flask import Flask, redirect, url_for
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

similar = []
image_url = ""
#db = mysql.connect(host = "localhost",
 #                   user = "root",
 #                   passwd = "Test123",
 #                   database = "Image_Path")
#cursor = db.cursor()

textModel = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
imgIndex = faiss.read_index("multi1MImageVectors.index")
textIndex = faiss.read_index('faissTextVector.index')
Imagemodel = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
Yolomodel = YOLO('yolov8x.pt')
Multitokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", low_cpu_mem_usage=True, do_rescale=False)
device_type = "cpu"
device = torch.device(device_type)
Imagemodel.to(device)
transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
app = Flask(__name__)


@app.route('/', methods = ['POST', 'GET'])
def index():
    return render_template('index.html', page=1, total_pages=10)


@app.route('/textSubmit', methods = ['POST', 'GET'])
def textSubmit():
    global similar
    global image_url
    image_url = None
    if request.method == "POST":
        text = request.form['query']
        #start = time.time()
        similar, k = TextSimilar(text, 1000)
        #end = time.time()
        #print("text time calculation", end-start)
        return redirect(url_for('pagination'))
    
@app.route('/textMultiLingual', methods = ['POST', 'GET'])
def textSubmitMulti():
    global similar
    global image_url
    image_url = None
    if request.method == "POST":
        text = request.form['query']
        #start = time.time()
        similar, k = TextSimilarMulti(text, 1000)
        #end = time.time()
        #print("text time calculation", end-start)
        return redirect(url_for('pagination'))

def TextSimilarMulti(text, k):
    #model.eval()
    inputs = Multitokenizer([text], padding="max_length", return_tensors="pt")
    with torch.no_grad():
        text_features = Imagemodel.get_text_features(**inputs)
    _, I = imgIndex.search(text_features, k) 
    similarList = I.tolist()
    similarPath = indices_to_images(similarList[0])
    return similarPath, k

        
        
def indices_to_images(indices):
    image_paths = []
    for i in indices:
        folder_name = i // 1000
        file_name = i % 1000
        folder_str = f"{folder_name:04d}"
        file_str = f"{file_name:03d}.jpg"
        image_paths.append("https://storage.googleapis.com/vislang-public/sbu-images" + "/" + folder_str + "/" + file_str)
    return image_paths
def TextSimilar(text, k):
    text = text
    max_length = textModel.config.max_position_embeddings
    inputs = tokenizer([text], padding="max_length", truncation=True,
    max_length=max_length, return_tensors="pt")
    outputs = textModel(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output
    feature_vectors = pooled_output.detach().numpy()
    _, I = textIndex.search(feature_vectors, k) 
    similarList = I.tolist()
    similarPath = indices_to_images(similarList[0])
    return similarPath, k

def ExtractYoloImages(image_path):
    yolo_results_list = Yolomodel([image_path])
    img = Image.open(image_path)
    cnt = 0
    img_labels = []
    img_files = [(image_path.split('/')[-1], "Original Image")]
    if len(yolo_results_list[0]) <= 1:
        # Only one or zero objects found
        return img_files
    for result in yolo_results_list[0]:
        img_label = result.names[int(result.boxes.cls[0])]
        img_labels.append(img_label)
        for x1, y1, x2, y2 in result.boxes.xyxy:
            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract the object patch
            patch = img.crop((x1, y1, x2, y2))
            path = "static/images/{}.jpg".format(img_label)
            # Save the patch
            patch.save(path, quality=100)
            img_files.append((img_label + ".jpg", "Detected: {}".format(img_label)))
            cnt += 1
    print("Number of objects found: {} with labels: {}".format(cnt, img_labels))
    return img_files

@app.route('/imageSubmit', methods = ['POST', 'GET'])
def imageSubmit():
    global similar
    global image_url
    if request.method == "POST":
        image = request.files['image']

        filepath = os.path.join('static/images', image.filename)
        print("filepath", filepath)
        image.save(filepath)
        img_files = ExtractYoloImages(filepath)
        # print(img_files)
        return render_template('index.html', files=img_files, page=1, total_pages=1)


@app.route('/imageQuery', methods=['POST', 'GET'])
def imageQuery():
    global similar
    global image_url
    if request.method == "POST":
        #print(request)
        data = request.form.get('image_url')
        image_path = data[1:]
        filename = image_path.split('/')[-1]
        # print(image_path)
        # filepath = os.path.join('static/images', image.filename)

        # image.save(filepath)
        print("image path", image_path)
        #image = Image.open(image_path)
        # start = time.time()
        similar, k = ImageSimilar(image_path, 500)

        # print(len(similar))
        # end = time.time()
        # print("image time calculation", end-start)
        image_url = url_for('static', filename=f'images/{filename}')
        return redirect(url_for('pagination'))
    
@app.route("/pagination")
def pagination():
    global image_url
    page = request.args.get('page', 1, type=int)
    per_page = 16  # Number of images per page
    start = (page - 1) * per_page
    end = start + per_page
    total_pages = (len(similar) + per_page - 1) // per_page
    paginated_images = similar[start:end]
    print(paginated_images)
    return render_template('index.html', image_url=image_url, path_array=paginated_images, page=page, total_pages=total_pages)

# @app.route("/pagination")
# def pagination():
#     global image_url
#     page = request.args.get('page', 1, type=int)
#     per_page = 16  # Number of images per page
#     start = (page - 1) * per_page
#     end = start + per_page
#     total_pages = (len(similar) + per_page - 1) // per_page
#     paginated_images = similar[start:end]
#     return render_template('index.html', image_url=image_url, path_array=paginated_images, page=page, total_pages=total_pages)
        
        
def indices_to_images(indices):
    image_paths = []
    for i in indices:
        folder_name = i // 1000
        file_name = i % 1000
        folder_str = f"{folder_name:04d}"
        file_str = f"{file_name:03d}.jpg"
        image_paths.append("https://storage.googleapis.com/vislang-public/sbu-images" + "/" + folder_str + "/" + file_str)
    return image_paths

def ImageSimilar(path, k):
    
    newImage = Image.open(path)
    inputs = transform(newImage)
    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
    inputs = processor(images=inputs, return_tensors="pt").to(device)
    features = Imagemodel.get_image_features(**inputs)
    features = np.array(features.tolist())
    _, I = imgIndex.search(features, k) 
    similarList = I.tolist()
    similarPath = indices_to_images(similarList[0])
    return similarPath, k

if __name__ == '__main__':
    app.run(debug = True)