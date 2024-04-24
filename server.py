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

@app.route('/imageSubmit', methods = ['POST', 'GET'])
def imageSubmit():
    global similar
    global image_url
    if request.method == "POST":
        image = request.files['image']
        filepath = os.path.join('static/images', image.filename)
        
        image.save(filepath)
        print("filepath", filepath)
        #start = time.time()
        similar, k = ImageSimilar(image, 500)
        
        #print(len(similar))
        #end = time.time()
        #print("image time calculation", end-start)
        image_url = url_for('static', filename=f'images/{image.filename}')
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
    return render_template('index.html', image_url=image_url, path_array=paginated_images, page=page, total_pages=total_pages) 
        
        
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