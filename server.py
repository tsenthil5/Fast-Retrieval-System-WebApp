from flask import Flask, request, redirect, url_for, render_template
from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import faiss
import numpy as np
from usearch.index import Index
from transformers import AutoTokenizer, SiglipTextModel
import time
import os
from flask import Flask, redirect, url_for
from ultralytics import YOLO
from annoy import AnnoyIndex
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
key = np.arange(1000000)
imgIndex = Index(ndim=1152)
imgIndex.load("image1MVectors.usearch")
similar = []
image_url = ""
textModel = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
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
        similar, k = TextSimilar(text, 100)
        return redirect(url_for('pagination'))
    
@app.route('/textMultiLingual', methods = ['POST', 'GET'])
def textSubmitMulti():
    global similar
    global image_url
    image_url = None
    if request.method == "POST":
        text = request.form['query']
        similar, k = TextSimilarMulti(text, 1000)
        return redirect(url_for('pagination'))

def TextSimilarMulti(text, k):
    inputs = Multitokenizer([text], padding="max_length", return_tensors="pt")
    with torch.no_grad():
        text_features = Imagemodel.get_text_features(**inputs)
        text_features = np.array(text_features.tolist()) 
    start = time.time()
    matches = imgIndex.search(text_features, k)
    end = time.time()
    similarPath = indices_to_images(list(matches.keys))
    
    print("text Multi time calculation", end-start)
    return similarPath, k

        
        
def indices_to_images(indices):
    indices = [int(i) for i in indices]
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
    yolo_results_list = Yolomodel([image_path], conf = 0.75)
    img = Image.open(image_path)
    cnt = 0
    img_labels = []
    img_files = [(image_path.split('/')[-1], "Original Image")]
    uniqueId = 0
    if len(yolo_results_list[0]) <= 1:
        return img_files
    for result in yolo_results_list[0]:
        detected = result.names[int(result.boxes.cls[0])]
        img_label = detected + str(uniqueId)
        img_labels.append(img_label)
        for x1, y1, x2, y2 in result.boxes.xyxy:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            patch = img.crop((x1, y1, x2, y2))
            path = "static/images/{}.jpg".format(img_label)
            patch.save(path, quality=100)
            img_files.append((img_label + ".jpg", "Detected: {}".format(detected)))
            cnt += 1
            uniqueId+=1
    return img_files

@app.route('/imageSubmit', methods = ['POST', 'GET'])
def imageSubmit():
    global similar
    global image_url
    if request.method == "POST":
        image = request.files['image']
        filepath = os.path.join('static/images', image.filename)
        image.save(filepath)
        img_files = ExtractYoloImages(filepath)
        intro_text = "Select any of the detected objects for similarity search:"
        return render_template('index.html', files=img_files, page=1, total_pages=1, intro_text = intro_text)


@app.route('/imageQuery', methods=['POST', 'GET'])
def imageQuery():
    global similar
    global image_url
    if request.method == "POST":
        data = request.form.get('image_url')
        image_path = data[1:]
        filename = image_path.split('/')[-1]
        newImage = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        inputs = transform(newImage)
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        inputs = processor(images=inputs, return_tensors="pt").to(device)
        features = Imagemodel.get_image_features(**inputs)
        features = np.array(features.tolist())
        similar, k = ImageSimilar(features, 500)
        image_url = url_for('static', filename=f'images/{filename}')
        return redirect(url_for('pagination'))
    
@app.route("/pagination")
def pagination():
    global image_url
    page = request.args.get('page', 1, type=int)
    per_page = 16 
    start = (page - 1) * per_page
    end = start + per_page
    total_pages = (len(similar) + per_page - 1) // per_page
    paginated_images = similar[start:end]
    return render_template('index.html', image_url=image_url, path_array=paginated_images, page=page, total_pages=total_pages)
        
def indices_to_images(indices):
    indices = [int(i) for i in indices]
    image_paths = []
    for i in indices:
        folder_name = i // 1000
        file_name = i % 1000
        folder_str = f"{folder_name:04d}"
        file_str = f"{file_name:03d}.jpg"
        image_paths.append("https://storage.googleapis.com/vislang-public/sbu-images" + "/" + folder_str + "/" + file_str)
    return image_paths

def ImageSimilar(features, k):
    matches = imgIndex.search(features.flatten(), k)
    similarPath = indices_to_images(list(matches.keys))
    return similarPath, k

if __name__ == '__main__':
    app.run(debug = True)