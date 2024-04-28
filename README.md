# Rapid Multimodal/Multilingual Query Resolution using ANNOY and USearch (Web App)

This project pioneers a system that simplifies the search for visual content by understanding and responding to various search inputs, including images, text, and audio. A unique feature of this project is developing a method that can quickly sift through a massive collection of 1 million images from the SBU Captions Dataset to find the most similar to a user’s request. This process is further enhanced by using the SigLIP model to extract image features from the SBU Captions dataset. The approach uses cutting-edge algorithms like ANNOY (Approximate Nearest Neighbors Oh Yeah) and USearch that are fast, accurate, and capable of delivering results in milliseconds. Additionally, the system is being enhanced with the ability to recognize specific objects within images using the YOLO (You Only Look Once) API, allowing for even more precise searches. A unique feature of this project is its multilingual capability, which enables searches in several languages, making it accessible to users worldwide. This project is about creating a new tool and setting a new standard for finding and interacting with visual information. 

## Pre-Requisites:
1. Download the SBU Captions Dataset from http://www.cs.rice.edu/~vo9/sbucaptions/sbu_images.tar
2. Install Python3 and required libraries as per the jupyter notebook files
3. Download the 1 Million USearch Index file from https://drive.google.com/file/d/1_xii_xGebGcmsM6_slzKGqq3_HN7CExT/view?usp=sharing
4. Download the 1 Million Annoy Index file from 
5. Download the 1 Million Text/Captions Index file from 
6. Download Hindi 8K Flickr Dataset from https://www.kaggle.com/code/dsmeena/image-captioning-with-flickr8k-hindi-using-pytorch/input
7. Download `trained_model.pth` for Hindi supported fine-tuned model from https://drive.google.com/file/d/1OzBEQp7dPdLFUj8rVzur9O1SnngBofwj/view?usp=sharing

## Code Structure:

**server.py** -> 

**GenerateGroundTruthJson.ipynb** -> This jupyter file helps to get the ground truth values for selected 1000 images from the SBU Captions dataset using the YOLO API. It generates a json file with key values as unique objects detected from the input images and values as the frequency by which that specific object has appeared in all images. Note: If the same object is appeared more than once in an image then we count it only once. 

**AnnoyAccuracyTest.ipynb** -> The above generated `ground_truth.json` file is used in this code to test the accuracy of ANNOY algorithm. We use the YOLO api again on the k-similar images output from those 1000 images Annoy index. The YOLO api helps to get the number of true positives based on an input image of a specific class like "car", "chair", "person", etc. We finally divide the true positives found by the total number of objects of same class really present (ground truth value) to get the accuracy metric. 

**USearchAccuracyTest.ipynb** -> This is similar to `AnnoyAccuracyTest.ipynb` file, the only difference being the usage of USearch algorithm instead of Annoy to measure its accuracy.

**Fine-TuneSigLIPForHindi.ipynb** -> This file is responsible for fine-tuning the existing SigLIP model to support Hindi Language as well for querying an image. The output of this file `trained_model.pth` can be used easily for testing its capabilities using PyTorch. We can load the pre-trained model easily using below lines and then perfrom whatever operation we want: 

```
# Load the pre-trained weights
model.load_state_dict(torch.load('path_to_your_model.pth'))

# Set the model to evaluation mode
model.eval()
```