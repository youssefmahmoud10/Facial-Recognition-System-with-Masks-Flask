from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import win32api

import os
import torch
from torch import nn
from torchvision import datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
from facenet_pytorch import MTCNN
import cv2
import time
import embeddingModels
from numpy import imag
import nbimporter
import TripletFolderClass
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1

app = Flask(__name__)

mtcnn0 = MTCNN(image_size=200, margin=0, keep_all=False, min_face_size=40) # keep_all=False
mtcnn = MTCNN(image_size=200, margin=0, keep_all=True, min_face_size=40) # keep_all=True

class InceptionResnet(nn.Module):
    def __init__(self, device, dropout=0.3):
        super(InceptionResnet, self).__init__()
        
        self.net = InceptionResnetV1(pretrained='vggface2', dropout_prob=dropout, device=device)
        self.out_features = self.net.last_linear.in_features
        
    def forward(self, x):
        return self.net(x)


class MaskNet(nn.Module):
    def __init__(self, model_name=None, dropout=0.0, embedding_size=512, device='cuda'):
        super(MaskNet, self).__init__()
        # Backbone
        self.model_name = model_name

        self.model = InceptionResnet(device, dropout=dropout)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # neck
        self.neck = nn.Sequential(
                nn.Linear(self.model.out_features, embedding_size, bias=True),
                nn.BatchNorm1d(embedding_size, eps=0.001),  
            )
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # backbone
        if self.model_name == None:
            return self.model(x)
        
masknet = MaskNet(model_name=None, embedding_size=512, dropout=0.3, device='cpu')
checkpoint = torch.load('./Models/InceptionResNetV1_Triplet.pth', map_location=torch.device('cpu'))

#MODEL_PATH = './Models/InceptionResNetV1_Triplet.pth'
#masknet.load_state_dict(checkpoint['model_state_dict'])

masknet.eval()

masknet.to('cpu')

dataset = datasets.ImageFolder('Database') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = [] # list of names corresponding to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True)
    if face is not None and prob>0.92:
        emb = masknet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])

# save data
data = [embedding_list, name_list]
torch.save(data, 'data.pt') # saving data.pt file

# Using webcam recognize face

# loading data.pt file
load_data = torch.load('data.pt')
embedding_list = load_data[0]
name_list = load_data[1]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/scan")
def scan():
    
    cascPath = r"haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    cam = cv2.VideoCapture(0)
    for i in range(300):
        ret, frame = cam.read()

        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )   

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.namedWindow("Face recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face recognition", 800, 600)
        cv2.moveWindow("Face recognition", 380, 110)
        cv2.imshow("Face recognition", frame)
        cv2.setWindowProperty("Face recognition", cv2.WND_PROP_TOPMOST, 1)

        if len(faces)>0 and i>50:
            path = r"scannedDatabase/"
            os.chdir(path) 
            cv2.imwrite("scannedPhoto.jpg", frame)
            os.chdir("../")
            break  

        if i==299:
            win32api.MessageBox(0, '  No faces detected. Try again  ', 'Sign up', 0x00001000)
            cam.release()
            cv2.destroyAllWindows()
            return ('', 204)

        k=cv2.waitKey(1)
        if k%256==27:
            break

    cam.release()
    cv2.destroyAllWindows()    

    return ('', 204)


@app.route("/signup", methods = ['POST', 'GET'])
def signup():

    if request.method == "POST":

        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
        def allowed_file(filename):
            return '.' in filename and \
                filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

        name = request.form["name"]
        image = request.files["image"]
    
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)  

            #Create directory if not exist
            path = 'Database/'+name
            if not os.path.exists(path):
                os.mkdir(path)

            UPLOAD_FOLDER = path
            app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        else:  
            imagePath = 'scannedDatabase/scannedPhoto.jpg'
            scannedImage = cv2.imread(imagePath)
            imageName = name+'.jpg'
            
            #Create directory if not exist
            path = 'Database/'+name
            if not os.path.exists(path):
                os.mkdir(path)

            os.chdir(path)
            cv2.imwrite(imageName, scannedImage)

            os.chdir('../../scannedDatabase/')
            os.remove('scannedPhoto.jpg')
            os.chdir('../')          	
    
        return render_template("login.html")
        
    return render_template("signup.html")

@app.route("/profile")
def profile():
    
    min_dist_to_close = 1

    cam = cv2.VideoCapture(0)

    for j in range(100):
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame, try again")
            break
            
        img = Image.fromarray(frame)
        img_cropped_list, prob_list = mtcnn(img, return_prob=True)
        
        if img_cropped_list is not None:
            boxes, _ = mtcnn.detect(img)
                    
            for i, prob in enumerate(prob_list):
                if prob>0.90:
                    #img_cropped_list = img_cropped_list.to('cuda')
                    emb = masknet(img_cropped_list[i].unsqueeze(0)).detach() 
                    
                    dist_list = [] # list of matched distances, minimum distance is used to identify the person
                    
                    for idx, emb_db in enumerate(embedding_list):
                        #dist = torch.dist(emb, emb_db).item()
                        dist = torch.linalg.norm(emb - emb_db).item()
                        #pdist = torch.nn.PairwiseDistance(p=2)
                        #dist = pdist(emb, emb_db)
                        dist_list.append(dist)

                    min_dist = min(dist_list) # get minumum dist value
                    min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                    name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                        
                    className=name
                    min_dist_to_close=min_dist

                    box = boxes[i] 
                    
                    original_frame = frame.copy() # storing copy of frame before drawing on it
                    
                    #if min_dist<0.90:
                        #similarity = ( 1 - min_dist) * 100
                        #str_sim = "{:.2f}".format(similarity)
                        #frame = cv2.putText(frame, name+' '+ str(min_dist),(int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)

                    frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)
        
        cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Recognition", 800, 600)
        cv2.moveWindow("Face Recognition", 380, 110)
        cv2.imshow("Face Recognition", frame)
        cv2.setWindowProperty("Face Recognition", cv2.WND_PROP_TOPMOST, 1)
        
        k = cv2.waitKey(1)
        if k%256==27: # ESC
            cam.release()
            cv2.destroyAllWindows()
            return ('', 204)

        if min_dist_to_close<0.65:
            break

        if j==70:
            win32api.MessageBox(0, '  Unrecognized face. Sign up now  ', 'Log in', 0x00001000)    
            cam.release()
            cv2.destroyAllWindows()
            return ('', 204)
        
    cam.release()
    cv2.destroyAllWindows()

    return render_template("profile.html", className=className)


if __name__ == "__main__":
    app.run()