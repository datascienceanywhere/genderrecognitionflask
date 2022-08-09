import matplotlib.image as matimg
from flask import render_template, request
from app.face_recognition import faceRecognitionPipeline
import os
import cv2

UPLOAD_FOLDER = 'static/upload'


def index():
    return render_template('index.html')


def app():
    return render_template('app.html')


def genderApp():
    
    if request.method == "POST":
        f = request.files['image_name']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path)
        # get predictions
        pred_image, prediction = faceRecognitionPipeline(path)
        
        # save predicted image in predict folder
        pred_filename = 'predicted_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)
        
        # generate report
        report = []
        for i , obj in enumerate(prediction):
            obj_gray = obj['roi']
            eigen_image = obj['eig_img'].reshape(100,100)
            gender_name = obj['prediction_name']
            score = round(obj['score']*100,2)
            
            gray_img_name = f'roi_{i}.jpg'
            eig_img_name = f'eigen_{i}.jpg'
            
            matimg.imsave(f'./static/predict/{gray_img_name}',obj_gray,cmap='gray')
            matimg.imsave(f'./static/predict/{eig_img_name}',eigen_image,cmap='gray')
            
            # save report
            report.append([gray_img_name,
                           eig_img_name,
                           gender_name,
                           score])
            
        return render_template('gender.html',
                                fileupload=True,
                                img_name=pred_filename,
                                results=report)
            
            
        
        
    return render_template('gender.html',fileupload=False)