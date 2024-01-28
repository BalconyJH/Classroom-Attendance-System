# Copyright (C) 2020 coneypo
# SPDX-License-Identifier: MIT

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera
# Mail:     coneypo@foxmail.com

# 从人脸图像文件中提取人脸特征存入 "features_all.csv" / Extract features from images and save into "features_all.csv"

import os

import cv2
import dlib
import numpy as np

# 要读取人脸图像文件的路径 / Path of cropped faces
path_images_from_camera = "static/data/data_faces_from_camera/"

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor("app/static/data_dlib/shape_predictor_68_face_landmarks.dat")

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("app/static/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


def extract_face_features_128d(path_img: str):
    img_rd = cv2.imread(path_img)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray, 1)

    if faces:
        shape = predictor(img_rd, faces[0])
        return face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        print("无法在图片中检测到人脸 / No face detected in the image")
        return None


def calculate_average_face_features(path: str) -> np.ndarray:
    face_feature_vectors = []
    for photo_name in os.listdir(path):
        photo_path = os.path.join(path, photo_name)
        features_128d = extract_face_features_128d(photo_path)
        if features_128d is not None:
            face_feature_vectors.append(features_128d)

    if face_feature_vectors:
        return np.array(face_feature_vectors).mean(axis=0)
    return np.zeros(128, dtype=int, order="C")
