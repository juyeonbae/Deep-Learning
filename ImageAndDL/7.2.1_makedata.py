# CNN으로 이미지 분류하기 - 이미지 데이터를 파이썬 데이터로 변환하기 

from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split  # pip3 install scikit-learn

# 1. 분류 대상 카테고리 선택하기 - 이미지 세트가 들어있는 폴더 이름, 분류 대상, 카테고리 지정
caltech_dir = "./Deep-Learning/ImageAndDL/image/101_ObjectCategories"
categories = ['chair','camera','butterfly','elephant','flamingo']
nb_classes = len(categories)

# 2. 이미지 크기 지정 - RGB 값을 나타내는 3개의 데이터가 필요 => 하나의 이미지는 12,288 요소로 나타낸다. 
image_w = 64
image_h = 64
pixels = image_w * image_h * 3

# 3. 이미지 데이터 읽어 들이기
X = []  # 실제 이미지 데이터 
Y = []  # 이미지가 어떤 것을 나타내는지 설명하는 레이블 데이터 
for idx, cat in enumerate(categories):
    # 4. 레이블 지정 - 레이블 생성(카테고리 수만큼 요소를 가짐)
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 5. 이미지
    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")  # glob함수 -> 확장자가 .jpg 인 것만 찾을 수 있다. 
    
    for i, f in enumerate(files):
        img = Image.open(f)  # 6. 이미지 파일을 읽고, 색상 모드를 RGB로 변환하고 64 x 64 픽셀로 리사이즈한다. 
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)  # asarray() 메서드를 사용해 PIL의 Image 데이터를 Numpy 배열 데이터로 변환한다. 

        X.append(data)
        Y.append(label)



X = np.array(X)
Y = np.array(Y)

print(f"X shape: {np.shape(X)}")  # X는 (샘플 수, 64, 64, 3)의 형태여야 함
print(f"Y shape: {np.shape(Y)}")  # Y는 (샘플 수, nb_classes)의 형태여야 함


# 7. 학습 전용 데이터와 테스트 전용 데이터 구분
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.savez("./Deep-Learning/ImageAndDL/image/5obj.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("ok,", len(Y))


