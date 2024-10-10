from keras.models import Sequential
import numpy as np
from PIL import Image
import os
from caltech101_keras import build_model  # build_model 함수 가져오기

# 카테고리 설정
categories = ['chair', 'camera', 'butterfly', 'elephant', 'flamingo']
nb_classes = len(categories)

# 이미지 크기 지정
image_w = 64
image_h = 64

# 1. 데이터 열기
data = np.load("./Deep-Learning/ImageAndDL/image/5obj.npz")
X_test = data['X_test']
y_test = data['y_test']

# 데이터 정규화하기
X_test = X_test.astype("float") / 256

# 2. 모델 정의
input_shape = (image_w, image_h, 3)
model = build_model(input_shape, nb_classes)

# 3. 저장된 가중치 불러오기
model_file = "/Users/baejuyeon/Documents/GitHub/Deep-Learning/ImageAndDL/image/5obj-model.weights.h5"
model.load_weights(model_file)

# 4. 예측하기
pre = model.predict(X_test)

# 오류 폴더가 없으면 생성
error_dir = "./Deep-Learning/ImageAndDL/image/error/"
if not os.path.exists(error_dir):
    os.makedirs(error_dir)

# 예측 결과 테스트하기
for i, v in enumerate(pre):
    pre_ans = v.argmax()  # 예측한 레이블
    ans = y_test[i].argmax()  # 정답 레이블
    dat = X_test[i]  # 이미지 데이터

    if ans == pre_ans:
        continue

    # 예측이 틀리면 무엇이 틀렸는지 출력하기
    print("[NG]", categories[pre_ans], "!=", categories[ans])
    print(v)

    # 이미지 출력하기
    fname = error_dir + str(i) + "-" + categories[pre_ans] + "-ne-" + categories[ans] + ".PNG"
    dat *= 256
    img = Image.fromarray(np.uint8(dat))
    img.save(fname)
