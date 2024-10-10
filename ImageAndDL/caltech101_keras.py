# CNN으로 분류해보기

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os
from PIL import Image

# 카테고리 정하기
categories = ['chair','camera','butterfly','elephant','flamingo']
nb_classes = len(categories)

# 이미지 크기 지정
image_w = 64
image_h = 64

# 1. 데이터 열기
data = np.load("./Deep-Learning/ImageAndDL/image/5obj.npz")
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# 데이터 정규화하기
X_train = X_train.astype("float") / 256
X_test = X_test.astype("float") / 256
print('X_train shape:', X_train.shape)

# 2. 모델 구축하기
def build_model(input_shape, nb_classes):
    model = Sequential()

    # 1. 첫 번째 Conv2D 레이어
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2. 두 번째 Conv2D 레이어
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3. Flatten 및 Dense 레이어
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

# 모델 컴파일하기 
input_shape = (image_w, image_h, 3)
model = build_model(input_shape, nb_classes)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 4. 모델 훈련하기
model.fit(X_train, y_train, batch_size=32, epochs=50)

# 5. 모델 평가하기 
score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])


# 6. 모델 가중치 저장하기 
hdf5_file = "./Deep-Learning/ImageAndDL/image/5obj-model.weights.h5"
"""if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 읽어 들이기
    model.load_weights(hdf5_file)
else:
    """
    # 학습한 모델을 파일로 저장하기
model.fit(X_train, y_train, batch_size=32, epochs=50)
model.save_weights(hdf5_file)    


# 가중치 불러오기
model.load_weights("/Users/baejuyeon/Documents/GitHub/Deep-Learning/ImageAndDL/image/5obj-model.weights.h5")
print("가중치를 정상적으로 불러왔습니다.")
'''
# 예측하기
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
    fname = "./Deep-Learning/ImageAndDL/image/error/" + str(i) + "-" + categories[pre_ans] + \
        "-ne-" + categories[ans] + ".PNG"
    dat *= 256
    img = Image.fromarray(np.uint8(dat))
    img.save(fname)


'''
