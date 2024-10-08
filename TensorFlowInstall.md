### macOS 설치
- macOS 패키지 매니저 Homebrew 설치 
- /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
- pyenv, anaconda 설치

1. pyenv install
    $ brew update
    $ brew install pyenv 

2. pyenv 경로 설정
    $ echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
    $ exec $SHELL -l
    * zsh 사용하는 경우
    $ echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    $ source ~/.zshrc

3. python3 설치
    $ pyenv install anaconda3-5.2.0

4. python3 활성화
    $ pyenv global anaconda3-5.2.0
    $ pyenv rehash

5. pip 명령어 설치
    $ easy_install pip

6. pip 업데이트
    $ pip install -U pip

7. Tensorflow(CPU version) install
    $ pip install -U tensorflow

8. 추가 라이브러리 
    $ pip install -U h5py graphviz pydot

* tensorflow 설치 확인
    python3 
    import tensorflow as tf
    tf.__version__

* anaconda 설치 
    https://www.anaconda.com/download

* jupyter notebook 설치
    pip3 install --upgrade pip
    pip3 install jupyter
    jupyter notebook 