FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

RUN mkdir /data

RUN apt-get update 
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip install --upgrade pip

RUN pip install pandas numpy seaborn opencv-python scikit-learn
