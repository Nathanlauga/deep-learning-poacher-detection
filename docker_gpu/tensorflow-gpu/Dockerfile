FROM tensorflow/tensorflow:2.2.0rc2-gpu-py3-jupyter

RUN mkdir /data

RUN apt-get update 
RUN apt-get install -y libsm6 libxext6 libxrender-dev graphviz

RUN pip install --upgrade pip

RUN pip install pandas numpy seaborn opencv-python scikit-learn easydict tqdm pydot==1.2.3
# RUN pip install mlflow==1.7.2 