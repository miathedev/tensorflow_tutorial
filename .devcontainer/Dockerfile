FROM tensorflow/tensorflow

#Add matplotlib
RUN pip install matplotlib pandas autopep8

#Add openCV
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx file git
RUN pip install opencv-python
