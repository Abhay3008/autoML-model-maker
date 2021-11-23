FROM centos 
RUN yum update -y 
RUN yum install python36 -y 
RUN yum install epel-release -y 
RUN yum update -y 
RUN curl https://bootstrap.pypa.io/get-pip.py" "get-pip.py"
RUN python3 get-pip.py 
RUN pip install setuptools
RUN pip install pillow 
RUN pip install keras==2.1.5 
RUN pip install scikit-learn 
RUN pip install opencv-python
RUN pip install opencv-python-headless 
RUN pip install pandas
RUN pip install tensorflow==1.5
RUN pip install matplotlib 
RUN pip install seaborn
