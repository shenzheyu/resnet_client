# ResNet Client

Create virtual environment and install packages.

```shell script
python3 -m venv venv
source ./venv/bin/activate
pip install tensorflow==1.14.0 tensorflow-serving-api==1.14.0 opencv-python
deactivate
```

Serve model via TF Serving

```shell script
sudo docker run -it --rm -p 8900:8500 --gpus all -v /home/shenz/resnet_client/models/resnet50:/models/resnet50 -e MODEL_NAME=resnet50 tensorflow/serving:latest-gpu --enable_batching=true
```