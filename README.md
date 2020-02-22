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
sudo docker run -it --rm -p 8900:8500 --gpus all -v /home/shenz/resnet_client/models/resnet_v2_fp32_savedmodel_NCHW/:/models/resnet_v2 -e MODEL_NAME=resnet_v2 tensorflow/serving:latest-gpu --enable_batching=true
```

Run client.py

```shell script
python client.py -server=127.0.0.1:8900 -batch_size=1 -img_path=./examples/000000000057.jpg
```

Run batch_test.py

```shell script
python batch_test.py -server=127.0.0.1:8900
```