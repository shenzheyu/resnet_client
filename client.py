import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time
import cv2
import numpy as np

tf.app.flags.DEFINE_string('server', '127.0.0.1:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_integer('batch_size', 2, 'No of images to batch -32,64,128  ')
tf.app.flags.DEFINE_string('img_path', '', 'Path to image')
FLAGS = tf.app.flags.FLAGS

_counter = 0
_start = 0
# load label to names mapping for visualization purposes


def parse_result(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.
    """
    global _counter
    global _start

    scores = result_future.outputs['probabilities']
    labels = result_future.outputs['classes']
    scores = tf.make_ndarray(scores)
    labels = tf.make_ndarray(labels)

    print("result no", _counter)
    print("scores output", (scores).shape)
    print("labels output", (labels).shape)

    # visualize detections
    for label, score in zip(labels, scores):
        # scores are sorted so we can break
        print("Label",label," Score ",score)
        # print("Label", label)
    _counter += 1
    end = time.time()


def do_inference(server, batch_size, img_path):
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet_v2'
    request.model_spec.signature_name = 'serving_default'

    # Going to read the image
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = image.astype('f')
    print("in image shape", image.shape)

    global _start
    _start = time.time()
    # print("Going to send the request")
    # try batching
    input = np.expand_dims(image, axis=0)
    inputs = np.append(input, input, axis=0)
    for _ in range(batch_size - 2):
        inputs = np.append(inputs, input, axis=0)

    print("in tf shape", inputs.shape)
    request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto
                                     (inputs, shape=inputs.shape))

    result = stub.Predict(request, 10.25)  #  seconds
    parse_result(result)

    print("Response Received Exiting")


def main(_):
    if not FLAGS.server:
        print('please specify server -server host:port')
        return
    if not FLAGS.img_path:
        print('please specify image path -img_path')
        return
    do_inference(FLAGS.server, FLAGS.batch_size, FLAGS.img_path)


if __name__ == '__main__':
    tf.app.run()