# Models

- Download and unzip [saved resnet50](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet/resnet50.tar.gz).

```shell script
wget http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NCHW.tar.gz
tar -zxvf resnet_v2_fp32_savedmodel_NCHW.tar.gz
```

- Use saved_model_cli to get the signature_def of this model.

```shell script
saved_model_cli show --all --dir ./resnet_v2_fp32_savedmodel_NCHW/1538687196/
```

>MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
>
>signature_def['predict']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input'] tensor_info:
        dtype: DT_FLOAT
        shape: (64, 224, 224, 3)
        name: input_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['classes'] tensor_info:
        dtype: DT_INT64
        shape: (64)
        name: ArgMax:0
    outputs['probabilities'] tensor_info:
        dtype: DT_FLOAT
        shape: (64, 1001)
        name: softmax_tensor:0
  Method name is: tensorflow/serving/predict
>
>signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input'] tensor_info:
        dtype: DT_FLOAT
        shape: (64, 224, 224, 3)
        name: input_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['classes'] tensor_info:
        dtype: DT_INT64
        shape: (64)
        name: ArgMax:0
    outputs['probabilities'] tensor_info:
        dtype: DT_FLOAT
        shape: (64, 1001)
        name: softmax_tensor:0
  Method name is: tensorflow/serving/predict