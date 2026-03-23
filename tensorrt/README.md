# Quantization Scripts for PartHOE

This folder contains the scripts needed to quantize the PartHOE-S model, as well as a stand-alone class for inference only. 
To run this, you will need to install TensorRT via pip:

```bash
pip install tensorrt
```

Since TensorRT does optimization based on the detected hardware, you will need to run the quantization scripts on a computer with the same GPU as the final deployment machine you intend to run the models in. 

Also keep in mind to check the version of TensorRT you are running, since TensorRT engine files are not cross-compatible:

```python
import tensorrt
print(tensorrt.__version__)
```

## Launching the quantization script - `quantize_parthoe.py`

Before running the script, make sure that `checkpoint_path` variable points to your `parthoe_s.pth` Pytorch weights. 

Feel free to play with the minimal, optimal and max batch sizes set by default depeding on your application. 

There is also a `verbose` flag available if you do not want to see all the logs that result from the different optimizations run by TensorRT when quantizing the model.

After the quantization, the script will run a test with random tensors to make sure that the dynamic batching is working properly and that the outputs are shaped as expected. 

## Running the quantized model - `PartHOE_TRT.py`

This file contains the standalone inference class to run the quantized model. 

Make sure to point the `engine_path` variable to your quantized engine file. Then the script will use the sample images under `testing_imgs` to run PartHOE-S on real images. 

If you plan to use this class on your implementation, keep in mind that the inputs to PartHOE need to be normalized with the COCO dataset's mean and std. 