import torch
import tensorrt as trt
import numpy as np
from collections import OrderedDict
import pycuda.driver as cuda
import pycuda.autoinit
from nn_utils import MinimalPartHOE
import os


class ModelWrapper:
    """Wrapper to handle model loading and preparation"""
    
    def __init__(self, checkpoint_path=None):
        self.model = MinimalPartHOE(
            img_size=(256, 192), 
            patch_size=16, 
            embed_dim=384, 
            depth=12, 
            num_heads=12
        )
        
        if checkpoint_path:
            # Load pretrained weights
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
    def get_model(self):
        return self.model


def export_to_onnx(model, output_path="parthoe_model.onnx"):
    """Export PyTorch model to ONNX format"""
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 192)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,  # Compatible with TensorRT 8.5
        do_constant_folding=True,
        input_names=['input'],
        output_names=['keypoints', 'orientation', 'confidence'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'keypoints': {0: 'batch_size'},
            'orientation': {0: 'batch_size'},
            'confidence': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {output_path}")
    return output_path


class TensorRTEngine:
    """Build and manage TensorRT engine"""
    
    def __init__(self, verbose=False):
        # Initialize TensorRT logger and builder
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = None
        self.parser = None
        self.config = None
        
    def build_engine(self, onnx_path, engine_path="parthoe_engine",
                 precision="fp16", min_batch_size=1, opt_batch_size=4, max_batch_size=6):
        """
        Build TensorRT engine from ONNX model
        
        precision: "fp32" or "fp16"
        """
        
        # Create network and parser
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(EXPLICIT_BATCH)
        self.parser = trt.OnnxParser(self.network, self.logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not self.parser.parse(model.read()):
                print("ERROR: Failed to parse ONNX file")
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                return None
        
        # Create builder config
        self.config = self.builder.create_builder_config()

        # Set memory pool size (updated API)
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Create optimization profile for dynamic shapes
        profile = self.builder.create_optimization_profile()
        profile.set_shape("input",
            (min_batch_size, 3, 256, 192),
            (opt_batch_size, 3, 256, 192),
            (max_batch_size, 3, 256, 192)
        )

        self.config.add_optimization_profile(profile)

        # Configure precision
        if precision == "fp16":
            self.config.set_flag(trt.BuilderFlag.FP16)
            print("Building with FP16 precision")
            
        else:
            print("Building with FP32 precision")

        # Build engine (updated API)
        print("Building TensorRT engine... This may take a while.")
        serialized_engine = self.builder.build_serialized_network(self.network, self.config)

        if serialized_engine is None:
            print("ERROR: Failed to build engine")
            return None

        # Save engine
        with open(engine_path+'_'+precision+'.trt', "wb") as f:
            f.write(serialized_engine)

        # Return deserialized engine object
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        print(f"Engine saved to {engine_path+'_'+precision+'.trt'}")
        return engine


class TensorRTInference:
    """Handle TensorRT inference"""
    
    def __init__(self, engine_path):
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get tensor info
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        # Allocate buffers
        self.buffers = {}
        self.stream = cuda.Stream()
    
    def _allocate_buffers(self):
        """Allocate GPU buffers for input/output"""
        buffers = {}
        
        for name in self.input_names + self.output_names:
            try:
                shape = self.engine.get_tensor_shape(name)
                print(f"Tensor {name}: shape={shape}")
                
                # Skip if shape contains -1 (unknown dimension)
                if any(dim == -1 for dim in shape):
                    print(f"Skipping {name} - dynamic shape, will allocate during inference")
                    continue
                    
                dtype = self.engine.get_tensor_dtype(name)
                
                # Convert TensorRT dtype to numpy
                if dtype == trt.DataType.FLOAT:
                    np_dtype = np.float32
                elif dtype == trt.DataType.HALF:
                    np_dtype = np.float16
                elif dtype == trt.DataType.INT8:
                    np_dtype = np.int8
                elif dtype == trt.DataType.INT32:
                    np_dtype = np.int32
                else:
                    np_dtype = np.float32
                
                size = trt.volume(shape)
                print(f"Allocating {size} elements of {np_dtype} for {name}")
                
                # Check if size is reasonable (< 100MB per tensor)
                if size * np_dtype().itemsize > 100 * 1024 * 1024:
                    print(f"Warning: Large allocation for {name}: {size * np_dtype().itemsize / 1024 / 1024:.1f} MB")
                
                host_mem = cuda.pagelocked_empty(size, np_dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                buffers[name] = {
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': np_dtype
                }
                
            except Exception as e:
                print(f"Error allocating buffer for {name}: {e}")
                continue
        
        return buffers
    
    def infer(self, input_data):
        """Run inference on input data"""
        input_name = self.input_names[0]
        
        # Set input shape
        self.context.set_input_shape(input_name, input_data.shape)
        
        # Allocate buffers dynamically if not already done
        if input_name not in self.buffers:
            self._allocate_dynamic_buffers()
        
        # Copy input data
        np.copyto(self.buffers[input_name]['host'], input_data.ravel())
        cuda.memcpy_htod_async(
            self.buffers[input_name]['device'], 
            self.buffers[input_name]['host'], 
            self.stream
        )
        
        # Set tensor addresses
        for name in self.input_names + self.output_names:
            self.context.set_tensor_address(name, int(self.buffers[name]['device']))
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy outputs
        results = []
        for name in self.output_names:
            cuda.memcpy_dtoh_async(
                self.buffers[name]['host'], 
                self.buffers[name]['device'], 
                self.stream
            )
        
        self.stream.synchronize()
        
        # Return outputs in correct shapes
        for name in self.output_names:
            output_shape = self.context.get_tensor_shape(name)
            results.append(self.buffers[name]['host'][:trt.volume(output_shape)].reshape(output_shape))
        
        return results

    def _allocate_dynamic_buffers(self):
        """Allocate buffers for dynamic shapes at runtime"""
        for name in self.input_names + self.output_names:
            if name not in self.buffers:
                shape = self.context.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
                
                if dtype == trt.DataType.FLOAT:
                    np_dtype = np.float32
                elif dtype == trt.DataType.HALF:
                    np_dtype = np.float16
                else:
                    np_dtype = np.float32
                
                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, np_dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.buffers[name] = {
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': np_dtype
                }


# Step 7: Main quantization pipeline
def quantize_parthoe_model(checkpoint_path, precision="fp16",
                           min_batch_size=1, opt_batch_size=4, max_batch_size=6, verbose=False):
    """
    Complete quantization pipeline
    
    Args:
        checkpoint_path: Path to PyTorch model checkpoint
        calibration_images: List of calibration images for INT8 (if needed)
        precision: "fp32" or "fp16"
    """
    
    print("Step 1: Loading PyTorch model...")
    model_wrapper = ModelWrapper(checkpoint_path)
    model = model_wrapper.get_model()
    
    print("Step 2: Exporting to ONNX...")
    onnx_path = export_to_onnx(model)
    
    print("Step 3: Preparing TensorRT builder...")
    trt_engine = TensorRTEngine(verbose=verbose)
    
    print(f"Step 4: Building TensorRT engine with {precision} precision...")
    engine = trt_engine.build_engine(
        onnx_path=onnx_path,
        precision=precision,
        min_batch_size=min_batch_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size,
        
    )
    
    if engine:
        print("Quantization completed successfully!")
        print("Engine file: parthoe_engine.trt")
        return f"parthoe_engine_{precision}.trt"
    else:
        print("Quantization failed!")
        return None

# Example usage
if __name__ == "__main__":

    min_batch_size = 1  # Minimal batch size
    opt_batch_size = 4  # Optimal batch size
    max_batch_size = 6  # Maximum batch size

    checkpoint_path="../weights/parthoe_s.pth"

    engine_path = quantize_parthoe_model(
        checkpoint_path=checkpoint_path,
        precision="fp16",
        min_batch_size=min_batch_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size,
        verbose=True
    )

    
    if engine_path:
        # Test inference
        print("Testing inference...")
        inference_engine = TensorRTInference(engine_path)
        
        # Create test input
        test_input = np.random.randn(1, 3, 256, 192).astype(np.float32)
        
        # Run inference
        results = inference_engine.infer(test_input)
        print(f"Inference completed. Got {len(results)} outputs.")
        print(f"Keypoints shape: {results[0].shape}")
        print(f"Orientation shape: {results[1].shape}")
        print(f"Confidence shape: {results[2].shape}")
