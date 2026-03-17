import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import time
import cv2
from pathlib import Path

class TensorRTInference:
    """Handle TensorRT inference with variable batch sizes"""
    
    def __init__(self, engine_path, max_batch_size=32):

        try:
            current_context = cuda.Context.get_current()
            print(f"Current CUDA context: {current_context}")
            print(f"Context device: {current_context.get_device()}")
            print(f"Context API version: {current_context.get_api_version()}")
        except cuda.LogicError as e:
            print(f"No CUDA context: {e}")
            import pycuda.autoinit
            current_context = cuda.Context.get_current()
            print(f"After autoinit - Context: {current_context}")
        
        # Test context is working
        try:
            cuda.Context.synchronize()
            print("CUDA context is active and synchronized")
        except Exception as e:
            print(f"CUDA context sync failed: {e}")

        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise Exception("Failed to deserialize TensorRT engine")
        else:
            print(f"Engine deserialized successfully.")

        self.context = self.engine.create_execution_context()

        if self.context == None:
            raise Exception("Execution context is None!")

        self.max_batch_size = max_batch_size
        
        # Get tensor info
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        # Pre-allocate buffers for maximum batch size
        self.buffers = {}
        self.stream = cuda.Stream()
        self._allocate_max_buffers()
    
    def _allocate_max_buffers(self):
        """Pre-allocate GPU buffers for maximum batch size to avoid reallocation"""
        for name in self.input_names + self.output_names:
            try:
                base_shape = list(self.engine.get_tensor_shape(name))
                
                # Replace batch dimension (-1 or first dim) with max_batch_size
                if base_shape[0] == -1:
                    max_shape = [self.max_batch_size] + base_shape[1:]
                else:
                    # If not dynamic, use original shape but prepare for batching
                    max_shape = [self.max_batch_size] + base_shape[1:]
                
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
                
                max_size = int(np.prod(max_shape))
                print(f"Pre-allocating {name}: max_shape={max_shape}, size={max_size}")
                
                # Allocate for maximum size
                host_mem = cuda.pagelocked_empty(max_size, np_dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.buffers[name] = {
                    'host': host_mem,
                    'device': device_mem,
                    'max_shape': max_shape,
                    'base_shape': base_shape,
                    'dtype': np_dtype,
                    'max_size': max_size
                }
                
            except Exception as e:
                print(f"Error allocating buffer for {name}: {e}")
                continue
    
    def infer(self, input_data):
        """Run inference on input data with variable batch size"""
        if len(input_data.shape) != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {input_data.shape}")
        
        batch_size = input_data.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}")
        
        input_name = self.input_names[0]
        
        # Set dynamic input shape based on current batch size
        current_input_shape = [batch_size] + list(input_data.shape[1:])
        self.context.set_input_shape(input_name, current_input_shape)
        
        # Calculate current tensor shapes for all outputs
        current_shapes = {}
        current_sizes = {}
        
        for name in self.input_names + self.output_names:
            # Skip if buffer allocation failed
            if name not in self.buffers:
                raise RuntimeError(f"Buffer for {name} was not allocated successfully")
                
            if name == input_name:
                current_shapes[name] = current_input_shape
            else:
                current_shapes[name] = self.context.get_tensor_shape(name)
            
            current_sizes[name] = int(np.prod(current_shapes[name]))
            
            # Verify we don't exceed pre-allocated buffer
            if current_sizes[name] > self.buffers[name]['max_size']:
                raise RuntimeError(f"Current size {current_sizes[name]} exceeds pre-allocated size {self.buffers[name]['max_size']} for {name}")
        
        # Copy input data to pre-allocated buffer
        input_buffer = self.buffers[input_name]
        flat_input = input_data.ravel()
        input_buffer['host'][:len(flat_input)] = flat_input
        
        cuda.memcpy_htod_async(
            input_buffer['device'], 
            input_buffer['host'][:current_sizes[input_name]], 
            self.stream
        )
        
        # Set tensor addresses for current execution
        for name in self.input_names + self.output_names:
            self.context.set_tensor_address(name, int(self.buffers[name]['device']))
        
        # Run inference
        success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        # Copy outputs back
        for name in self.output_names:
            output_size = current_sizes[name]
            cuda.memcpy_dtoh_async(
                self.buffers[name]['host'][:output_size], 
                self.buffers[name]['device'], 
                self.stream
            )
        
        self.stream.synchronize()
        
        # Return outputs in correct shapes
        results = []
        for name in self.output_names:
            output_shape = current_shapes[name]
            output_size = current_sizes[name]
            output_data = self.buffers[name]['host'][:output_size].reshape(output_shape)
            results.append(output_data.copy())  # Copy to avoid buffer reuse issues
        
        return results
    
    def __del__(self):
        """Clean up GPU memory"""
        try:
            for buffer_info in self.buffers.values():
                if 'device' in buffer_info:
                    buffer_info['device'].free()
        except:
            pass

def batch_inference(inference_engine, images, max_batch_size=None):
    """Process multiple images in batches"""
    if max_batch_size is None:
        max_batch_size = inference_engine.max_batch_size
    
    all_results = []
    
    for i in range(0, len(images), max_batch_size):
        batch_images = images[i:i + max_batch_size]
        batch_input = np.stack(batch_images, axis=0)
        
        results = inference_engine.infer(batch_input)
        all_results.extend(results)
    
    return all_results

def load_images(image_folder):
    coco_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    coco_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    images = []
    for path in Path(image_folder).glob("*.png"):
        img = cv2.imread(str(path))          # BGR, uint8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (192, 256))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)         # HWC -> CHW
        images.append(img)

    batch = np.stack(images, axis=0)         # (N, 3, 256, 192)
    return (batch - coco_mean) / coco_std


if __name__ == "__main__":

    import pycuda.autoinit
    
    engine_path = "./parthoe_engine_fp16.trt"
    image_folder = "./testing_imgs"
    max_batch_size = 6

    inference_engine = TensorRTInference(engine_path, max_batch_size=max_batch_size)
    all_images = load_images(image_folder)  # (N, 3, 256, 192)

    for i in range(100):
        batch_size = np.random.randint(1, max_batch_size + 1)
        indices = np.random.choice(len(all_images), batch_size, replace=False)
        batch_input = all_images[indices]

        t_start = time.time()
        results = inference_engine.infer(batch_input)
        t_total = time.time() - t_start

        time.sleep(1. / 6.)

        print(f"Batch {i}: size={batch_size}, time={t_total:.4f}s")
        print(f"  Input shape: {batch_input.shape}")
        print(f"  Output shapes: {[r.shape for r in results]}")

        if batch_size == 1:
            for r in results:
                print(r)
            break