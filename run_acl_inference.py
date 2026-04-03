import acl
import numpy as np
import os
import time

# ACL config
device_id = 0
model_path = "/home/HwHiAiUser/ICT/qwen_fp16.om"

def check_ret(ret, message):
    if ret != 0:
        raise Exception(f"{message} failed ret={ret}")

class AclModel:
    def __init__(self, device_id, model_path):
        self.device_id = device_id
        self.model_path = model_path
        self.model_id = None
        self.context = None
        self.stream = None
        self.input_dataset = None
        self.output_dataset = None
        self.model_desc = None

    def init(self):
        ret = acl.init()
        check_ret(ret, "acl.init")
        ret = acl.rt.set_device(self.device_id)
        check_ret(ret, "acl.rt.set_device")
        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret(ret, "acl.rt.create_context")
        self.stream, ret = acl.rt.create_stream()
        check_ret(ret, "acl.rt.create_stream")
        
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret(ret, "acl.mdl.load_from_file")
        
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret(ret, "acl.mdl.get_desc")

    def execute(self, inputs):
        # Prepare dataset
        self.input_dataset = acl.mdl.create_dataset()
        for input_data in inputs:
            # Copy data to device
            input_ptr, ret = acl.rt.malloc(input_data.nbytes, 0) # ACL_MEM_MALLOC_HUGE_FIRST
            check_ret(ret, "acl.rt.malloc")
            
            # Use memcpy to copy local data to device
            # Note: For python bytes/numpy, we need careful pointer handling or use numpy support if available in pyacl
            # Assuming input_data is numpy array
            
            # Ideally we use acl.util.numpy_to_ptr or similar if available, or direct mem copy
            # Here is a simplified way using numpy buffer interface
            if not input_data.flags['C_CONTIGUOUS']:
                input_data = np.ascontiguousarray(input_data)
                
            ptr = acl.util.numpy_to_ptr(input_data)
            ret = acl.rt.memcpy(input_ptr, input_data.nbytes, ptr, input_data.nbytes, 1) # ACL_MEMCPY_HOST_TO_DEVICE
            check_ret(ret, "acl.rt.memcpy")
            
            data_buffer = acl.create_data_buffer(input_ptr, input_data.nbytes)
            acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer)

        # Output dataset setup
        self.output_dataset = acl.mdl.create_dataset()
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        for i in range(output_size):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            out_ptr, ret = acl.rt.malloc(size, 0)
            check_ret(ret, "acl.rt.malloc output")
            data_buffer = acl.create_data_buffer(out_ptr, size)
            acl.mdl.add_dataset_buffer(self.output_dataset, data_buffer)

        # Execute
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        check_ret(ret, "acl.mdl.execute")

        # Get output
        results = []
        for i in range(output_size):
            data_buffer = acl.mdl.get_dataset_buffer(self.output_dataset, i)
            data_ptr = acl.get_data_buffer_addr(data_buffer)
            size = acl.get_data_buffer_size(data_buffer)
            
            # Copy back to host
            host_data = np.zeros(size // 4, dtype=np.float32) # Assuming fp32/int32 output, adjust if needed
            host_ptr = acl.util.numpy_to_ptr(host_data)
            
            ret = acl.rt.memcpy(host_ptr, size, data_ptr, size, 2) # ACL_MEMCPY_DEVICE_TO_HOST
            check_ret(ret, "acl.rt.memcpy out")
            results.append(host_data)
            
        return results

    def release(self):
        # Clean up resources (simplified)
        acl.rt.destroy_stream(self.stream)
        acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

def main():
    try:
        model = AclModel(device_id, model_path)
        print("Initializing ACL...")
        model.init()
        print("Model loaded.")
        
        # Create dummy inputs matching the shape used in ATC
        # input_ids:1,128;attention_mask:1,128;position_ids:1,128
        input_ids = np.random.randint(0, 1000, (1, 128)).astype(np.int64)
        attention_mask = np.ones((1, 128)).astype(np.int64)
        position_ids = np.arange(128).reshape(1, 128).astype(np.int64)
        
        print("Executing inference...")
        start = time.time()
        outputs = model.execute([input_ids, attention_mask, position_ids])
        print(f"Inference time: {time.time() - start:.4f}s")
        
        print("Output shape:", outputs[0].shape)
        # Note: The output is flattened float32 array, need to reshape based on model output logic if needed
        
        model.release()
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
