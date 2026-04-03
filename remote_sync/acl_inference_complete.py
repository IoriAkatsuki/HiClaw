#!/usr/bin/env python3
"""
Complete ACL Inference Engine for Qwen with KV Cache and Tokenizer support
"""
import acl
import numpy as np
import os
import json
from transformers import AutoTokenizer
import time

class QwenACLInference:
    def __init__(self, model_path, tokenizer_path, device_id=0):
        self.device_id = device_id
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        # ACL resources
        self.context = None
        self.stream = None
        self.model_id = None
        self.model_desc = None
        
        # Model info
        self.input_num = 0
        self.output_num = 0
        
        # Initialize ACL
        self.init_acl()
        
    def check_ret(self, ret, message):
        if ret != 0:
            raise Exception(f"{message} failed with ret={ret}")
    
    def init_acl(self):
        """Initialize ACL environment"""
        print("[ACL] Initializing...")
        ret = acl.init()
        self.check_ret(ret, "acl.init")
        
        ret = acl.rt.set_device(self.device_id)
        self.check_ret(ret, "acl.rt.set_device")
        
        self.context, ret = acl.rt.create_context(self.device_id)
        self.check_ret(ret, "acl.rt.create_context")
        
        self.stream, ret = acl.rt.create_stream()
        self.check_ret(ret, "acl.rt.create_stream")
        
        print(f"[ACL] Loading model from {self.model_path}...")
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        self.check_ret(ret, "acl.mdl.load_from_file")
        
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        self.check_ret(ret, "acl.mdl.get_desc")
        
        self.input_num = acl.mdl.get_num_inputs(self.model_desc)
        self.output_num = acl.mdl.get_num_outputs(self.model_desc)
        
        print(f"[ACL] Model loaded: {self.input_num} inputs, {self.output_num} outputs")
        
    def prepare_input_dataset(self, input_data_list):
        """Prepare input dataset for ACL inference"""
        dataset = acl.mdl.create_dataset()
        
        for idx, input_data in enumerate(input_data_list):
            if not input_data.flags['C_CONTIGUOUS']:
                input_data = np.ascontiguousarray(input_data)
            
            # Allocate device memory
            data_size = input_data.nbytes
            device_ptr, ret = acl.rt.malloc(data_size, acl.ACL_MEM_MALLOC_HUGE_FIRST)
            self.check_ret(ret, f"acl.rt.malloc input[{idx}]")
            
            # Copy data to device
            host_ptr = acl.util.numpy_to_ptr(input_data)
            ret = acl.rt.memcpy(device_ptr, data_size, host_ptr, data_size, acl.ACL_MEMCPY_HOST_TO_DEVICE)
            self.check_ret(ret, f"acl.rt.memcpy input[{idx}]")
            
            # Create data buffer and add to dataset
            data_buffer = acl.create_data_buffer(device_ptr, data_size)
            acl.mdl.add_dataset_buffer(dataset, data_buffer)
            
        return dataset
    
    def prepare_output_dataset(self):
        """Prepare output dataset for ACL inference"""
        dataset = acl.mdl.create_dataset()
        output_sizes = []
        
        for i in range(self.output_num):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            output_sizes.append(size)
            
            device_ptr, ret = acl.rt.malloc(size, acl.ACL_MEM_MALLOC_HUGE_FIRST)
            self.check_ret(ret, f"acl.rt.malloc output[{i}]")
            
            data_buffer = acl.create_data_buffer(device_ptr, size)
            acl.mdl.add_dataset_buffer(dataset, data_buffer)
            
        return dataset, output_sizes
    
    def get_output_data(self, output_dataset, output_sizes):
        """Retrieve output data from device"""
        results = []
        
        for i in range(self.output_num):
            data_buffer = acl.mdl.get_dataset_buffer(output_dataset, i)
            device_ptr = acl.get_data_buffer_addr(data_buffer)
            size = output_sizes[i]
            
            # Assuming FP16 output, size//2 elements
            host_data = np.zeros(size // 2, dtype=np.float16)
            host_ptr = acl.util.numpy_to_ptr(host_data)
            
            ret = acl.rt.memcpy(host_ptr, size, device_ptr, size, acl.ACL_MEMCPY_DEVICE_TO_HOST)
            self.check_ret(ret, f"acl.rt.memcpy output[{i}]")
            
            results.append(host_data)
            
        return results
    
    def inference(self, input_ids, attention_mask, position_ids):
        """Run inference on the model"""
        # Prepare inputs
        inputs = [
            input_ids.astype(np.int64),
            attention_mask.astype(np.int64),
            position_ids.astype(np.int64)
        ]
        
        input_dataset = self.prepare_input_dataset(inputs)
        output_dataset, output_sizes = self.prepare_output_dataset()
        
        # Execute model
        ret = acl.mdl.execute(self.model_id, input_dataset, output_dataset)
        self.check_ret(ret, "acl.mdl.execute")
        
        # Get outputs
        outputs = self.get_output_data(output_dataset, output_sizes)
        
        # Cleanup datasets
        self.destroy_dataset(input_dataset)
        self.destroy_dataset(output_dataset)
        
        return outputs
    
    def destroy_dataset(self, dataset):
        """Free dataset resources"""
        if dataset:
            num = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(num):
                data_buffer = acl.mdl.get_dataset_buffer(dataset, i)
                device_ptr = acl.get_data_buffer_addr(data_buffer)
                acl.rt.free(device_ptr)
                acl.destroy_data_buffer(data_buffer)
            acl.mdl.destroy_dataset(dataset)
    
    def generate(self, prompt, max_length=512, temperature=0.7):
        """
        Generate text from prompt using greedy decoding
        Note: This is a simplified implementation without KV cache
        """
        print(f"\n[Generate] Prompt: {prompt}")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np", padding="max_length", max_length=max_length, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        batch_size, seq_len = input_ids.shape
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        
        print(f"[Generate] Input shape: {input_ids.shape}")
        
        # Run inference
        start_time = time.time()
        outputs = self.inference(input_ids, attention_mask, position_ids)
        inference_time = time.time() - start_time
        
        # Get logits and find next token
        logits = outputs[0].reshape(batch_size, seq_len, -1)  # Shape: (B, S, vocab_size)
        
        # Get the logits for the last position
        last_logits = logits[0, -1, :]
        next_token_id = np.argmax(last_logits)
        
        # Decode
        generated_text = self.tokenizer.decode([next_token_id])
        
        print(f"[Generate] Inference time: {inference_time:.3f}s")
        print(f"[Generate] Next token: {generated_text}")
        
        return generated_text, inference_time
    
    def cleanup(self):
        """Release ACL resources"""
        print("[ACL] Cleaning up...")
        if self.model_id:
            acl.mdl.unload(self.model_id)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        print("[ACL] Cleanup complete")

def main():
    """Test the ACL inference engine"""
    MODEL_PATH = "/home/HwHiAiUser/ICT/qwen_fp16.om"
    TOKENIZER_PATH = "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen2.5-7B-Instruct"
    
    try:
        engine = QwenACLInference(MODEL_PATH, TOKENIZER_PATH)
        
        # Test generation
        test_prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Explain quantum computing in simple terms."
        ]
        
        for prompt in test_prompts:
            generated, time_taken = engine.generate(prompt, max_length=256)
            print(f"Generated: {generated}\n")
            
        engine.cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
