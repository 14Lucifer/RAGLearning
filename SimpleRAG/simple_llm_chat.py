import time
import yaml
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from system_info import get_system_info, display_gpu_info

def main():

    # Display system information
    print("\nSystem Information:")
    print("-------------------")
    system_info = get_system_info()
    for key, value in system_info.items(): 
        print(f"{key}: {value}")
    display_gpu_info()  # Print GPU info.
    print("\n")         # Line breaker.


    # Load configuration from YAML file
    with open("config.yaml", "r") as file:
        config           = yaml.safe_load(file)
        model_path       = config['model']['path']
        model_name       = config['model']['name']
        txt_max_length   = config['text_generation']['max_length']
        txt_top_k        = config['text_generation']['top_k']
        txt_top_p        = config['text_generation']['top_p']
        txt_tmp          = config['text_generation']['temperature']


    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    # Configure BitsAndBytes quantization
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    # bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    # bnb_config = BitsAndBytesConfig(load_in_16bit=True)


    # Load the Model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # quantization_config=bnb_config,  # Apply quantization.
        torch_dtype=torch.float16,
        device_map="auto"  
    )


    # Enable DataParallel for multiple GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)


    # Start chat interface
    print(f"\nChat with \033[1m{model_name}\033[0m. Type 'exit' to end the conversation.")

    while True:
        user_input = input("\n\033[31m\033[1m[Question : ]\n\033[0m")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        device = next(model.parameters()).device
        inputs = tokenizer(user_input, return_tensors="pt").to(device)

        start_time = time.time()  # Start timer

        # Perform text generation with configured parameters
        with torch.no_grad():
            outputs = model.module.generate(
                **inputs, 
                max_length  =txt_max_length,  # Controls response length
                top_k       =txt_top_k,       # Consider top 50 tokens (increase diversity)
                top_p       =txt_top_p,       # Use nucleus sampling (filter by probability mass)
                temperature =txt_tmp,         # Add randomness (higher = more creative)
                do_sample   =True             # Enables sampling instead of greedy decoding
            )

        
        elapsed_time = time.time() - start_time  # End time

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n")
        print(f"\033[32m\033[1m[Model Response : ]\n\033[0m{response}")
        print(f"\033[1m* Time taken : \033[0m{elapsed_time:.4f} seconds")
        
        # Empty GPU cache
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
