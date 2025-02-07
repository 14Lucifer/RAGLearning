# system_info.py
import platform
import psutil
import torch

def get_system_info():
    # OS and Kernel Version
    os_info = platform.uname()
    os_name = os_info.system  # OS name (e.g., 'Linux' or 'Windows')
    os_version = os_info.version  # Version of the OS
    kernel_version = os_info.release  # Kernel version

    # CPU Information
    cpu_count = psutil.cpu_count(logical=False)  # Physical CPU cores
    cpu_threads = psutil.cpu_count(logical=True)  # Total CPU threads

    # Memory Information (in GB)
    memory_info = psutil.virtual_memory()
    total_memory_gb = memory_info.total / (1024 ** 3)

    # GPU Information using torch (PyTorch)
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # in GB
        gpu_info.append({'GPU': gpu_name, 'Memory (GB)': gpu_memory})

    # Return collected information as a dictionary
    return {
        '- OS Name': os_name,
        '- OS Version': os_version,
        '- Kernel Version': kernel_version,
        '- CPU Count': f"{cpu_count} Physical CPU cores, {cpu_threads} Logical CPU threads",
        '- Total Memory (GB)': f"{total_memory_gb:.2f} GB"
    }

def display_gpu_info():
    gpu_count = torch.cuda.device_count()
    print(f"- Mounted GPU Card Count: {gpu_count}")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # in GB
        print(f"- GPU {i}: {gpu_name} Memory {gpu_memory:.2f}GB")
