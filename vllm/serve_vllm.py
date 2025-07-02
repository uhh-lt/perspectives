import os
import subprocess

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv("../.env")

VLLM_API_KEY = os.getenv("VLLM_API_KEY")
VLLM_PORT = os.getenv("VLLM_EXPOSED")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
HUGGING_FACE_CACHE_DIR = os.getenv("HUGGING_FACE_CACHE_DIR")
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Check that all env variables are set
if not VLLM_PORT:
    raise ValueError(
        "VLLM_EXPOSED not found in the .env file. Please add VLLM_EXPOSED=<port> to the file."
    )
if not CUDA_VISIBLE_DEVICES:
    raise ValueError(
        "CUDA_VISIBLE_DEVICES not found in the .env file. Please add CUDA_VISIBLE_DEVICES=<device_ids> to the file."
    )
if not HUGGING_FACE_CACHE_DIR:
    raise ValueError(
        "HUGGING_FACE_CACHE_DIR not found in the .env file. Please add HUGGING_FACE_CACHE_DIR=<path> to the file."
    )
if not HUGGING_FACE_HUB_TOKEN:
    raise ValueError(
        "HUGGING_FACE_HUB_TOKEN not found in the .env file. Please add HUGGING_FACE_HUB_TOKEN=<your_token> to the file."
    )
if not VLLM_API_KEY:
    raise ValueError(
        "API key not found in the .env file. Please add VLLM_API_KEY=<your_api_key> to the file."
    )


def main() -> None:
    """Main function for starting the VLLM server."""
    # Ensure the Hugging Face cache directory exists
    if not os.path.exists(str(HUGGING_FACE_CACHE_DIR)):
        os.makedirs(str(HUGGING_FACE_CACHE_DIR), exist_ok=True)

    # Ensure permissions for the cache directory
    os.chmod(str(HUGGING_FACE_CACHE_DIR), 0o777)

    print("Starting VLLM server with the following configuration:")
    print(f"Model Name: {VLLM_MODEL_NAME}")
    print(f"API Key: {VLLM_API_KEY}")
    print(f"Port: {VLLM_PORT}")
    print(f"CUDA Devices: {CUDA_VISIBLE_DEVICES}")
    print(f"Hugging Face Cache Directory: {HUGGING_FACE_CACHE_DIR}")

    # Construct the Docker command
    vllm_command = [
        "vllm serve",
        VLLM_MODEL_NAME,
        "--dtype",
        "auto",
        "--api-key",
        VLLM_API_KEY,
        "--port",
        VLLM_PORT,
        "--max-model-len",
        "4K",
    ]

    # Execute the serve command
    try:
        print(f"Running vllm command: {' '.join(vllm_command)}")
        subprocess.run(" ".join(vllm_command), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")


if __name__ == "__main__":
    main()
