import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "mexwell/amazon-reviews-multi", path="datasets/amazon/data"
)

print("Path to dataset files:", path)
