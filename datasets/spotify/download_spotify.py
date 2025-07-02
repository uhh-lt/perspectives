import kagglehub

# Download latest version
path = kagglehub.dataset_download("devdope/900k-spotify", path="datasets/spotify/data")

print("Path to dataset files:", path)
