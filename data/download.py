import kagglehub

# Download latest version
path = kagglehub.competition_download('diabetic-retinopathy-detection')

print("Path to competition files:", path)
