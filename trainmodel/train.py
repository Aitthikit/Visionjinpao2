import roboflow

roboflow.login()

rf = roboflow.Roboflow()

# create a project
rf.create_project(
    project_name="project name",
    project_type="project-type",
    license="project-license" # "private" for private projects
)

workspace = rf.workspace("WORKSPACE_URL")
project = workspace.project("PROJECT_URL")
version = project.version("VERSION_NUMBER")

# upload a dataset
project.upload_dataset(
    dataset_path="./dataset/",
    num_workers=10,
    dataset_format="yolov8", # supports yolov8, yolov5, and Pascal VOC
    project_license="MIT",
    project_type="object-detection"
)

# upload model weights
version.deploy(model_type="yolov8", model_path=f”{HOME}/runs/detect/train/”)

# run inference
model = version.model

img_url = "https://media.roboflow.com/quickstart/aerial_drone.jpeg"

predictions = model.predict(img_url, hosted=True).json()

print(predictions)