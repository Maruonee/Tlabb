from ultralytics import YOLO
import wandb
# from wandb.integratioen.ultralytics import add_wandb_callback
# Add the Weights & Biases callback to the model.
# This will work for training, evaluation and prediction
image_size = 512
batch_size = 32
custom_epochs = 300
# custom_model = "/home/tlab4090/datasets/opt/yolov8.yaml" #base
custom_model = "/home/tlab4090/datasets/opt/yolov10l.yaml"  #CBAM
#custom_model = "/home/tlab4090/datasets/opt/yolov8cbamswin.yaml" #SWIN
#custom_model = "/home/tlab4090/datasets/opt/yolov8swin.yaml" #CBAM_SWIN
# pretrain_model = 'yolov8n.pt'
data_adr = "/home/tlab4090/datasets/opt/coco.yaml"
test_list = "/home/tlab4090/datasets/opt/test.txt"
# Load a model
model = YOLO(custom_model)  # build a new model from scratch
# model = YOLO(pretrain_model)  # load a pretrained model (recommended for training)
# add_wandb_callback(model, enable_model_checkpointing=True)

# Use the model
model.train(
    data = data_adr,
    epochs = custom_epochs,
    batch = batch_size,
    imgsz = image_size,
    device = 0,
    workers = 32,
    pretrained = False,
    verbose = True
    ) # train the model
# # Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category
    
# # Run batched inference on a list of images
results = model.predict(
    test_list,
    save = True,
    imgsz = image_size,
    conf = 0.2,
    save_crop = True,
    save_txt = True,
    device = 0,
    save_conf = True,
    iou = 0.5,
    )