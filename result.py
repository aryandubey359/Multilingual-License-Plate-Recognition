import re

# Load the network configuration file
with open("yolov4_train.cfg", "r") as f:
    cfg = f.read()

# Split the configuration file into a list of layers
layers = cfg.split("\n\n")

# Print the header row
print("{:<10} {:<10} {:<10}".format("Layer", "Type", "Filters"))

# Iterate over the layers and extract the filter size for each YOLO layer
for i, layer in enumerate(layers):
    if "yolo" in layer:
        filters = int(re.search("filters=(\d+)", layer).group(1))
        print("{:<10} {:<10} {:<10}".format(i, "YOLO", filters))
