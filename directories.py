import os
import random
import shutil

classes = os.listdir("PlantVillage-Dataset/raw/color")
# print(classes)

"""
Uncomment the block of code to create a directory tree
"""
testing_path = "Dataset/testing"
training_path = "Dataset/training"
validate_path = "Dataset/validating"
for cls in classes:
    path = os.path.join(testing_path, cls)
    os.mkdir(path)
    path = os.path.join(training_path, cls)
    os.mkdir(path)
    path = os.path.join(validate_path, cls)
    os.mkdir(path)

types = ["color", "grayscale", "segmented"]

new_data_paths = ["Dataset/training",
                  "Dataset/testing",
                  "Dataset/validating"]

old_data_paths = ["PlantVillage-Dataset/raw/color",
                  "PlantVillage-Dataset/raw/grayscale",
                  "PlantVillage-Dataset/raw/segmented"]


def choose_destination():
    random_num = random.randint(1, 10)
    if random_num <= 8: return new_data_paths[0]
    if random_num == 9: return new_data_paths[1]
    if random_num == 10: return new_data_paths[2]


def manage_class(class_name):
    source_paths = []
    for img_type in range(3):
        source_paths.append(os.path.join(old_data_paths[img_type], class_name))

    files = {"color": os.listdir(source_paths[0]),
             "grayscale": os.listdir(source_paths[1]),
             "segmented": os.listdir(source_paths[2])}
    files_quantity = len(os.listdir(os.path.join(old_data_paths[0], class_name)))
    for i in range(files_quantity):
        destination_path = os.path.join(choose_destination(), class_name)
        j = 0
        for img_type in files.keys():
            source = os.path.join(source_paths[j], files.get(img_type)[i])
            destination = os.path.join(destination_path, class_name + "_" + img_type + "_" + str(i + 1) + ".jpg")
            shutil.move(source, destination)
            j += 1


for cls in classes:
    manage_class(cls)

