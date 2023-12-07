import os
import shutil

source_dir = './fid_data/generated_images_KD'  # Path to your generated_images directory
dest_dir = './fid_data/combined_generated_images_KD'  # Path to the destination directory

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for class_dir in os.listdir(source_dir):
    class_dir_path = os.path.join(source_dir, class_dir)
    
    if os.path.isdir(class_dir_path):  # Make sure it's a directory
        for filename in os.listdir(class_dir_path):
            file_path = os.path.join(class_dir_path, filename)
            
            # Ensure we're copying files (not directories)
            if os.path.isfile(file_path):
                # Prepend the class ID to the filename
                new_filename = f"{class_dir}_{filename}"
                shutil.copy(file_path, os.path.join(dest_dir, new_filename))

print("All images have been copied and renamed in:", dest_dir)

# import os
# import shutil
# import random

# source_dir = './data/cifar10-64/test'  # Path to the test directory of CIFAR-10
# dest_dir = './sampled_cifar10_images'  # Path to the destination directory
# images_per_class = 1024  # Number of images to sample per class

# if not os.path.exists(dest_dir):
#     os.makedirs(dest_dir)

# for class_dir in os.listdir(source_dir):
#     class_dir_path = os.path.join(source_dir, class_dir)
    
#     if os.path.isdir(class_dir_path):
#         all_filenames = os.listdir(class_dir_path)
#         sampled_filenames = random.sample(all_filenames, min(images_per_class, len(all_filenames)))

#         for filename in sampled_filenames:
#             file_path = os.path.join(class_dir_path, filename)
            
#             # Ensure we're moving files (not directories)
#             if os.path.isfile(file_path):
#                 # Prepend the class ID to the filename
#                 new_filename = f"{class_dir}_{filename}"
#                 shutil.move(file_path, os.path.join(dest_dir, new_filename))

# print("Sampled images have been moved to:", dest_dir)

# import os
# import shutil
# import random

# source_dir = './data/cifar10-64/train'  # Path to the test directory of CIFAR-10
# dest_dir = './sampled_cifar10_images'  # Path to the destination directory
# images_per_class = 1024  # Number of images to sample per class

# if not os.path.exists(dest_dir):
#     os.makedirs(dest_dir)

# for class_dir in os.listdir(source_dir):
#     class_dir_path = os.path.join(source_dir, class_dir)
#     class_dest_dir = os.path.join(dest_dir, class_dir)
    
#     if not os.path.exists(class_dest_dir):
#         os.makedirs(class_dest_dir)

#     if os.path.isdir(class_dir_path):
#         all_filenames = os.listdir(class_dir_path)
#         sampled_filenames = random.sample(all_filenames, min(images_per_class, len(all_filenames)))

#         for filename in sampled_filenames:
#             file_path = os.path.join(class_dir_path, filename)
            
#             # Ensure we're moving files (not directories)
#             if os.path.isfile(file_path):
#                 shutil.move(file_path, os.path.join(class_dest_dir, filename))

# print("Sampled images have been moved to their respective class folders in:", dest_dir)

