import os
import shutil
import random

#Create directories
data_dir = 'data/dogs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    #new directories
    class_dirs = ['dogs/','cats/']
    for  class_dir in class_dirs:
        new_dir = data_dir + subdir + class_dir
        os.makedirs(new_dir, exist_ok = True)

random.seed(1)
# Splitting Dogs and Cats Data into Separate Folders for train,test
# 25 % for testing
val_split = 0.25

src_dir = 'data/train/'
for file in os.listdir(src_dir):
    src = src_dir + file
    dst_dir = data_dir + 'train/'
    if random.random() < val_split:
        dst_dir = data_dir + 'test/'
    if file.startswith('cat'):
        dst = dst_dir+'cats/'+file
        shutil.copyfile(src,dst)
        print('copied cats')
    elif file.startswith('dog'):
        dst = dst_dir+'dogs/'+file
        shutil.copyfile(src, dst)
        print('copied dogs')

print('done')










