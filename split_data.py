import numpy as np
import os


### lists where we will store our paths to images and labels ####

image_paths = []
label_paths = []

#IMPORTANT! Specify how much of training data we want to split! ###
# In case you want to work with smaller samples

# I had 218 for example
num_files = 218


#### define progress bar ####
def progress_bar(parsed, total):
    workdone = (parsed)/(total)
    print("\rProgress: [{0:50s}] {1:.1f}%".format(
        '#' * int(workdone * 50), workdone*100), end='', flush=True)


for i in range(num_files):
    #########################################################
    ### IMPORTANT!!! Change your data path accordingly!!! ###
    #########################################################
    training_data = np.load(
        'E://CrossPlatform//Data//GTA5Training//training_data-{}.npy'.format(i+1))

    for k, sample in enumerate(training_data):
        # set the path where you want your splti data to be stored
        img_file_dir = "data//images//"+str(i+1)
        if not os.path.exists(img_file_dir):
            os.makedirs(img_file_dir)
        img_file_name = 'img_'+str(i+1)+"_" + str(k+1) + ".npy"
        img_file_path = os.path.join(img_file_dir, img_file_name)

        np.save(img_file_path, sample[0])
        image_paths.append(img_file_path)

        # same here
        lbl_file_dir = "data//labels//"+str(i+1)
        if not os.path.exists(lbl_file_dir):
            os.makedirs(lbl_file_dir)
        lbl_file_name = 'lbl_'+str(i+1)+"_" + str(k+1) + ".npy"
        lbl_file_path = os.path.join(lbl_file_dir, lbl_file_name)

        np.save(lbl_file_path, sample[1])
        label_paths.append(lbl_file_path)

    progress_bar(i, len(range(num_files)))

### Save the data for later use ###
np.save('data//image_paths.npy', image_paths)
np.save('data//label_paths.npy', label_paths)
