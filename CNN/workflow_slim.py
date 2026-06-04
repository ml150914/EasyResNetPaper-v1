#!/home/lorenzo-mobilia/.conda/envs/tensorflow_env/bin/python 

import tensorflow as tf
import os
import time
from tensorflow import keras
from tensorflow.keras import layers  # Import layers module 
from tensorflow.keras.models import Sequential
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import os
import sys
import argparse

# Import the models you want to use
from models import EasyResNet, SmallNet, ResNet16, EasyResNet1, EasyResNet2, EasyResNet3, EasyResNet4, ResNet16_smaller, ResNet16_var1, DeepNet, ResNetScratch
from read_data import create_tf, read_png_metadata

MODEL_MAP = {
    "EasyResNet": EasyResNet,
    "SmallNet" : SmallNet,
    "ResNet16" : ResNet16,
    "EasyResNet1" : EasyResNet1,
    "EasyResNet2" : EasyResNet2,
    "EasyResNet3" : EasyResNet3,
    "EasyResNet4" : EasyResNet4,
    "ResNet16_smaller" : ResNet16_smaller,
    "ResNet16_var1" : ResNet16_var1,
    "DeepNet": DeepNet,
    "ResNetScratch": ResNetScratch
 }

parser = argparse.ArgumentParser(usage='',
    description="Run the CNN training and test")
parser.add_argument("--data-input-dir", type=str, required=True,
                    help="Insert folder path to data")
parser.add_argument("--data-public-out", type=str, required=True,
                    help="Insert folder path to public.html sanity check")
parser.add_argument("--data-out", type=str, required=True,
                    help="Insert folder path to output dir")
parser.add_argument("--model", type = str, required=True,
                    help = "Insert the model")
args = parser.parse_args()

selected_model = args.model

# Enable MirroredStrategy for multi-CPU
strategy = tf.distribute.MirroredStrategy()

# Define the batch dimension and the image dimension
batch_size = 64
img_height = 256
img_width  = 512

# Check directories
for path, name in [(args.data_input_dir, "data_input_dir"),
                   (args.data_public_out, "data_public_out"),
                   (args.data_out, "data_out")]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: Directory '{path}' ({name}) does not exist.")

data_dir = args.data_input_dir
out_public_dir = args.data_public_out
out_dir = args.data_out

# create tensorflow dataframe 
train_ds, val_ds, test_ds, file_paths_test = create_tf(data_dir,img_width, img_height, batch_size)

# Check everything is ok
for images, _ in val_ds.take(1):
    for i in range(9):
        #augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().squeeze(), cmap = 'gray')
        plt.savefig(out_public_dir + '/sanity_check_images.png')
    plt.close()

# Struggle against overfitting with data augmentation                                            
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.2),
    #layers.RandomZoom(0.1),
  ]
)
    
# Compile the model
Model = MODEL_MAP[selected_model](shape =(img_height,img_width,1), classes = 2, data_augmentation = None)
initial_lr = 1e-3
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=100,  # Assuming ~100 epochs, adjust if needed
    alpha=1e-5 / initial_lr  # Final LR is 1e-5
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),                            
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),                                                                            
              metrics=['accuracy']) 
Model.summary()

# Early Stopping
callback = keras.callbacks.EarlyStopping(monitor='loss',
                                         patience=3,
                                         min_delta = 0.001)
# Train the model
epochs = 20
history = Model.fit(    
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    #verbose = 2                                                                                                            
)
# Save the model 
Model.save(f'{MODEL_MAP[selected_model]}.h5')

# Plot accuracy                               
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig(out_public_dir + '/Train_val_Accuracy_sep_dataset.png')
#plt.show()

# Plot loss                                                        
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(out_public_dir + '/Train_val_loss_sep_dataset.png')
#plt.show()

#-------------------------- Analyse test dataset 
counter = 0
# Iterate through the test dataset
# --------> Test
total_inference_time = 0
num_images_tested = 0
counter = 0 # Name of the test datum analyzed
for image in test_ds.take(1):
    print("Image shape Test: ", image.numpy().shape)

for images in test_ds:

    # Start timing the batch prediction
    start_batch_time = time.time()
    
    # Predict probabilities for the batch
    probs = Model.predict(images)

    # End timing for the batch
    end_batch_time = time.time()

    # Calculate the time for this batch
    batch_inference_time = (end_batch_time - start_batch_time) * 1000  # Convert to milliseconds
    total_inference_time += batch_inference_time
    num_images_tested += images.shape[0]
    
    for i in range(images.shape[0]):  # Iterate over batch size
        img = images[i].numpy()  # Convert to numpy array
        img = img / 255.0
        predicted_probs = probs[i]
        title = (#f"True Class: {true_label}\n"
                 f"Class Inj Prob: {predicted_probs[0]:.2f} | "
                 f"Class Noise Prob: {predicted_probs[1]:.2f}")
        if(predicted_probs[0] >= 0.5):
            predicted_label = 'inj' # inj
        else: predicted_label = 'noise' # noise
    
        # Plot the image
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap = 'gray')
        plt.axis('off')  # Turn off axis
        plt.title(title, fontsize=12)

        # read metadata associated to the image
        metadata_old = read_png_metadata(file_paths_test[counter])
                
        # Add the CNN statistics to the metadata
        metadata = {'Noise_Prob' : predicted_probs[1], 'Inj_Prob' : predicted_probs[0], 'Predicted_label': predicted_label}

        # update
        combined_metadata = metadata_old.copy()
        combined_metadata.update(metadata)

        #print(combined_metadata)
        # Save the image to the output directory
        save_path = out_dir  +  '/TTMap_' + str(counter) + '.png'
        plt.savefig(save_path)
        #print(save_path)
        #print('\n')
                        
        # Add metadata
        im = Image.open(save_path)
        meta = PngInfo()
        for x in combined_metadata:
            meta.add_text(x, str(combined_metadata[x]))
        im.save(save_path, 'png', pnginfo = meta)
        counter = counter+1
    
# Calculate and display average inference time per image                                                   
average_inference_time = total_inference_time / num_images_tested
print(f"Average inference time per image: {average_inference_time:.2f} ms")
