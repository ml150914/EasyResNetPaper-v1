import tensorflow as tf
import os

from PIL import Image
from PIL.PngImagePlugin import PngInfo

def create_tf(input_path, img_width, img_height, batch_size):
    #----- coordinates to read the data --------#
    path_train_inj = input_path + '/train/inj/'
    path_train_noise = input_path + '/train/noise/'

    path_val_inj = input_path + '/val/inj/'
    path_val_noise = input_path + '/val/noise/'

    path_test = input_path + '/test'

    #------------------------------------------#

    # Initializer the lists for metadata
    metadata_train_inj_list = []
    metadata_train_noise_list = []

    metadata_val_inj_list = []
    metadata_val_noise_list = []

    metadata_test_list = []
    # Function to read the metadata
    def read_metadata(metadata_list, path_data):
        for root, _, files in os.walk(path_data):
            metadata_dict = {}
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')): # Adjust extensions as needed 
                    file_path = os.path.join(root, file)
                    img = Image.open(file_path)
                    metadata_dict[file_path] = img.info # Save metadata using file path as key
                    metadata_list.append(metadata_dict[file_path])
            return metadata_list
    #------------- read the metadata
    md_train_inj = read_metadata(metadata_train_inj_list, path_train_inj)
    md_train_noise = read_metadata(metadata_train_noise_list, path_train_noise)

    md_val_inj = read_metadata(metadata_val_inj_list, path_val_inj)
    md_val_noise = read_metadata(metadata_val_noise_list, path_val_noise)

    md_test = read_metadata(metadata_test_list, path_test)

    # Create tfdataset
    list_train_ds = tf.data.Dataset.list_files(input_path+'/train/*/*', shuffle=False)
    list_val_ds = tf.data.Dataset.list_files(input_path+'/val/*/*', shuffle=False)
    list_test_ds = tf.data.Dataset.list_files(input_path+'/test/*', shuffle=False)

    # Shuffle the tfdataset
    list_train_ds = list_train_ds.shuffle(len(list_train_ds), reshuffle_each_iteration=False)
    list_val_ds = list_val_ds.shuffle(len(list_val_ds), reshuffle_each_iteration=False)
    list_test_ds = list_test_ds.shuffle(len(list_test_ds), reshuffle_each_iteration=False)

    # Initialize the classes
    class_names = ['inj', 'noise']

    # link the path to the image (each image has a unique random name)
    file_paths_training = list(list_train_ds.as_numpy_iterator())
    file_paths_validation = list(list_val_ds.as_numpy_iterator())
    file_paths_test = list(list_test_ds.as_numpy_iterator())

    # Check the length of the dataset
    print('Training ', len(list_train_ds))
    print('Validation ', len(list_val_ds))
    print('Test ', len(list_test_ds))

    # Convert datasets to Python sets of file paths
    train_paths = set([path.numpy().decode('utf-8') for path in list_train_ds])
    val_paths = set([path.numpy().decode('utf-8') for path in list_val_ds])
    test_paths = set([path.numpy().decode('utf-8') for path in list_test_ds])

    # Check for overlaps bia intersection
    train_val_overlap = train_paths & val_paths
    train_test_overlap = train_paths & test_paths
    val_test_overlap = val_paths & test_paths

    # Print results
    if not train_val_overlap and not train_test_overlap and not val_test_overlap:
        print("No images are shared among the datasets.")
    else:
        print("Overlap detected:")
        if train_val_overlap:
            print(f"Train-Val Overlap: {train_val_overlap}")
        if train_test_overlap:
            print(f"Train-Test Overlap: {train_test_overlap}")
        if val_test_overlap:
            print(f"Val-Test Overlap: {val_test_overlap}")

    # Optimizing the reading data phase (avoiding bottleneck)
    AUTOTUNE = tf.data.AUTOTUNE

    #-------- Associate the label to the image
    # Extract the label function
    def get_label(file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names # inj 0 noise 1
        # Integer encode the label
        return tf.argmax(one_hot)

    # Extract the image
    def decode_img(img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=1)
        # Standard resize
        return tf.image.resize(img, [img_height, img_width])

    # Associate the label and the image
    def process_path(file_path):
        label = get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label
    #------------------------------------

    # Now create the tfdataset to train and validate the Net
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = list_train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = list_val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # Configure the data-reading performance
    def configure_for_performance(ds):
        ds = ds.batch(batch_size)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    # Now create the tf dataset for the test
    
    def decode_img_test(img):
        img = tf.io.decode_jpeg(img, channels=1)
        return tf.image.resize(img, [img_height, img_width])

    def process_path_test(file_path):
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img
    # Create the Test dataset
    test_ds = list_test_ds.map(process_path_test, num_parallel_calls=AUTOTUNE)
    test_ds = configure_for_performance(test_ds)
    
    return train_ds, val_ds, test_ds, file_paths_test

# Function to read metadata from the image                                                                                            
def read_png_metadata(image_path):
    """Extracts metadata added using PngInfo from a PNG file."""
    with Image.open(image_path) as img:
        metadata = {key: img.info[key] for key in img.info.keys()}
    return metadata

def create_tf_training(input_path, img_width, img_height, batch_size):
    #----- coordinates to read the data --------# 
    path_train_inj = input_path + '/train/inj/'
    path_train_noise = input_path + '/train/noise/'

    path_val_inj = input_path + '/val/inj/'
    path_val_noise = input_path + '/val/noise/'

    path_test = input_path + '/test'
    #------------------------------------------# 

    # Initializer the lists for metadata                           
    metadata_train_inj_list = []
    metadata_train_noise_list = []

    metadata_val_inj_list = []
    metadata_val_noise_list = []

    metadata_test_list = []
    # Function to read the metadata                                                                
    def read_metadata(metadata_list, path_data):
        for root, _, files in os.walk(path_data):
            metadata_dict = {}
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')): # Adjust extensions as needed
                    
                    file_path = os.path.join(root, file)
                    img = Image.open(file_path)
                    metadata_dict[file_path] = img.info # Save metadata using file path as key                                                
                    metadata_list.append(metadata_dict[file_path])
            return metadata_list
    #------------- read the metadata                                                                                                          
    md_train_inj = read_metadata(metadata_train_inj_list, path_train_inj)
    md_train_noise = read_metadata(metadata_train_noise_list, path_train_noise)

    md_val_inj = read_metadata(metadata_val_inj_list, path_val_inj)
    md_val_noise = read_metadata(metadata_val_noise_list, path_val_noise)

    md_test = read_metadata(metadata_test_list, path_test)

    # Create tfdataset                                                                                                                        
    list_train_ds = tf.data.Dataset.list_files(input_path+'/train/*/*', shuffle=False)
    list_val_ds = tf.data.Dataset.list_files(input_path+'/val/*/*', shuffle=False)
    list_test_ds = tf.data.Dataset.list_files(input_path+'/test/*', shuffle=False)

    # Shuffle the tfdataset                                                                                                                   
    list_train_ds = list_train_ds.shuffle(len(list_train_ds), reshuffle_each_iteration=False)
    list_val_ds = list_val_ds.shuffle(len(list_val_ds), reshuffle_each_iteration=False)
    list_test_ds = list_test_ds.shuffle(len(list_test_ds), reshuffle_each_iteration=False)

    # Initialize the classes                                                                                                                  
    class_names = ['inj', 'noise']

    # link the path to the image (each image has a unique random name)                                                                        
    file_paths_training = list(list_train_ds.as_numpy_iterator())
    file_paths_validation = list(list_val_ds.as_numpy_iterator())
    file_paths_test = list(list_test_ds.as_numpy_iterator())

    # Check the length of the dataset                                                                                                         
    print('Training ', len(list_train_ds))
    print('Validation ', len(list_val_ds))
    print('Test ', len(list_test_ds))

    # Convert datasets to Python sets of file paths                                                                                           
    train_paths = set([path.numpy().decode('utf-8') for path in list_train_ds])
    val_paths = set([path.numpy().decode('utf-8') for path in list_val_ds])
    test_paths = set([path.numpy().decode('utf-8') for path in list_test_ds])

    # Check for overlaps bia intersection                                                                                                     
    train_val_overlap = train_paths & val_paths
    train_test_overlap = train_paths & test_paths
    val_test_overlap = val_paths & test_paths

    # Print results                                                                                                                           
    if not train_val_overlap:
    #and not train_test_overlap and not val_test_overlap:
        print("No images are shared among the datasets.")
    else:
        print("Overlap detected:")
        if train_val_overlap:
            print(f"Train-Val Overlap: {train_val_overlap}")
        if train_test_overlap:
            print(f"Train-Test Overlap: {train_test_overlap}")
        if val_test_overlap:
            print(f"Val-Test Overlap: {val_test_overlap}")

    # Optimizing the reading data phase (avoiding bottleneck)                                                                                 
    AUTOTUNE = tf.data.AUTOTUNE

    #-------- Associate the label to the image                                                                                                
    # Extract the label function                                                                                                              
    def get_label(file_path):
        # Convert the path to a list of path components                                                                                       
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory                                                                                           
        one_hot = parts[-2] == class_names # inj 0 noise 1                                                                                    
        # Integer encode the label                                                                                                            
        return tf.argmax(one_hot)

    # Extract the image                                                                                                                       
    def decode_img(img):
        # Convert the compressed string to a 3D uint8 tensor                                                                                  
        img = tf.io.decode_jpeg(img, channels=1)
        # Standard resize                                                                                                                     
        return tf.image.resize(img, [img_height, img_width])

    # Associate the label and the image                                                                                                       
    def process_path(file_path):
        label = get_label(file_path)
        # Load the raw data from the file as a string                                                                                         
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label
    #------------------------------------                                                                                                     

    # Now create the tfdataset to train and validate the Net                                                                                  
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.                                                           
    train_ds = list_train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = list_val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # Configure the data-reading performance                                                                                                  
    def configure_for_performance(ds):
        ds = ds.batch(batch_size)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    # Now create the tf dataset for the test                                                                                                  

    def decode_img_test(img):
        img = tf.io.decode_jpeg(img, channels=1)
        return tf.image.resize(img, [img_height, img_width])

    def process_path_test(file_path):
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img
    # Create the Test dataset                                                                                                                 
    test_ds = list_test_ds.map(process_path_test, num_parallel_calls=AUTOTUNE)
    test_ds = configure_for_performance(test_ds)

    return train_ds, val_ds, test_ds, file_paths_test
