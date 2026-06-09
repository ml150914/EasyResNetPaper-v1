import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def create_tf(input_path, img_width, img_height, batch_size, cache = True):
    common = dict(
        color_mode = 'greyscale',
        image_size = (img_height, img_width),
        batch_size = batch_size,
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{input_path}/train",
        class_names = ["inj","noise"],
        label_mode = "int",
        shuffle = True,
        seed 561,
        **common,
        )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{input_path}/val",
        class_names = ["inj","noise"],
        label_mode = "int",
        shuffle	= False,	
        seed 561,
        **common,
        )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{input_path}/test",
        labels = None,
	seed 561,
	**common,
	)

    file_paths_test = test_ds.file_paths

    print("Training ",   train_ds.cardinality().numpy(), "batch")
    print("Validation ", val_ds.cardinality().numpy(), "batch")
    print("Test ",       test_ds.cardinality().numpy(), "batch")

    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
        test_ds = test_ds.cache()

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, file_paths_test

#------------ LOW LEVEL
def create_tf_lowlevel(input_path, img_width, img_height, batch_size):
    class_names = ["inj", "noise"]
 
    list_train_ds = tf.data.Dataset.list_files(f"{input_path}/train/*/*", shuffle=False)
    list_val_ds = tf.data.Dataset.list_files(f"{input_path}/val/*/*", shuffle=False)
    list_test_ds = tf.data.Dataset.list_files(f"{input_path}/test/*", shuffle=False)
 
    n_train = list_train_ds.cardinality().numpy()
    file_paths_test = [p.decode("utf-8") for p in list_test_ds.as_numpy_iterator()]
 
    def get_label(file_path):
        parts = tf.strings.split(file_path, "/")
        return tf.argmax(tf.cast(parts[-2] == class_names, tf.int32))
 
    def decode_img(img):
        # decode_image gestisce sia PNG che JPEG (decode_jpeg fallisce sui PNG!)
        img = tf.io.decode_image(img, channels=1, expand_animations=False)
        img.set_shape([None, None, 1])
        return tf.image.resize(img, [img_height, img_width])
 
    def process_path(file_path):
        img = decode_img(tf.io.read_file(file_path))
        return img, get_label(file_path)
 
    def process_path_test(file_path):
        return decode_img(tf.io.read_file(file_path))
 
    train_ds = (
        list_train_ds
        .map(process_path, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(n_train, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        list_val_ds
        .map(process_path, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        list_test_ds
        .map(process_path_test, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
 
    return train_ds, val_ds, test_ds, file_paths_test
 
