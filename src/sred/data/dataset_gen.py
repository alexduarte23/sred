import tensorflow as tf
import numpy as np
import pathlib

def group_filenames(dirs: list, items: list) -> list:
    """Creates list of PNG filenames grouped by the given item layout

    Parameters
    ----------
    dirs : list
        Directory paths from where to extract filenames
    
    items : list
        Grouping layout
    
    Returns
    -------
    list
        Grouped filenames
    """
    def create_group(files, i_0):
        fitted_items = np.clip(np.array(items) + i_0, 0, len(files)-1)
        return [bytes(files[i]) for i in fitted_items]
    
    data = []
    for d in dirs:
        filenames = sorted(list(pathlib.Path(d).glob('*.png')))
        for i in range(len(filenames)):
            data += [create_group(filenames, i)]
    return data

def create_filename_dataset(dirs: list, items: list, seed: int = 12345, deep_shuffle: bool = True) -> tf.data.Dataset:
    """Creates tensorflow dataset of shuffled and grouped PNG filenames

    Parameters
    ----------
    dirs : list
        Directory paths from where to extract filenames
    
    items : list
        Grouping layout
    
    seed : int
        Shuffling seed
    
    deep_shuffle : bool
        Whether to fully shuffle the data
    
    Returns
    -------
    tf.data.Dataset
        Shuffled and grouped dataset of filenames
    """

    # Group filenames
    groups = group_filenames(dirs, items)
    # Shuffle
    if deep_shuffle:
        np.random.seed(seed)
        np.random.shuffle(groups)
    # Convert to dataset
    dataset = tf.data.Dataset.from_tensor_slices(groups)
    
    return dataset, len(groups)


def build_all_datasets(input_dirs: list, target_dirs: list, batch_size: int, val_split: float = 0, test_split: float = 0, test_with_targets: bool = False,
        input_items: list = [-4,-2,0], target_items: list = [-1], test_items: list = [-2,-1,0], seed: int = 12345) -> tuple[tf.data.Dataset]:
    """Build train, validation and test datatsets suited for the SReD model

    Parameters
    ----------
    input_dirs : list
        Directory paths from where to extract input filenames
    
    target_dirs : list
        Directory paths from where to extract target filenames

    batch_size : int
        Batch size for the train and validation datasets

    val_split : float
        Validation split [0..1]

    test_split : float
        Test split [0..1]

    test_with_targets : bool
        Whether the test dataset should have with target frames

    input_items : list
        Input grouping layout

    target_items : list
        Target grouping layout

    test_items : list
        Test input grouping layout
    
    seed : int
        Shuffling seed
    
    Returns
    -------
    tuple[tf.data.Dataset]
        Built datasets: train_ds, [..val_ds, [..test_ds]]
    """

    train_input_fds, dataset_size = create_filename_dataset(input_dirs, input_items, seed)
    test_input_fds, _ = create_filename_dataset(input_dirs, test_items, seed)
    target_fds, _ = create_filename_dataset(target_dirs, target_items, seed)

    train_fds = tf.data.Dataset.zip((train_input_fds, target_fds))
    test_fds = tf.data.Dataset.zip((test_input_fds, target_fds))

    val_size = int(dataset_size * val_split)
    test_size = int(dataset_size * test_split)

    val_ds = test_fds.take(val_size)
    test_ds = test_fds.skip(val_size).take(test_size)
    train_ds = train_fds.skip(val_size + test_size)

    # Online shuffle train
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(10)

    # Decode filenames to image tensors
    val_ds = val_ds.map(lambda x, y: (_decode_d(x), _decode_d(y)))
    train_ds = train_ds.map(lambda x, y: (_decode_d(x), _decode_d(y)))
    if test_with_targets:
        test_ds = test_ds.map(lambda x, y: (_decode_d(x), _decode_d(y)))
    else:
        test_ds = test_ds.map(lambda x, y: _decode_d(x))

    val_ds = val_ds.batch(batch_size)
    train_ds = train_ds.batch(batch_size)
    test_ds = test_ds.batch(1)

    out = (train_ds,)
    if val_size > 0: out += (val_ds,)
    if test_size > 0: out += (test_ds,)
    if len(out) == 1: out = train_ds

    return out

def build_dataset(input_dirs: list, input_items: list, batch_size: int, target_dirs: list = None, target_items: list = None,
        seed: int = 12345, deep_shuffle: bool = False, repeat: bool = False, cont_shuffle: int = 0) -> tf.data.Dataset:
    """Build dataset of depth images

    Parameters
    ----------
    input_dirs : list
        Directory paths from where to extract input filenames

    input_items : list
        Input grouping layout
    
    batch_size : int
        Batch size for the train and validation datasets

    target_dirs : list
        Directory paths from where to extract target filenames.
        If not provided then input_dirs is used

    target_items : list
        Target grouping layout.
        If not provided then dataset is built without targets
    
    seed : int
        Deeep shuffling seed
    
    deep_shuffle : int
        Whether to fully shuffle the data
    
    repeat : int
        Whether to make the dataset infinitely repeat
    
    cont_shuffle : int
        If positive then dataset will continuously shuffle
        using a shuffling winodw of the given size
    
    Returns
    -------
    tf.data.Dataset
        Built datasets
    """

    input_fds, _ = create_filename_dataset(input_dirs, input_items, seed, deep_shuffle)
    if target_items is not None:
        target_dirs = input_dirs if target_dirs is None else target_dirs
        target_fds, _ = create_filename_dataset(target_dirs, target_items, seed, deep_shuffle)
        fds = tf.data.Dataset.zip((input_fds, target_fds))
        ds = fds.map(lambda x, y: (_decode_d(x), _decode_d(y)))
    else:
        ds = input_fds.map(lambda x: _decode_d(x))
    
    if repeat: ds = ds.repeat()
    if cont_shuffle > 0: ds = ds.shuffle(cont_shuffle)
    if batch_size > 0: ds = ds.batch(batch_size)
    
    return ds

def build_test_dataset(input_dirs: list, items: list = [-2,-1,0], seed: int = 12345, deep_shuffle: bool = False) -> tf.data.Dataset:
    """Build dataset of depth images

    Parameters
    ----------
    input_dirs : list
        Directory paths from where to extract input filenames

    items : list
        Input grouping layout
    
    seed : int
        Deeep shuffling seed
    
    deep_shuffle : int
        Whether to fully shuffle the data

    Returns
    -------
    tf.data.Dataset
        Built datasets
    """
    test_fds, _ = create_filename_dataset(input_dirs, (-2, -1, 0), seed, deep_shuffle)
    test_ds = test_fds.map(lambda x: _decode_d(x))
    test_ds = test_ds.batch(1)
    
    return test_ds




def _read_img(filename, dtype):
    """ Read image into tensor of float32 [0..1] values """
    img_tensor = tf.io.read_file(filename)
    img_tensor = tf.io.decode_png(img_tensor, dtype=dtype)
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32) # also scales to [0,1]
    
    #img_tensor = tf.image.random_flip_left_right(img_tensor)
    return img_tensor

def _join(tensors):
    """ Join image tensors """
    return tf.concat(tensors, axis=-1)

def _decode_d(filenames):
    """ Decode tensor of filenames into tensor of normalized depth images """
    n = filenames.shape[0]
    images = [_read_img(filenames[i], tf.dtypes.uint16) for i in range(n)]
    return tf.concat(images, axis=-1)

def _decode_rgb(filenames):
    """ Decode tensor of filenames into tensor of normalized RGB images """
    n = filenames.shape[0]
    images = [_read_img(filenames[i], tf.dtypes.uint8) for i in range(n)]  
    return tf.concat(images, axis=-1)