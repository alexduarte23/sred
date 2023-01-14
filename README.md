# SelfReDepth: A Self-Supervised Real-Time Depth Denoising and Restoration Approach for Consumer-Grade Sensors

![SelfReDepth's architecture](./images/banner.png)

SelfReDepth is a self-supervised deep learning approach to produce denoised and inpainted depth maps from RGB-D video sequences captured with depth sensors, such as the Kinect v2. The provided implementation exhibits real-time performance during inference if ran on sufficiently powerfull graphical hardware.

SelfReDepth was fine-tuned and tested particularly with Time-of-Flight depth sensing technology in mind, but it is fully prepared to denoise and restored depth maps captured with different sensing technologies (e.g. Structure Light). To operate SelfReDepth needs only to be provided with the raw depth + RGB frame sequences produced by a consumer-grade sensor, and ideally also the sensor's intrinsic and extrinsic parameters.


## Installation

<br>

**Warning:** Only native Windows is supported out-of-the-box for now. To add support for Unix or MacOS, [dynamic_linked_utils.py](./src/sred/utils/dynamic_linked_utils.py) and [fast_utils/](./src/sred/utils/fast_utils/) must be altered and compiled so that the shared library functions in your platform.

<br>

### **Install Package**
To install SelfReDepth you should clone this repository and locally install the package:

    $ git clone https://github.com/alexduarte23/sred.git
    $ cd sred
    $ pip install --upgrade pip setuptools wheel
    $ pip install -e .

Certain sections of SelfReDepth also use tkinter (usually packaged with python). If you run into missing tkinter errors while using SelfReDepth, follow the instructions provided in the [tkinter documentation](https://tkdocs.com/tutorial/install.html).



### **Compile C++ Code (optional)**

Part of SelfReDepth is implemented in C++ and linked to python as a shared library.
Pre-compiled instances of this library for Windows and Linux are already [provided](./src/sred/utils/fast_utils/shared/). If these don't work on your system or you're using MacOS, you must re-build fast_utils from source.

On Windows:
1. Build fast_utils:
    - [Windows] Open the provided Visual Studio 2020 solution [here](./src/sred/utils/fast_utils/), and build the project (ctrl+B) in Release mode and x64 architecture. The resulting file should be at <./src/sred/utils/fast_utils/x64/Release/fast_utils.dll>.
    - [Linux/MacOS] Run the provided Makefile: `$ make`. The resulting file should be at <./src/sred/utils/fast_utils/fast_utils.so> (or .dylib for MacOS).
2. In <./src/sred/utils/fast_utils/shared/>, replace the provided shared library file by the one you just compiled.

    - Alternatively, you can also re-direct the utils module to your re-compiled file.
        ```python
        sred.utils.setDLL('your_dll_path.dll') # or .so for Linux and .dylib for MacOS
        ```


## GPU Support

SelfReDepth supports GPU accelerated computation through Tensorflow. To enable it you must have and NVIDIA graphics card and install the CUDA and cuDNN versions compatible with your tensorflow version.

More information on this can be found in the following links:
- [Tensorflow webpage on GPU support](https://www.tensorflow.org/install/pip#windows-native)
- [CUDA and cuDNN version compatibility](https://www.tensorflow.org/install/source#gpu)
- [Installation guide by NVIDIA](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)

**Warning**: Tensorflow only provides GPU support for native Windows up to version 2.10. As such, during installation on a Windows system, SelfReDepth limits the Tensorflow installation to said version. If you wish to use a more recent version, upgrade Tensorflow manually. E.g.: `pip install tensorflow --upgrade`


## How To Use

In [example](./examples/train_and_test.ipynb) you can find a working example written as a jupyter notebook.
    - To run the example you need to install jupyter and gdown:

        $ pip install notebook gdown

More succinctly, to **prepape the model** with your data you should:
1. Import SelfReDepth
    
    ```python
    import sred
    ```
2. Generate targets for the model (saved to disk)
    
    ```python
    target_gen = sred.data.TargetGenerator(target_gen_params)
    target_gen.generate(d_dirs, rgb_dirs, target_dirs, exist_ok=True)
    ```

3. Build the training datasets
    
    ```python
    train_ds, val_ds, test_ds = sred.data.build_all_datasets(
        d_dirs,
        target_dirs,
        batch_size = 4,
        val_split = 0.1,
        test_split = 0.04
    )
    ```

4. Define and train the model
    
    ```python
    model = sred.SReDModel(frame_shape=(424,512,1), input_frames=3, residual=-1)
    model.default_compile()
    model.summary()
    res = model.default_fit(
        train_ds,
        epochs = 100,
        steps_per_epoch = 200,
        validation_data = val_ds,
        output_dir = output_dir
    )
    ```

You can then **use the train model** to denoise and inpaint depth maps:

```python
model.predict(test_ds)
```
OR

```python
model.predict(image) # ndarray of shape (1,H,W,C)
```

## Extra

SelfReDepth also comes with additional image handling and dataset generation functions that might be useful during development.

As well as 3 GUI [tools](./src/sred/tools/), of which the following two are particularly relevant:
- **depth_viewer**: helps visualizing depth videos.
- **registration_tuner**: helps determining and fine-tuning the sensor instinsic and extrinsic paraemters used for target generation section of SelfReDepth.

Which can be accessed by importing:

```python
import sred.tools
```