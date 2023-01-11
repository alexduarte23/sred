from typing import Union
import tensorflow as tf
import numpy as np
import time
import datetime
import os
import pathlib
import json



class BasicConvBlock(tf.keras.layers.Layer):
    """ Describes a basic SelfReDepth convolution block """

    def __init__(self, out_channels, down=False, **kargs):
        super(BasicConvBlock, self).__init__(**kargs)
        s = 2 if down else 1
        self.conv = tf.keras.layers.Conv2D(out_channels, 3, strides=s, padding="same")
        #self.batch = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        #x = self.batch(x)
        x = self.relu(x)
        return x

class FirstBlock(tf.keras.layers.Layer):
    """ Describes the first block of SelfReDepth's net """

    def __init__(self, num_channels, **kargs):
        super(FirstBlock, self).__init__(**kargs)
        self.conv_1 = BasicConvBlock(num_channels)#tf.keras.layers.Conv2D(num_channels, 3, padding="same", activation='relu')
        self.conv_2 = BasicConvBlock(num_channels)#tf.keras.layers.Conv2D(num_channels, 3, padding="same", activation='relu')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        return x

class DownBlock(tf.keras.layers.Layer):
    """ Describes a down-sampling block in SelfReDepth's net """

    def __init__(self, num_channels, **kargs):
        super(DownBlock, self).__init__(**kargs)
        #self.pool = tf.keras.layers.MaxPooling2D(2, padding="same")
        self.pool = BasicConvBlock(num_channels, down=True)
        self.conv = BasicConvBlock(num_channels)#tf.keras.layers.Conv2D(num_channels, 3, padding="same", activation='relu')

    def call(self, inputs):
        x = self.pool(inputs)
        x = self.conv(x)
        return x

class UpBlock(tf.keras.layers.Layer):
    """ Describes an up-sampling block in SelfReDepth's net """

    def __init__(self, channels_1, channels_2=None, cropping=None, **kargs):
        super(UpBlock, self).__init__(**kargs)
        
        self.cropping = cropping
        if channels_2 == None:
            channels_2 = channels_1
        
        #self.upsample = tf.keras.layers.UpSampling2D(2)
        self.upsample = tf.keras.layers.Conv2DTranspose(channels_1, (3,3), strides=(2,2), padding='same', activation='relu')
        if cropping != None:
            self.crop = tf.keras.layers.Cropping2D(cropping)
        self.concat = tf.keras.layers.Concatenate()
        #self.concat = tf.keras.layers.Add()
        self.conv_1 = BasicConvBlock(channels_1)#tf.keras.layers.Conv2D(channels_1, 3, padding="same", activation='relu')
        self.conv_2 = BasicConvBlock(channels_2)#tf.keras.layers.Conv2D(channels_2, 3, padding="same", activation='relu')
        

    def call(self, inputs, skip=None):
        if skip is not None:
            inputs = self.concat([skip, inputs])
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.upsample(x)
        if self.cropping is not None:
            x = self.crop(x)
        return x

class LastBlock(tf.keras.layers.Layer):
    """ Describes the last block of SelfReDepth's net """
    
    def __init__(self, num_channels, num_out, **kargs):
        super(LastBlock, self).__init__(**kargs)
        self.concat = tf.keras.layers.Concatenate()
        self.conv_1 = BasicConvBlock(num_channels)
        self.conv_2 = BasicConvBlock(num_channels)
        self.conv_last = tf.keras.layers.Conv2D(num_out, 3, padding="same")

    def call(self, inputs, skip):
        x = self.concat([skip, inputs])
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_last(x)
        return x



# Model

class SReDModel(tf.keras.Model):
    """ Full SelfReDepth denoising and inpainting model """

    def __init__(self, frame_shape: tuple[int] = (424,512,1), input_frames: int = 3, residual: int = -1, **kargs) -> None:
        """
        Parameters
        ----------
        frame_shape : tuple
            Dimensions H x W x C of each input frame

        input_frames : int
            Number of frames in each input set

        residual : int
            Index for residual computation (index to frame in input closest to t=0)
        """
        super(SReDModel, self).__init__(**kargs)
        
        channels_per_frame = frame_shape[-1]
        
        input_shape = (None, frame_shape[0], frame_shape[1], frame_shape[2] * input_frames)
        self.residual = residual

        i_crops = [int(np.ceil(frame_shape[0]/2**it)) % 2 for it in range(5)]
        j_crops = [int(np.ceil(frame_shape[1]/2**it)) % 2 for it in range(5)]
        
        self.first = FirstBlock(32)
        self.down_1 = DownBlock(32)
        self.down_2 = DownBlock(48)
        self.down_3 = DownBlock(48)
        self.down_4 = DownBlock(64)
        self.down_5 = DownBlock(128)
        self.up_1 = UpBlock(128, cropping=((i_crops[-1],0), (j_crops[-1],0)))
        self.up_2 = UpBlock(64,  cropping=((i_crops[-2],0), (j_crops[-2],0)))
        self.up_3 = UpBlock(48,  cropping=((i_crops[-3],0), (j_crops[-3],0)))
        self.up_4 = UpBlock(48,  cropping=((i_crops[-4],0), (j_crops[-4],0)))
        self.up_5 = UpBlock(32,  cropping=((i_crops[-5],0), (j_crops[-5],0)))
        self.last = LastBlock(32, channels_per_frame)
        
        # to display summary correctly
        self.call(tf.keras.layers.Input(input_shape[1:]))
        self.build(input_shape)
    
    
    def default_compile(self, optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam', loss: Union[str, tf.keras.losses.Loss] = 'mae') -> None:
        """Compile model using the default SReD parameters

        Parameters
        ----------
        optimizer : str | tf.keras.optimizers
            Regression optimizer

        loss : str | tf.keras.losses
            Fitting loss
        """
        self.compile(optimizer=optimizer, loss=loss)
    

    def default_fit(self, training_data: tf.data.Dataset, epochs: int, steps_per_epoch: int, batch_size: int = None, validation_data: tf.data.Dataset = None,
            save_output: bool = True, output_dir: str = '', label: str = '', callbacks: list[tf.keras.callbacks.Callback] = None) -> dict:
        """Fit model using the default SReD training procedure

        Parameters
        ----------
        training_data : tf.data.Dataset
            Dataset of training depth images with inpainted targets

        epochs : int
            Number of maximum epochs to run

        steps_per_epoch : int
            Number of steps per epoch

        batch_size : int
            Batch size. Infered from data if not given

        validation_data : tf.data.Dataset
            Dataset of validation data with inpainted targets

        save_output : bool
            Whether to save fitting history and important chackpoints to disk

        output_dir : str
            Location of where output should be saved

        label : str
            Optional unique id/name of this fitting process

        callbacks : list[tf.keras.callbacks]
            Custom callbacks (replace the default EarlyStopping and ReduceLROnPlateau callbacks)


        Returns
        ----------
        dict
            history, total time taken and paths of saved files
        """
        
        # prepare output paths
        if save_output:
            output_dir = pathlib.Path(output_dir)
            if label is None or label == '':
                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
                checkpoint_dir = output_dir / f"SReD_training_{st}"
            else:
                checkpoint_dir = output_dir / label

            os.makedirs(checkpoint_dir, exist_ok=True)
            #checkpoint_path = os.path.join(model_dir, "checkpoints", "cp-{epoch:04d}.h5")
            initial_weights_path = checkpoint_dir / "initial-weights.h5"
            self.save_weights(initial_weights_path)
            best_checkpoint_path = checkpoint_dir / "cp-weights-best.h5"
            last_checkpoint_path = checkpoint_dir / "cp-weights-last.h5"
            history_path = checkpoint_dir / "history.json"
        else:
            initial_weights_path = None
            best_checkpoint_path = None
            last_checkpoint_path = None
            history_path = None

        # define checkpoints
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, min_delta=0.0001, verbose=1)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(last_checkpoint_path, save_weights_only=True)
        best_cp_callback = tf.keras.callbacks.ModelCheckpoint(best_checkpoint_path, save_weights_only=True, save_best_only=True)

        if callbacks is None:
            callbacks = [reduce_lr, earlystop]
        if save_output:
            callbacks += [cp_callback, best_cp_callback]

        if batch_size is None:
            batch_size = next(iter(training_data.take(1)))[0].shape[0]

        # train model
        start_time = time.time()
        history = self.fit(
            training_data,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            callbacks = callbacks
        )
        training_time = time.time() - start_time

        # save history datapoints
        if save_output:
            with open(history_path, 'w') as f:
                if 'lr' in history.history: del history.history['lr']
                json.dump(history.history, f)

        return {
            'history': history,
            'time': training_time,
            'initial_weights_path': initial_weights_path,
            'best_checkpoint_path': best_checkpoint_path,
            'last_checkpoint_path': last_checkpoint_path,
            'history_path': history_path
        }


    def call(self, inputs):
        #f0, f1, f2 = inputs
        #in_tensor = tf.concat([f0, f1, f2], axis=-1)
        
        # Encoder
        x0 = self.first(inputs)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        x = self.down_5(x4)
        
        # Decoder
        x = self.up_1(x)
        x = self.up_2(x, x4)
        x = self.up_3(x, x3)
        x = self.up_4(x, x2)
        x = self.up_5(x, x1)
        x = self.last(x, x0)
        
        # Residual
        if self.residual != None:
            f_res = inputs[..., self.residual, np.newaxis]
            x = f_res - x
        
        return x