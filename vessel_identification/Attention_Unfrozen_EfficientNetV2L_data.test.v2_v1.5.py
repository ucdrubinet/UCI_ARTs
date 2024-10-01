"""
v1.5
the focal_dice_like_loss_multiclass_weighted causes the algorithm to not be punished for randomly guessing vessel pixel when there is no vessel

Because my mask is binary I need to use dice loss

v1.0

train a new model from scratch for vessel identification
"""

from jarvis.utils.general import gpus
gpus.autoselect()

import glob, numpy as np, pandas as pd, tensorflow as tf, time, random
from tensorflow.keras import Input, Model, layers, optimizers, losses, callbacks, utils
from IPython.display import clear_output, HTML, Javascript, display

import sys  
sys.path.append('/home/jjlou/Jerry/jerry_packages')
from jerry_utils import restart_kernel, load_dataset_v1, load_dataset
import jerry_losses, jerry_metrics 

name = 'Attention_Unfrozen_EfficientNetV2L_v1.5'
batch_size = 3
epochs = 30
learning_rate = 1e-3
learning_ratio = 0.99

loss = jerry_losses.dice_loss
metrics = [
    jerry_metrics.dice_metric(cls=1),
    jerry_metrics.hausdorff_metric(cls=1)
]

root = '/home/jjlou/Jerry/wsi-arterio/vessel_detection_and_rough_segmentation/data_test'

#### Attention_Unfrozen_EfficientNetV2L ####

def prepare_model(shape=(512, 512, 3)):
    
    # --- Transfer learning
    base_model = tf.keras.applications.EfficientNetV2L(input_shape=shape, include_top=False)
    base_model.trainable = True

    # Use the activations of these layers
    layer_names = [
        'block1d_add',               # 256x256
        'block2g_add',               # 128x128, layer 120
        'block4a_expand_activation', # 64x64
        'block6a_expand_activation', # 32x32, layer 551
        'top_activation'             # 16x16
    ]
    
    #Total 1028 layers, layer 551 corresponds to layer_names[3] so freeze the first 4 layers
    fine_tune_at = 551 
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    
    # --- Dr. Chang's base setup
    kwargs = {    
        'kernel_size':(3,3), 
        'padding':'same',
        'kernel_initializer':'he_normal'}
    
    # --- Define functions
    conv = lambda x, filters, strides=1, k=(3,3), r=1 : layers.SeparableConv2D(
        filters=filters,
        strides=strides,
        dilation_rate=r,
        kernel_size=k,
        padding='same',
        kernel_initializer='he_normal')(x)
    
    tran = lambda x, filters : layers.Conv2DTranspose(
        filters=filters,
        strides=(2, 2),
        **kwargs)(x)

    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.LeakyReLU()(x)
    
    conv1 = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=(2, 2))))
    
    tran2 = lambda filters, x : relu(norm(tran(x, filters)))
    
    concat = lambda x, y : layers.Concatenate()([x,y])
    
    # --- Attention
    def create_attention_gating(g, x, subsamp=(2, 2), **kwargs):
        # --- Prepare g (gating signal)
        g = layers.SeparableConv2D(filters=x.shape[-1], kernel_size=1, strides=1)(g)
        # --- Prepare x (skip connection signal)
        x = layers.SeparableConv2D(filters=x.shape[-1], kernel_size=1, strides=subsamp)(x)
        # --- Add and ReLU
        a = layers.LeakyReLU()(g + x)
        # --- Compress to single feature map + sigmoid
        a = layers.SeparableConv2D(filters=1, strides=1, kernel_size=1, activation='sigmoid')(a)
        # --- Resampler
        if subsamp != (1, 1):
            a = layers.UpSampling2D(size=subsamp)(a)
        return a
    
    # --- Create model
    inputs = tf.keras.layers.Input(shape=shape, dtype='float32')
    down = down_stack(inputs) # transfer layers from pre-trained model
    
    # Encoder
    l0 = conv1(8, inputs)
    l1 = conv1(16,concat(down[0],conv2(16, l0)))
    l2 = conv1(32,concat(down[1],conv2(32, l1)))
    l3 = conv1(48,concat(down[2],conv2(48, l2)))
    l4 = conv1(96,concat(down[3],conv2(96, l3)))
    l5 = conv1(64,concat(down[4],conv2(64, l4)))
    l6 = conv1(72,conv2(72,l5))
    l7 = conv1(96,conv2(96,l6))
    
    # Decoder  
    l = [l6, l5, l4, l3, l2, l1, l0]
    f = [96, 72, 64, 48, 32, 16, 8]
    g = l7
    for i in range(len(l)):
        x = l[i]
        a = create_attention_gating(g, x)
        g = conv1(f[i], concat(tran2(f[i], g), x * a))

    # --- Create logits
    logits = conv1(2, g)

    # --- Create model
    model = Model(inputs=inputs, outputs=logits) 
  
    return model

#tf.numpy_function creates output with unknown shape.. so need to reshape it
def reshape(image, mask):
    image = tf.reshape(image, [1,512,512,3])
    mask = tf.reshape(mask, [1,512,512,1])
    return tf.cast(image, 'uint8'), tf.cast(mask, 'uint8')

color = glob.glob(f'{root}/train_Color.Basic.Aug_x5_sharded/*/*/*')
gray = glob.glob(f'{root}/train_Gray.Basic.Aug_sharded/*/*')
train_path = color + gray
valid_path = glob.glob(f'{root}/external_cropped/*/*')

history_path = f'{root}/models_raw/{name}.csv'
model_save = f'{root}/models_raw/{name}.hdf5'

train = load_dataset_v1(train_path)
valid = load_dataset(valid_path)
valid = valid.map(lambda i,m: reshape(i,m), num_parallel_calls=tf.data.AUTOTUNE)

steps_per_epoch = len(train)//500
validation_steps= len(valid)//200

# --- Learning rate scheduler
lr_scheduler = callbacks.LearningRateScheduler(lambda epoch, lr : lr * learning_ratio)

model = prepare_model()

# --- Compile model
model.compile(
    optimizer=optimizers.Adam(learning_rate=learning_rate),
    loss=loss,
    metrics=metrics)

# --- Train
model_history = model.fit(
    x=train,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid,
    validation_steps=validation_steps,
    callbacks=[lr_scheduler])

history = pd.DataFrame.from_dict(model_history.history)
history.to_csv(history_path)

model.save(model_save)