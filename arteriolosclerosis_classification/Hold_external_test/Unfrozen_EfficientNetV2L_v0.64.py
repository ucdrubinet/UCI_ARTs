"""
CHANGELOG

v0.64
- remove gray

v0.61
- add in steps_per_epoch = len(train)//batch_size
- shuffle train fully

v0.2
- use color x38 and gray x19

v0.1 11/6/2023
- add one layer to dense layers
- make l5 and l6 the widest
"""

from jarvis.utils.general import gpus
gpus.autoselect()

import glob, numpy as np, pandas as pd, tensorflow as tf, os, datetime
from tensorflow.keras import Input, Model, layers, optimizers, losses, callbacks, utils
from IPython.display import clear_output, HTML, Javascript, display

import sys  
sys.path.append('/home/jjlou/Jerry/jerry_packages')
from jerry_utils import restart_kernel, load_dataset

name = 'Unfrozen_EfficientNetV2L_v0.64'
root = '/home/jjlou/Jerry/wsi-arterio/arteriosclerotic_vessel_detection_and_fine_segmentation/Arteriolosclerosis_classification/data_test'
batch_size = 30
epochs = 50
learning_rate = 1e-3
learning_ratio = 0.99
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

#### Unfrozen_EfficientNetV2L ####

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
    
    # Flatten
    f0 = tf.keras.layers.Flatten(input_shape=(4, 4))(l7)
    f1 = tf.keras.layers.Dense(32, activation='gelu')(f0)
    f2 = tf.keras.layers.Dense(8, activation='gelu')(f1)

    # --- Create logits
    logits = tf.keras.layers.Dense(2)(f2)

    # --- Create model
    model = Model(inputs=inputs, outputs=logits) 
  
    return model

model_path = f'{root}/models_raw/{name}.hdf5'
history_path = f'{root}/models_raw/{name}.csv'
train_path1 = glob.glob(f'{root}/Train_Color.BasicMorph.Aug_x38/*')
train_path2 = glob.glob(f'{root}/Train_Gray.BasicMorph.Aug_x19/*')
train_path = train_path1 # remove gray
valid_path = glob.glob(f'{root}/external_processed/*')

train = load_dataset(train_path)
train = train.shuffle(len(train))
valid = load_dataset(valid_path)
steps_per_epoch = len(train)//batch_size

model = prepare_model()

lr_scheduler = callbacks.LearningRateScheduler(lambda epoch, lr : lr * learning_ratio)

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
    callbacks=[lr_scheduler])

history = pd.DataFrame.from_dict(model_history.history)
history.to_csv(history_path)

model.save(model_path)