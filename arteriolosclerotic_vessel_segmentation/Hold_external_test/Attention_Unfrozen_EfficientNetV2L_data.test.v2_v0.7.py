"""
v0.7 
- color x50, gray x10
- .43.17

v0.6
- add in shuffle buffer 10000
- change color from .90.60 to .43.17
- color x30, gray x30
- change back to original load_dataset version

v0.5
- change color from .90.60 to .43.17
- color x30, gray x30

v0.4
- color x30, gray x30

v0.3
- color x40, gray x20

v0.2
- color x45, gray x15

v0.1
- color x48, gray x12

v0.0
- color x50, gray x10
"""

from jarvis.utils.general import gpus
gpus.autoselect()

import glob, numpy as np, pandas as pd, tensorflow as tf, time, random
from tensorflow.keras import Input, Model, layers, optimizers, losses, callbacks, utils
from IPython.display import clear_output, HTML, Javascript, display

import sys  
sys.path.append('/home/jjlou/Jerry/jerry_packages')
from jerry_utils import restart_kernel, load_dataset_v1
import jerry_losses, jerry_metrics 

name = 'Attention_Unfrozen_EfficientNetV2L_v0.7'
batch_size = 30
epochs = 500
learning_rate = 1e-2/5
learning_ratio = 0.99
loss = jerry_losses.focal_dice_like_loss_multiclass_weighted
metrics = [
    jerry_metrics.dice_metric(cls=1),
    jerry_metrics.hausdorff_metric(cls=1)
]

root = '/home/jjlou/Jerry/wsi-arterio/arteriosclerotic_vessel_detection_and_fine_segmentation/Vessel_WallsLumen_Segmentation/data_test_v2'

"""
def load_dataset(load, shard_size):
    random.shuffle(load) 
    random.shuffle(load)
    frags = []
    for i in range(0, len(load), shard_size):
        frag = tf.data.Dataset.load(load[i])
        for s in load[i+1:i+shard_size]:
            shard = tf.data.Dataset.load(s)
            frag = frag.concatenate(shard)
        frags.append(frag)
    dataset = frags[0]
    for f in frags[1:]:
        dataset = dataset.concatenate(f)
    return dataset    
"""

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
    logits = conv1(3, g)

    # --- Create model
    model = Model(inputs=inputs, outputs=logits) 
  
    return model

def make_predictions(dataset=None, num=1):
    for image, mask in dataset.take(num):
        pred = model.predict(image)
        pred = tf.math.argmax(pred, axis=-1)
        pred = pred[..., tf.newaxis]
        mask = tf.stack([mask, mask, mask], -1) 
        mask = tf.squeeze(mask, 3) # allow np stacking of all arrays later on by converting to same shape
        pred = tf.stack([pred, pred, pred], -1)
        pred = np.squeeze(pred, 3)
        progression.append([image, mask, pred])

#Save observations of how the model trains after each epoch
class SaveCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        make_predictions(train,1) # Sets what to look at when examining pics of training process
        
def reshape(image, mask):
    mask = tf.reshape(mask, [1,512,512,1])
    return image, mask

train1 = glob.glob(f'{root}/train_Color.Basic.Aug_x50_.43.17/*/*/*')
train2 = glob.glob(f'{root}/train_Gray.Color.BasicMorph.Aug_x10/*/*/*')
model_path = f'{root}/models_raw/{name}.hdf5'
history_path = f'{root}/models_raw/{name}.csv'
progression_path = f'{root}/models_raw/{name}.npy'

train_path = train1 + train2
train = load_dataset_v1(train_path)
steps_per_epoch = int(len(train)//batch_size)
        
model = prepare_model()
progression = []

# --- Learning rate scheduler
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
    callbacks=[lr_scheduler, SaveCallback()])

history = pd.DataFrame.from_dict(model_history.history)
history.to_csv(history_path)

progression = np.stack(progression)
np.save(progression_path, progression)

model.save(model_path)