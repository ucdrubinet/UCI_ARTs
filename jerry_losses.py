"""
CHANGELOG

v0.2 10/27/2023
- use 1 - (K.sum(dice_like)/pred.shape[-1]) for dice_like_loss_multiclass_weighted

"""

import numpy as np, tensorflow as tf, keras.backend as K

# Differentiable version of argmax
def softargmax(x, beta=1e10):
    x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
    return tf.math.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

# Return the mask and prediction for class c
def mask_pred_processor(true, pred, c):
    true = true == c
    true = tf.cast(true, tf.float32)
    pred = tf.nn.softmax(pred, axis=-1)
    pred = pred[..., c]
    true = K.flatten(true)
    pred = K.flatten(pred)
    return true, pred

#### DICE-LIKE LOSS ####
# Dice-like loss
def dice_like_score(true, pred, c, alpha, beta):
    # set up mask and pred
    true = true == c
    true_d = tf.cast(true, tf.float32)
    # Higher alpha = increased FP punishment and TN reward.
    # Higher beta = increased difference in reward for TP and FN
    true_n = (true_d - alpha)*(1 + alpha + beta) 
    pred = tf.nn.softmax(pred, axis=-1)
    pred = pred[..., c]
    #true_n = tf.reshape(true_n, (true_n.shape[-1]*2))
    #pred = tf.reshape(pred, (pred.shape[-1]*2))
    true_n = K.flatten(true_n)
    pred = K.flatten(pred)
    # dice
    intersection = K.sum(true_n * pred)
    union = K.sum(true_d) + K.sum(pred)
    dice_like = (2. * intersection) / (union)
    return dice_like

# Averaged weighted multiclass dice-like loss with dice-like calculated separtely per class
# len(weights) must = pred.shape[-1] - 1
def dice_like_loss_multiclass_weighted(true, pred, alpha, beta, weights): 
    dice_like = []
    class_weights = weights/K.sum(weights)
    for c in range(pred.shape[-1]): # the class index must be pred.shape[-1].
        class_dice_like = dice_like_score(true, pred, c, alpha, beta) * class_weights[c]
        dice_like.append(class_dice_like)
    return 1 - (K.sum(dice_like)/pred.shape[-1])

def focal_dice_like_loss_multiclass_weighted(true, pred, alpha=0.05, beta=0.1, weights=[1.0, 2.0, 3.0], gamma=1.1, kappa=1.1):
    loss = dice_like_loss_multiclass_weighted(true, pred, alpha, beta, weights)
    return K.pow(loss, gamma) * kappa

#### DICE LOSS ####
# Single Class Dice loss
def dice_loss(true, pred, c=1):
    true, pred = mask_pred_processor(true, pred, c)
    intersection = K.sum(true * pred)
    union = K.sum(true) + K.sum(pred)
    return 1 - (2. * K.sum(intersection)) / K.sum(union)

# Multiclass Dice loss
def dice_score(true, pred, c):
    true, pred = mask_pred_processor(true, pred, c)
    intersection = K.sum(true * pred)
    union = K.sum(true) + K.sum(pred)
    return intersection, union

def dice_multiclass(true, pred):
    intersection = []
    union = []
    for c in range(1, pred.shape[-1]): #ignore 0 background class since dice score does this too
        i,u = dice_score(true, pred, c)
        intersection.append(i)
        union.append(u)
    return 1 - (2. * K.sum(intersection)) / K.sum(union)

# Focal multiclass dice loss
######

#### BINARY CROSS ENTROPY ####
fbce = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, from_logits=True, alpha=0.75, gamma=2.0)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def focal_binary_cross_entropy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype='uint8')
    y_pred = tf.cast(y_pred, dtype='uint8')
    return K.binary_crossentropy(y_true, y_pred, from_logits=True)

#### Multi-task Classification and Segmentation Loss ####
"""
PSEUDOCODE

Def loss function: (input this as loss function into the model)
- inputs:
    - true
    - pred
    - weights [segmentation weight, classification weight]
    - segmentation loss function (can be dice, dice_like, Tversky, etc)
        - parameters of segmentation loss function
    - classification loss function (binary cross entorpy)
- calculate weight function
- gets segmentation true and pred from the dictionary
- gets classification true and pred from the dictionary
- calculate segmentation loss using segmentation loss function
- calculate classification loss using classification loss function
- return loss (segmentation loss * s weight + calculation loss * c weight)
"""
def DiceLike_FCBE(y_true, y_pred):
    segmentation_true = y_true[0]
    segmentation_pred = y_pred[0]
    classification_true = y_true[1]
    classification_pred = y_pred[1]

    segmentation_loss = focal_dice_like_loss_multiclass_weighted(segmentation_true, segmentation_pred)
    classification_loss = focal_binary_cross_entropy(classification_true, classification_pred)
    
    loss = tf.math.reduce_mean([segmentation_loss, classification_loss])
    loss = tf.cast(loss, dtype='float64')
    
    return loss

class MultiTask_Classification_Segmentation_Loss(tf.keras.losses.Loss):
    def __init__(self, mt_weights=[1.0, 1.0], segmentation_loss_fn=None, classification_loss_fn=None, classification_weights=None, **kwargs):
        super().__init__()
        self._weights = mt_weights
        self.segmentation_loss_fn = segmentation_loss_fn
        self.classification_loss_fn = classification_loss_fn
        self.classification_weights = classification_weights
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        self.seg_weights = kwargs['seg_weights']
        self.gamma = kwargs['gamma']
        self.kappa = kwargs['kappa']
    
    def call(self, y_true, y_pred, **kwargs):
        segmentation_true = y_true
        segmentation_pred = y_pred
        classification_true = y_true
        classification_pred = y_pred
        
        MultiTask_weights = self._weights/K.sum(self._weights)
        
        segmentation_loss = self.segmentation_loss_fn(segmentation_true, segmentation_pred, alpha=self.alpha, beta=self.beta, weights=self.seg_weights, gamma=self.gamma, kappa=self.kappa) * MultiTask_weights[0]
        classification_loss = self.classification_loss_fn(y_true=classification_true, y_pred=classification_pred) * MultiTask_weights[1]
        segmentation_loss = tf.cast(segmentation_loss, dtype='float64')
        classification_loss = tf.cast(classification_loss, dtype='float64')
        
        loss = tf.math.reduce_mean([segmentation_loss, classification_loss])
        loss = tf.cast(loss, dtype='float64')
        #self.add_loss(loss)
        
        return loss
    
    def get_config(self):
        return {
            'MultiTask Weights': self.weights,
            'Segmentation Function': self.segmentation_loss_fn,
            'Classification Function': self.classification_loss_fn
        }
              

################### Under Construction #################################
"""
#Tversky loss
#https://github.com/nabsabraham/focal-tversky-unet/
def tversky(true, pred, c, smooth, alpha): 
    y_true_pos, y_pred_pos = mask_pred_processor(true, pred, c)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_multiclass(true, pred, smooth, alpha, weights):
    tversky = []
    class_weights = weights/K.sum(weights)
    for c in range(1, pred.shape[-1]): # the class index must be pred.shape[-1]. Ignore 0 background.
        class_tversky = tversky(true, pred, c, smooth, alpha) * class_weights[c]
        tversky.append(class_tversky)
    return K.sum(tversky)/pred.shape[-1]

def focal_tversky_multiclass(true, pred, smooth=1e-7, alpha=0.7, weights=[1.0, 2.5], gamma=1.1, kappa = 1.1):
    loss = tversky_multiclass(true, pred, smooth, alpha, weights)
    return K.pow(loss, gamma) * kappa
"""