import numpy as np, tensorflow as tf, keras.backend as K
from scipy.spatial.distance import directed_hausdorff

def true_pred(true, pred, c):
    if c == 0: # assume we never really want to calculate metrics for background 0 class so 0 is for all classes
        true = true[..., 0]
        pred = tf.math.argmax(pred, axis=-1)
    else:
        true = true[..., 0] == c
        pred = tf.math.argmax(pred, axis=-1) == c
    return true, pred

# --- Dice
def dice_metric(y_true=None, y_pred=None, cls=None):    
    def dice(y_true, y_pred, c):
        true, pred = true_pred(y_true, y_pred, c)
        true = tf.cast(true, dtype=tf.int64)
        pred= tf.cast(pred, dtype=tf.int64)
        A = tf.math.count_nonzero(true & pred) * 2
        B = tf.math.count_nonzero(true) + tf.math.count_nonzero(pred)
        if tf.math.count_nonzero(true) == 0 and tf.math.count_nonzero(pred) == 0:
            A = tf.constant(1, dtype=tf.int64)
            B = tf.constant(1, dtype=tf.int64)
        return A / B
    
    def dice_all(y_true, y_pred):
        return dice(y_true, y_pred, 0) 
    
    def dice_1(y_true, y_pred):
        return dice(y_true, y_pred, 1)
    
    def dice_2(y_true, y_pred):
        return dice(y_true, y_pred, 2)
    
    def dice_3(y_true, y_pred):
        return dice(y_true, y_pred, 3)
    
    def dice_4(y_true, y_pred):
        return dice(y_true, y_pred, 4)
    
    def dice_5(y_true, y_pred):
        return dice(y_true, y_pred, 5)
    
    def dice_6(y_true, y_pred):
        return dice(y_true, y_pred, 6)
    
    def dice_7(y_true, y_pred):
        return dice(y_true, y_pred, 7)
    
    def dice_8(y_true, y_pred):
        return dice(y_true, y_pred, 8)
    
    def dice_9(y_true, y_pred):
        return dice(y_true, y_pred, 9)

    funcs = {
        0: dice_all,
        1: dice_1,
        2: dice_2,
        3: dice_3,
        4: dice_4,
        5: dice_5,
        6: dice_6,
        7: dice_7,
        8: dice_8,
        9: dice_9}
    
    assert cls < 10, 'ERROR only up to 9 classes implemented in custom.dsc() currently'
    
    return [funcs[i] for i in range(cls)]


# --- Hausdorff
def hausdorff_metric(y_true=None, y_pred=None, cls=None):    
    def hausdorff(y_true, y_pred, c):
        true, pred = true_pred(y_true, y_pred, c)
        true = tf.squeeze(true)
        pred = tf.squeeze(pred)
        def run_hd(true, pred):
            out = directed_hausdorff(true, pred)[0]
            out = np.float32(out)
            return out
        d = tf.numpy_function(run_hd, [true, pred], (tf.float32))
        return d
    
    def hausdorff_all(y_true, y_pred):
        return hausdorff(y_true, y_pred, 0) 
    
    def hausdorff_1(y_true, y_pred):
        return hausdorff(y_true, y_pred, 1)
    
    def hausdorff_2(y_true, y_pred):
        return hausdorff(y_true, y_pred, 2)
    
    def hausdorff_3(y_true, y_pred):
        return hausdorff(y_true, y_pred, 3)
    
    def hausdorff_4(y_true, y_pred):
        return hausdorff(y_true, y_pred, 4)
    
    def hausdorff_5(y_true, y_pred):
        return hausdorff(y_true, y_pred, 5)
    
    def hausdorff_6(y_true, y_pred):
        return hausdorff(y_true, y_pred, 6)
    
    def hausdorff_7(y_true, y_pred):
        return hausdorff(y_true, y_pred, 7)
    
    def hausdorff_8(y_true, y_pred):
        return hausdorff(y_true, y_pred, 8)
    
    def hausdorff_9(y_true, y_pred):
        return hausdorff(y_true, y_pred, 9)

    funcs = {
        0: hausdorff_all,
        1: hausdorff_1,
        2: hausdorff_2,
        3: hausdorff_3,
        4: hausdorff_4,
        5: hausdorff_5,
        6: hausdorff_6,
        7: hausdorff_7,
        8: hausdorff_8,
        9: hausdorff_9}
    
    assert cls < 10, 'ERROR only up to 9 classes implemented in custom.dsc() currently'
    
    return [funcs[i] for i in range(cls)]
