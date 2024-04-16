import glob, numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, os, gc, random, time
from IPython.display import HTML, Javascript, display
    
def restart_kernel():
    display(HTML(
        """
            <script>
                IPython.notebook.kernel.restart();
                setTimeout(function(){IPython.notebook.execute_all_cells();}, 7000)
            </script>
        """
    ))     
    
def shard_dataset(dataset=None, save_path=None):
    count = 0
    for x,y in dataset:
        mini_shard = tf.data.Dataset.from_tensors((x,y))
        mini_shard.save(f'{save_path}/shard_{count}')
        count += 1

# load large shards with many invidual patches by concatenating.. need to shuffle this (best full shuffle)
def load_dataset(load=None):
    # consumes load, a list of file paths to each shard to be loaded and concatenated into one dataset
    random.shuffle(load) 
    random.shuffle(load) 
    dataset = tf.data.Dataset.load(load[0])                        
    for s in load[1:]:
        shard = tf.data.Dataset.load(s)
        assert not len(shard) == 0, f'No elements in: {s}'
        shard = shard.shuffle(len(shard))
        dataset = dataset.concatenate(shard)
    return dataset

# use concatenate to combine many individual patches
def load_dataset_v0(load, shard_size):
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

# use tf.data.Dataset.load to load each shard in a dataset of filenames
def load(filename):
    filename = filename.decode("utf-8")
    data = tf.data.Dataset.load(filename)
    for i, m in data:
        image = i
        mask = m
    return image, mask  

def load_dataset_v1(files):
    # takes load list of filenames
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(dataset))
    dataset = dataset.map(lambda f: tf.numpy_function(load, [f], [tf.uint8, tf.uint8]), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def aug_dataset(functions=None, load_path=None, save_path=None, shard_size=None, modifier='', repeat_num=None):
    # add a modifier if aiming to "concatenate" data by saving to the same folder
    # input functions as a list
    dataset = tf.data.Dataset.load(load_path)
    num_shards = len(dataset)//shard_size + 1 # actual number of elements in shard will be smaller than shard_size
    if not glob.glob(f'{save_path}/{modifier}shard_{num_shards-1}/*/*/*.snapshot'):
        for i in range(num_shards):
            save_file = f'{save_path}/{modifier}shard_{i}'
            if not glob.glob(f'{save_file}/*/*/*.snapshot'):
                assert not glob.glob(f'{save_file}/*/*.shard'), f'{save_file} did not save'
                shard = dataset.shard(num_shards=num_shards, index=i)
                if len(shard) < 25000:
                    shard = shard.shuffle(shard.cardinality(), reshuffle_each_iteration=True)
                if repeat_num != None:
                    shard = shard.repeat(repeat_num)
                for f in functions:
                    shard = f(shard)
                shard.save(save_file)
                restart_kernel()
                time.sleep(10)
            else:
                continue
    else:
        return

def show(image):
    plt.figure(figsize=(7,7))
    plt.imshow(image)
    
def im_display(display_list, count):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'Annotated Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(f"{title[i]} {count}")
    plt.imshow((display_list[i]))
    plt.axis('off')
  plt.show()

def im_display_classification(i, l , count):
    plt.figure(figsize=(7, 7))
    plt.title(f'Input Image:{count} label:{l}')
    plt.imshow(i[0])
    plt.show()

def show_dataset(dataset=None, num=None, classification=False):
    count = 0
    if not classification:
        for i, m in dataset.take(num):
            if len(i.shape) > 3:
                i = i[0]
                m = m[0]             
            im_display([i, m], count)
            count+=1
    else:
        for i, l in dataset.take(num):
            im_display_classification(i, l, count)
            count += 1  