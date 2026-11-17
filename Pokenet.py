"""
First read in list of Pokemon, do some data cleaning, and create a bar plot
of type frequency, with the most frequent type listed first
"""

import logging
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

from keras.preprocessing import image
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from tensorflow.keras import layers

from utils import *
import glob
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from tensorflow.keras import layers
from pathlib import Path

warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)

poke_df = pd.read_csv("pokemon.csv")

poke_df = poke_df.rename(columns={"#": "No"})
poke_df = poke_df.rename(columns={"Type 1": "Type1"})
poke_df = poke_df.rename(columns={"Type 2": "Type2"})
poke_df = poke_df.drop_duplicates(subset=['No'])
poke_df['Type2'] = poke_df.Type2.fillna('')

conditions = [poke_df['Type2'] == '', poke_df['Type2'] != '']

values = [poke_df['Type1'], poke_df['Type1'] + '|' + poke_df['Type2']]

poke_df['Types_v2'] = np.select(conditions, values)
poke_df

pokepalette = sns.color_palette(["blue","tan", "lightskyblue", "forestgreen", "magenta",
                  "chartreuse", "saddlebrown", "purple", "red", "sienna",
                  "black", "crimson", "yellow", "silver", "midnightblue",
                  "indigo", "hotpink", "cyan"])

label_freq = poke_df['Types_v2'].apply(lambda s: str(s).split('|')).explode().value_counts().sort_values(ascending=False)

poke_df['Types'] = poke_df['Type1'] + ',' + poke_df['Type2']
poke_df['Types'] = poke_df['Types'].apply(lambda s: [l for l in str(s).split(',') if l not in [""]])
poke_df.head()

# Bar plot
plt.figure(figsize=(12,10))
sns.barplot(y=label_freq.index.values, x=label_freq, order=label_freq.index, palette = pokepalette)
plt.title("Type frequency", fontsize=14)
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

"""
Create our training dataset: we load in images of Pokemon sprites from generations 1 through 5
using the get_images() function, which takes in as input the folder for each generation,
so we will run this function five times in total.
Because the number of individual Pokemon is pretty smal, we're using Pokemon sprites from multiple 
generations and games, in addition to back sprites, shiny variants, and female Pokemon as 
a built-in way of augmenting the size of our training data. Through this, our training dataset
consists of over 15,000 images.
"""
def getImages(path):
    
    poke_df2 = poke_df
    
    poke_df2 = poke_df2[poke_df2['No'] < 650] # filter Pokemon from generation 5 and below
    
    png_files = glob.glob(path + "/**/*.png", recursive = True)
    
    png_filter = [i for i in png_files if Path(i).stem.isdigit()]
    
    png_filter = [i for i in png_filter if int(Path(i).stem) in list(poke_df2['No'])]
    
    gen_poke = pd.DataFrame(png_filter)
    
    poke_df2 = poke_df2.set_index('No')
    
    
    gen_poke['Types'] = [poke_df2['Types'][int(Path(i).stem)] for i in png_filter]
    
    gen_poke = gen_poke.rename(columns={0: "sprites"})
    
    X_train, X_val, y_train, y_val = train_test_split(gen_poke['sprites'], gen_poke['Types'], test_size=0.2, random_state=44)
    
    X_train.index = list(range(len(X_train)))
    y_train.index = list(range(len(y_train)))
    
    X_val.index = list(range(len(X_val)))
    y_val.index = list(range(len(y_val)))
    
    return X_train, X_val, y_train, y_val
  
X_train_gen1, X_val_gen1, y_train_gen1, y_val_gen1 = getImages('sprites/sprites/pokemon/versions/generation-i')
X_train_gen2, X_val_gen2, y_train_gen2, y_val_gen2 = getImages('sprites/sprites/pokemon/versions/generation-ii')
X_train_gen3, X_val_gen3, y_train_gen3, y_val_gen3 = getImages('sprites/sprites/pokemon/versions/generation-iii')
X_train_gen4, X_val_gen4, y_train_gen4, y_val_gen4 = getImages('sprites/sprites/pokemon/versions/generation-iv')
X_train_gen5, X_val_gen5, y_train_gen5, y_val_gen5 = getImages('sprites/sprites/pokemon/versions/generation-v')

X_train = pd.concat([X_train_gen1, X_train_gen2, X_train_gen3, X_train_gen4, X_train_gen5], ignore_index = True)
X_val = pd.concat([X_val_gen1, X_val_gen2, X_val_gen3, X_val_gen4, X_val_gen5], ignore_index = True)
y_train = pd.concat([y_train_gen1, y_train_gen2, y_train_gen3, y_train_gen4, y_train_gen5], ignore_index = True)
y_val = pd.concat([y_val_gen1, y_val_gen2, y_val_gen3, y_val_gen4, y_val_gen5], ignore_index = True)

# Shuffle data
p = np.random.permutation(len(y_train))

X_train = X_train[p]
y_train = y_train[p]

X_train.index = list(range(len(X_train)))
y_train.index = list(range(len(y_train)))

q = np.random.permutation(len(y_val))

X_val = X_val[q]
y_val = y_val[q]

X_val.index = list(range(len(X_val)))
y_val.index = list(range(len(y_val)))

"""
We next binarize our types using one-hot encoding and convert our data to a format
amenable for multi-label classification
"""
# Fit the multi-label binarizer on the training set
print("Labels:")
mlb = MultiLabelBinarizer()
mlb.fit(y_train)

# Loop over all labels and show them
N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))
    
# transform the targets of the training and test sets
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)

# Print example of Pokemon png files and their binary targets
for i in range(3):
    print(X_train[i], y_train_bin[i])

"""
Format the dataset in such a way that TensorFlow can easily process it
"""
def parse_function(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label

def create_dataset(filenames, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    
    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset
    
train_ds = create_dataset(X_train, y_train_bin)
val_ds = create_dataset(X_val, y_val_bin)

"""
Use Transfer Learning to improve performance and decrease training time.
Then add on a dense hidden layer and an output layer corresponding to 
each of the 18 Pokemon types. Finally, we can compile and train our model
"""

IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
CHANNELS = 3 # Keep RGB color channels to match the input format of the model
BATCH_SIZE = 256 # Big enough to measure an F1-score
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 1024 # Shuffle the training data by a chunck of 1024 observations

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))

# this prevents the parameters of the pre-trained model being adjusted during the training
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(1024, activation='relu', name='hidden_layer'),
    layers.Dense(N_LABELS, activation='sigmoid', name='output')
])

model.summary()

LR = 1e-5 # Learning rate. Keep it small when transfer learning
EPOCHS = 60

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
  loss=macro_soft_f1,
  metrics=[macro_f1])

history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=create_dataset(X_val, y_val_bin))


# Visualize training and validation loss
losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history)

"""
Finally, we can test the model. I purpose trainied it on the first 5 generations of Pokemon,
because the dataset goes up to generation 6, so I was curious how well it would
perform on newer Pokemon without having seen their images in neither the training
nor validation phase. I should also note that from generation 6 onwards, Pokemon
shifted from regular 2D sprites to 3D models of Pokemon, which make it all the more
interesting to see how our model fares with data it has never seen before.
"""

def show_prediction(name, model):
    
    # Get each Pokemon's name and type from the generation 6 games
    pokeId = poke_df.loc[poke_df['Name']==name]['No'].iloc[0]
    types = poke_df.loc[poke_df['Name']==name]['Types'].iloc[0]
    img_path = os.path.join('sprites/sprites/pokemon/versions/generation-vi/x-y/', str(pokeId)+'.png')

    # Read and prepare image
    img = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
    img = image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)

    # Generate prediction
    prediction = (model.predict(img) > 0.5).astype('int')
    prediction = pd.Series(prediction[0])
    prediction.index = mlb.classes_
    prediction = prediction[prediction==1].index.values

    # Predict each Pokemon's type
    print('\n\n{}\nType\n{}\n\nPrediction\n{}\n'.format(name, types, list(prediction)))
    
Pokemon = ["Delphox",
          "Clauncher",
          "Noivern",
          "Quilladin",
          "Gogoat",
          "Hawlucha",
          "Goomy",
          "Sylveon"]

for poke in Pokemon:
    show_prediction(poke, model)