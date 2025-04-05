!pip install tensorflow
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
from google.colab import drive
drive.mount('/content/drive')
dataset =tf.keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/Colab Notebooks/Potato",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names=dataset.class_names
class_names
len(dataset)
for image_batch,labels_batch in dataset.take(1):
  plt.imshow(image_batch[1].numpy().astype("uint8"))
  plt.title(class_names[labels_batch[0]])
  plt.axis("off")
  plt.figure(figsize=(10,10))
for image_batch,label_batch in dataset.take(1):
  for i in range(12):
    ax=plt.subplot(3,4,i+1)
    plt
    len(dataset)
    train_size=0.8
len(dataset)*train_size
train_ds=dataset.take(54)
len(train_ds)
test_ds=dataset.skip(54)
len(test_ds)
val_size=0.1
len(dataset)*val_size
val_ds = test_ds.take(6)
len(val_ds)
def get_dataset_partitions_tf(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=10000):
  ds_size=len(ds)
  if shuffle:
    ds=ds.shuffle(shuffle_size,seed=12)
  train_size=int(train_split*ds_size)
  val_size=int(val_split*ds_size)

  train_ds =ds.take(train_size)
  val_ds = ds.skip(train_size).take(val_size)
  test_ds = ds.skip(train_size).skip(val_size)

  return train_ds,val_ds,test_ds
  train_ds,val_ds,test_ds=get_dataset_partitions_tf(dataset)
  len(train_ds)
  len(val_ds)
  len(test_ds)
  train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
data_augmentation = Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])
IMAGE_SIZE=224
resize_and_rescale = Sequential([
  layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.Rescaling(1.0/255)
])
input_shape = (IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes = 3
model=tf.keras.Sequential([
    resize_and_rescale,
    layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_classes,activation='softmax')
])
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
EPOCHS=10
history=model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)
scores=model.evaluate(test_ds)
scores
_,accuracy=model.evaluate(test_ds)
print(f"Test Accuracy :{accuracy}")
history
history.params
history.history.keys()
for images_batch ,labels_batch in train_ds.take(1):
  plt.imshow(images_batch[0].numpy().astype("uint8"))
  print(images_batch[0].numpy().astype("uint8"))
  first_image=images_batch[0].numpy().astype("uint8")
  first_label=labels_batch[0].numpy()

  print("first image to predict")
  plt.imshow(first_image)
  print("actual label",class_names[first_label])

  batch_prediction=model.predict(images_batch)
  print("predicted label",class_names[np.argmax(batch_prediction[0])])
