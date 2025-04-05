🍠 Potato Leaf Disease Classification using CNN
This project uses a Convolutional Neural Network (CNN) with TensorFlow and Keras to classify potato leaf images into three categories. It includes data preprocessing, model building, training, evaluation, and prediction.

The dataset used in this project is a collection of potato leaf images with three classes:

1.Healthy

2.Early Blight

3.Late Blight

The dataset is loaded from Google Drive using TensorFlow's image_dataset_from_directory.

🛠️ Requirements
Make sure you have the following libraries installed:
!pip install tensorflow matplotlib


🚀 How It Works
1. Data Loading
Images are loaded from Google Drive:

from google.colab import drive
drive.mount('/content/drive')

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/Colab Notebooks/Potato",
    shuffle=True,
    image_size=(256, 256),
    batch_size=32
)
2. Data Partitioning
The dataset is split into training, validation, and test sets using a custom function.

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


3. Preprocessing and Augmentation
Images are resized and rescaled. Data augmentation includes random flipping and rotation.

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(224, 224),
    layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])


4. Model Architecture
A CNN with 5 convolutional layers is used for classification.

model = tf.keras.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    ...
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

5. Model Compilation and Training
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(train_ds, epochs=10, validation_data=val_ds)

6. Evaluation
loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy}")

8. Prediction Example
for images_batch, labels_batch in train_ds.take(1):
    prediction = model.predict(images_batch)
    print("Predicted:", class_names[np.argmax(prediction[0])])
    print("Actual:", class_names[labels_batch[0].numpy()])

   
📈 Accuracy Achieved
The model achieved around 0.98828125% accuracy on the test dataset.
✍️ Author
Sachin Sarvesh S C
