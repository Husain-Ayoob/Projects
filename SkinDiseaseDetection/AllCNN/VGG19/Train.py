import tensorflow as tf
from keras.applications import VGG19
from keras import layers, models

train_data_dir = 'Dataset/SkinDisease/archive/train'
test_dir = 'Dataset/SkinDisease/archive/test'

# Data Augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loading Phase with Data Augmentation
train_dataset = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse',  # assuming you have sparse labels
    shuffle=True,
    seed=123
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150, 150),
    batch_size=32)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(150, 150),
    batch_size=32)

# Load pre-trained VGG19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the layers in the base model
base_model.trainable = False

# Create a new model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(19, activation='softmax')
])

# Compilation Phase
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training Phase
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=50
)

# Evaluation Phase
loss, accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {accuracy*100:.2f}%")
