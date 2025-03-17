import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def train_model(model, train_images, train_labels, val_images, val_labels, batch_size=8, epochs=10):
    """
    Train the model using data augmentation on the training images.
    Data augmentation is applied only to the training set.
    """
    # Define data augmentation for training images
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    # Compute any statistics required for augmentation
    datagen.fit(train_images)

    print(f'After augmentation: {train_images.shape} is the shape.')
    
    train_generator = datagen.flow(train_images, train_labels, batch_size=batch_size)

    
    # For validation, use a generator without augmentation
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(val_images, val_labels, batch_size=batch_size)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_images) // batch_size,
        validation_data=val_generator,
        validation_steps=len(val_images) // batch_size,
        epochs=epochs
    )
    return history
