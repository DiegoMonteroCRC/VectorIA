"""
Módulo 1: Identificación de Mosquitos con Red Convolucional (CNN)
Proyecto 2: VectorIA - CUC
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def entrenar_cnn_mosquitos(data_path="../data/raw/images/", 
                           img_size=(128, 128), 
                           batch_size=32, 
                           epochs=15, 
                           model_save_path="../models/cnn_propia_mosquitos.h5"):
    """
    Entrena la arquitectura CNN propia para clasificar las 3 especies de mosquitos
    aplicando data augmentation y pesos de clase para balancear.
    """
    image_shape = (img_size[0], img_size[1], 3)
    
    # 1. Generadores de datos con aumentación y partición 80/20
    image_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        rescale=1/255.0,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.20
    )

    test_datagen = ImageDataGenerator(
        rescale=1/255.0,
        validation_split=0.20
    )

    train_image_gen = image_gen.flow_from_directory(
        data_path,
        target_size=img_size,
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    test_image_gen = test_datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = len(train_image_gen.class_indices)
    clases_nombres = list(train_image_gen.class_indices.keys())

    # 2. Pesos de clase para compensar desbalance
    pesos = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_image_gen.classes),
        y=train_image_gen.classes
    )
    class_weights = dict(enumerate(pesos))

    # 3. Creación del Modelo (CNN sugerida)
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # 4. Entrenamiento con EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        train_image_gen,
        epochs=epochs,
        validation_data=test_image_gen,
        callbacks=[early_stop],
        class_weight=class_weights
    )

    # 5. Guardado del modelo
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Modelo guardado exitosamente en {model_save_path}")

    # 6. Evaluación y reporte de métricas
    pred_probabilities = model.predict(test_image_gen)
    predictions = np.argmax(pred_probabilities, axis=-1)
    y_true = test_image_gen.classes

    print("\nReporte de Clasificación:")
    print(classification_report(y_true, predictions, target_names=clases_nombres))

    return model


if __name__ == "__main__":
    entrenar_cnn_mosquitos()

