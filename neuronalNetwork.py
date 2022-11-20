import tensorflow as tf
from keras import layers, Sequential, optimizers
import matplotlib.pyplot as plt
from utils import applyFilters
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

PATH_TO_DATASET = "testImages/"
PATH_TO_DATASET_FILTER = "testImagesFilter/"
PATH_TO_FRAGMENTS = 'cropImages/'

batch_size = 32
img_height = 180
img_width = 180

# for folder in os.listdir(PATH_TO_DATASET):
#     for image in os.listdir(PATH_TO_DATASET + folder):
#         img = cv2.imread(PATH_TO_DATASET + folder + '/' + image)
#         imageWithFilter = applyFilters(img)
#         pathToFilter = PATH_TO_DATASET_FILTER + folder + '/' + image
#         cv2.imwrite(pathToFilter, imageWithFilter)


def modelCreation():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        PATH_TO_DATASET,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        PATH_TO_DATASET,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    class_names = train_ds.class_names

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    # print(np.min(first_image), np.max(first_image))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 3

    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    epochs = 50

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    # test_loss, test_acc = model.evaluate(val_ds, verbose=2)

    # predictions, labels = val_ds
    # print([predictions, np.argmax(model.predict(predictions[0]))])
    # print([labels, np.argmax(labels[0])])

    train_data = np.concatenate([x for x, y in val_ds], axis=0)
    train_label = np.concatenate([y for x, y in val_ds], axis=0)

    index_data = np.argmax(model.predict(train_data), axis=1)

    y_true = []
    y_pred = []

    for i in range(len(index_data)):
        y_true.append(class_names[train_label[i]])
        y_pred.append(class_names[index_data[i]])

    len_focas = []
    len_rocas = []
    len_agua = []

    for i in range(len(y_true)):
        if (y_true[i] == "foca"):
            len_focas.append(y_true[i])
        elif (y_true[i] == "rocas"):
            len_rocas.append(y_true[i])
        else:
            len_agua.append(y_true[i])
#     predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])
#     labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    accuracy_agua = cm[0][0]/(len(len_rocas))
    accuracy_focas = cm[1][1]/(len(len_focas))
    accuracy_rocas = cm[2][2]/(len(len_rocas))

    print("Accuracy agua: ", accuracy_agua)
    print("Accuracy focas: ", accuracy_focas)
    print("Accuracy rocas: ", accuracy_rocas)

    print("Accuracy total: ", accuracy_score(y_true, y_pred))
# for x, y in val_ds:
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names)
    # plt.plot(acc, label='accuracy')
    # plt.plot(loss, label = 'losses')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')

    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()
    disp.plot()
    plt.show()
    model.save('modelOpti.h5')


modelCreation()
