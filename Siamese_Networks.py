import json
import time

import keras
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from matplotlib import pyplot as plt
from numpy import expand_dims
from numpy.random import seed
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.random_seed import set_random_seed
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

seed(3)
set_random_seed(3)
np.random.seed(42)


def rand_affine_trans(image_pairs, number_of_aug=4):
    """
    The function will create N affine augmentations for the each pair of images.
    The augmentations include: shift, shear, zoom and rotation each with a probability of 0.5.
    The parameter ranges for each augmentation is as described in the paper "Siamese Neural Networks for One-shot Image Recognition"
    :param image_pairs: A list of image pairs
    :param number_of_aug: The number of augmentations to add for each pair
    :return: list of image pairs with affine augmentations.
    """
    num_of_pairs = len(image_pairs)
    width_shift = [5, 5]
    height_shift = [5, 5]
    rotation_ang = 15
    shear_ang = 17
    zoom = [0.8, 2]
    print('Creating Affine Distortions')
    for i in tqdm(range(number_of_aug)):
        random_numbers = np.random.random(size=(num_of_pairs * 2, 4))
        for pair_index in range(num_of_pairs):
            image_1 = image_pairs[pair_index][0][:, :, :]
            image_2 = image_pairs[pair_index][1][:, :, :]

            if random_numbers[pair_index * 2, 0] > 0.5:
                datagen = ImageDataGenerator(width_shift_range=width_shift, height_shift_range=height_shift)
                img1_for_aug = expand_dims(image_1, 0)
                it = datagen.flow(img1_for_aug, batch_size=1)
                image = it.next()[0].astype('float32')
                image_1 = image
            if random_numbers[pair_index * 2, 1] > 0.5:
                datagen = ImageDataGenerator(rotation_range=rotation_ang)
                img1_for_aug = expand_dims(image_1, 0)
                it = datagen.flow(img1_for_aug, batch_size=1)
                image = it.next()[0].astype('float32')
                image_1 = image
            if random_numbers[pair_index * 2, 2] > 0.5:
                datagen = ImageDataGenerator(zoom_range=zoom)
                img1_for_aug = expand_dims(image_1, 0)
                it = datagen.flow(img1_for_aug, batch_size=1)
                image = it.next()[0].astype('float32')
                image_1 = image
            if random_numbers[pair_index * 2, 3] > 0.5:
                datagen = ImageDataGenerator(shear_range=shear_ang)
                img1_for_aug = expand_dims(image_1, 0)
                it = datagen.flow(img1_for_aug, batch_size=1)
                image = it.next()[0].astype('float32')
                image_1 = image
            if random_numbers[pair_index * 2 + 1, 0] > 0.5:
                datagen = ImageDataGenerator(width_shift_range=width_shift, height_shift_range=height_shift)
                img2_for_aug = expand_dims(image_2, 0)
                it = datagen.flow(img2_for_aug, batch_size=1)
                image = it.next()[0].astype('float32')
                image_2 = image
            if random_numbers[pair_index * 2 + 1, 1] > 0.5:
                datagen = ImageDataGenerator(rotation_range=rotation_ang)
                img2_for_aug = expand_dims(image_2, 0)
                it = datagen.flow(img2_for_aug, batch_size=1)
                image = it.next()[0].astype('float32')
                image_2 = image
            if random_numbers[pair_index * 2 + 1, 2] > 0.5:
                datagen = ImageDataGenerator(zoom_range=zoom)
                img2_for_aug = expand_dims(image_2, 0)
                it = datagen.flow(img2_for_aug, batch_size=1)
                image = it.next()[0].astype('float32')
                image_2 = image
            if random_numbers[pair_index * 2 + 1, 3] > 0.5:
                datagen = ImageDataGenerator(shear_range=shear_ang)
                img2_for_aug = expand_dims(image_2, 0)
                it = datagen.flow(img2_for_aug, batch_size=1)
                image = it.next()[0].astype('float32')
                image_2 = image

            image_pairs.append([image_1, image_2])

    return image_pairs


def create_image_pairs_from_df(df, pos_type, use_augmentation=False):
    """

    :param df: df we created from the train-test splitting txt file
    :param pos_type: wether its the positive pairs or negative (for columns names reason)
    :param use_augmentation: bool - iff T increase data size with augmentation techniques.
    :return:
    """
    arr = []
    for _, row in tqdm(df.iterrows()):
        if pos_type:
            name1 = row['PName']
            name2 = row['PName']
            idx1 = row['Pidx1']
            idx2 = row['Pidx2']

        else:
            name1 = row['AName']
            name2 = row['Nname']
            idx1 = row['Aidx']
            idx2 = row['Nidx']

        img_path1 = f'data/lfw2/lfw2/{name1}/{name1}_{f"{idx1:04}"}.jpg'
        image1 = tf.keras.preprocessing.image.load_img(img_path1, color_mode="grayscale")
        image1_arr = keras.preprocessing.image.img_to_array(image1)
        img_path2 = f'data/lfw2/lfw2/{name2}/{name2}_{f"{idx2:04}"}.jpg'
        image2 = tf.keras.preprocessing.image.load_img(img_path2, color_mode="grayscale")
        image2_arr = keras.preprocessing.image.img_to_array(image2)

        arr.append([image1_arr, image2_arr])
    if use_augmentation:
        return rand_affine_trans(arr)
    else:
        return arr


def load_datasets(use_augmentation=False):
    """

    :param use_augmentation:  bool - iff T increase data size with augmentation techniques.
    :return: train,val,test pairs after normalization
    """
    train_pos_df = pd.read_csv('data/pairsDevTrainPos.csv')
    train_neg_df = pd.read_csv('data/pairsDevTrainNeg.csv')
    test_pos_df = pd.read_csv('data/pairsDevTestPos.csv')
    test_neg_df = pd.read_csv('data/pairsDevTestNeg.csv')

    print('Loading Datasets')
    train_pos_df, val_pos_df = train_test_split(train_pos_df, test_size=0.2, random_state=42)
    train_neg_df, val_neg_df = train_test_split(train_neg_df, test_size=0.2, random_state=42)

    train_pos_arr = create_image_pairs_from_df(train_pos_df, pos_type=True, use_augmentation=use_augmentation)
    train_neg_arr = create_image_pairs_from_df(train_neg_df, pos_type=False, use_augmentation=use_augmentation)
    val_pos_arr = create_image_pairs_from_df(val_pos_df, pos_type=True)
    val_neg_arr = create_image_pairs_from_df(val_neg_df, pos_type=False)
    test_pos_arr = create_image_pairs_from_df(test_pos_df, pos_type=True)
    test_neg_arr = create_image_pairs_from_df(test_neg_df, pos_type=False)

    train_pos_arr = norm_pairs_list(train_pos_arr)
    train_neg_arr = norm_pairs_list(train_neg_arr)
    val_pos_arr = norm_pairs_list(val_pos_arr)
    val_neg_arr = norm_pairs_list(val_neg_arr)
    test_pos_arr = norm_pairs_list(test_pos_arr)
    test_neg_arr = norm_pairs_list(test_neg_arr)

    return train_pos_arr, train_neg_arr, val_pos_arr, val_neg_arr, test_pos_arr, test_neg_arr


def build_siamese_nn(input_shape, with_batchnorm, with_dropout):
    """
    Constructs the Siamese Net for training.
    :param input_shape: image size
    :param with_batchnorm: if true, batch normalization layers will be added.
    :param with_dropout: if false no dropout, else apply dropout.
    :return:
    A Siamese Net for training
    """
    kernel_initializer_convs = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
    kernel_initializer_fc = tf.keras.initializers.RandomNormal(mean=0., stddev=0.2)
    bias_initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01)

    regularization = 1e-3

    conv_net = Sequential()
    conv_net.add(Conv2D(filters=64,
                        kernel_size=(10, 10),
                        activation='relu',
                        input_shape=input_shape,
                        kernel_regularizer=l2(regularization),
                        kernel_initializer=kernel_initializer_convs,
                        bias_initializer=bias_initializer,
                        name='Conv1'))
    if with_batchnorm:
        conv_net.add(BatchNormalization())
    conv_net.add(MaxPool2D(pool_size=(3, 3), strides=(3, 3)))

    conv_net.add(Conv2D(filters=128,
                        kernel_size=(7, 7),
                        activation='relu',
                        kernel_regularizer=l2(regularization),
                        kernel_initializer=kernel_initializer_convs,
                        bias_initializer=bias_initializer,
                        name='Conv2'))

    if with_batchnorm:
        conv_net.add(BatchNormalization())

    conv_net.add(MaxPool2D(pool_size=(3, 3), strides=(3, 3)))

    conv_net.add(Conv2D(filters=128,
                        kernel_size=(4, 4),
                        activation='relu',
                        kernel_regularizer=l2(regularization),
                        kernel_initializer=kernel_initializer_convs,
                        bias_initializer=bias_initializer,
                        name='Conv3'))

    if with_batchnorm:
        conv_net.add(BatchNormalization())

    conv_net.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    conv_net.add(Conv2D(filters=256,
                        kernel_size=(4, 4),
                        activation='relu',
                        kernel_regularizer=l2(regularization),
                        kernel_initializer=kernel_initializer_convs,
                        bias_initializer=bias_initializer,
                        name='Conv4'))

    if with_batchnorm:
        conv_net.add(BatchNormalization())

    conv_net.add(Flatten())

    if with_dropout:
        conv_net.add(Dropout(0.2))

    conv_net.add(Dense(
        units=4096,
        activation='sigmoid',
        kernel_regularizer=l2(regularization),
        kernel_initializer=kernel_initializer_fc,
        bias_initializer=bias_initializer,
        name='Dense1'))

    first_img = Input(input_shape)
    second_img = Input(input_shape)

    # applying the same convolution network to both imgs
    emb_first_img = conv_net(first_img)
    emb_second_img = conv_net(second_img)

    # computing the l1 distances of the imgs features
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_dist = L1_layer([emb_first_img, emb_second_img])

    # compute similarty score of the two

    if with_dropout:
        l1_dist = Dropout(0.2)(l1_dist)

    similarity_score = Dense(1,
                             activation='sigmoid',
                             name='Dense2',
                             kernel_initializer=kernel_initializer_fc,
                             bias_initializer=bias_initializer)(l1_dist)

    model = Model(inputs=[first_img, second_img], outputs=similarity_score)
    return model


def norm_pairs_list(pairs_list):
    """
    Scaling images to the range of 0-1
    :param pairs_list: a list of pairs of images
    :return: normalized data
    """
    return [[pair[0] / 255, pair[1] / 255] for pair in pairs_list]


def build_model(initial_learning_rate, n_steps_in_training_epoch, with_batchnorm, with_dropout):
    """
    Builds a Siamese Network for training. The model uses a SGD optimizer with a learning rate decay. The decay rate is
    0.99.
    :param initial_learning_rate: initial learning rate
    :param n_steps_in_training_epoch: decay steps of learning rate
    :return: The Siamese model to train.
    """
    model = build_siamese_nn(input_shape=(250, 250, 1), with_batchnorm=with_batchnorm, with_dropout=with_dropout)
    model.summary()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=n_steps_in_training_epoch,
        decay_rate=0.99,
        staircase=False)

    # Define the optimizer and compile the model
    optimizer = SGD(learning_rate=lr_schedule, momentum=0.5)
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy', tf.keras.metrics.AUC()], optimizer=optimizer)
    return model


def create_xy_keras_input_from_pairs(pos_pairs, neg_pairs):
    """
    Adjust the pair list for keras input
    :param pos_pairs:
    :param neg_pairs:
    :return:
    """
    pos_y = [1] * len(pos_pairs)
    neg_y = [0] * len(neg_pairs)
    x = np.array(pos_pairs + neg_pairs)
    y = np.array(pos_y + neg_y)

    x_y_zipped = list(zip(x, y))
    np.random.shuffle(x_y_zipped)
    x, y = zip(*x_y_zipped)
    x, y = np.array(x), np.array(y)
    x = [x[:, 0], x[:, 1]]
    return x, y


def plot_graphs(history, fig_name):
    """
    Plot loss and accuracy graphs
    :param history:
    :param fig_name:
    :return:
    """
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model Auc')
    plt.ylabel('Auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{fig_name}_auc.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{fig_name}_loss.png')
    plt.show()


def main():
    expr_title = 'lr=0.01_reg=1e-3_drop=F_BN=F_Aug=T'
    print(expr_title)
    batch_size = 64
    initial_learning_rate = 0.01
    use_augmentation = True
    with_batchnorm = False
    with_dropout = False

    train_pos_arr, train_neg_arr, val_pos_arr, val_neg_arr, test_pos_arr, test_neg_arr = load_datasets(
        use_augmentation=use_augmentation)

    x_train, y_train = create_xy_keras_input_from_pairs(train_pos_arr, train_neg_arr)
    x_val, y_val = create_xy_keras_input_from_pairs(val_pos_arr, val_neg_arr)
    x_test, y_test = create_xy_keras_input_from_pairs(test_pos_arr, test_neg_arr)

    n_steps_in_training_epoch = int((len(train_pos_arr) + len(train_neg_arr)) / batch_size)
    model = build_model(initial_learning_rate, n_steps_in_training_epoch, with_batchnorm, with_dropout)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3, min_delta=20)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)

    print('starting training')
    start = time.time()

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=100, verbose=1,
                        validation_data=(x_val, y_val), callbacks=[es, mc])
    end = time.time()
    train_time = int(end - start)
    print(train_time)
    print('Finished training')

    # # evaluate the model
    train_loss, train_acc, train_auc = model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_acc, val_auc = model.evaluate(x_val, y_val, verbose=0)

    print('starting Evaluation')
    start = time.time()

    test_loss, test_acc, test_auc = model.evaluate(x_test, y_test, verbose=0)

    end = time.time()
    eval_time = int(end - start)

    print(eval_time)
    print('Finished Evaluation')

    # load the saved model
    # saved_model = load_model('best_model.h5')
    print('Train Acc: %.3f, Test  Acc: %.3f' % (train_acc, test_acc))

    # Log results to a json file
    expr_dict = {'history': history.history, 'params': history.params, 'title': expr_title,
                 'train_time': train_time, 'eval_time': eval_time,
                 'best_ep_results': {'train_loss': train_loss, 'train_acc': train_acc, 'train_auc': train_auc,
                                     'test_loss': test_loss, 'test_acc': test_acc, 'test_auc': test_auc,
                                     'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc}}
    curr_time = int(time.time())
    results_path = f'{expr_title}_{curr_time}'

    with open(f'{results_path}.json', 'w') as fp:
        json.dump(expr_dict, fp)

    plot_graphs(history, results_path + '_fig')


if __name__ == '__main__':
    main()
