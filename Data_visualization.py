import os, cv2
import numpy as np
import pickle, scipy
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

tf.__version__

def load_image_files(data_path):

    data_dir_list = os.listdir(data_path)
    img_data = []
    for dataset in data_dir_list:
        img_list = os.listdir(data_path + '/' + dataset)
        print('Loaded the images of dataset-' + '{}\n'.format(dataset))
        for img in img_list:
            input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
            input_img_resize = cv2.resize(input_img, (224, 224))
            img_data.append(input_img_resize)

    img_data = np.array(img_data)
    return img_data

def load_feature_files(path,feature_file):

    f = open(feature_file, 'rb')
    feature_vectors = (pickle.load(f))[0]
    f.close

    f = open('action_reco_label.pickle', 'rb')
    names = (pickle.load(f))
    f.close

    feature_vectors = np.array(feature_vectors)
    feature_vectors = np.reshape(feature_vectors,(np.shape(feature_vectors)[0],4096))
    print("feature_vectors_shape:", np.shape(feature_vectors))
    print ("num of images:",np.shape(feature_vectors)[0])
    print("size of individual feature vector:", np.shape(feature_vectors)[1])

    num_of_samples = np.shape(feature_vectors)[0]

    metadata_file = open(os.path.join(path, 'metadata_4_classes.tsv'), 'w')
    metadata_file.write('Class\tName\n')

    k = 100  # num of samples in each class
    j = 0
    for i in range(num_of_samples):
        c = names[i]
        if i % k == 0:
          j = j + 1

        metadata_file.write('{}\t{}\n'.format(j, c))
    metadata_file.close()

    return feature_vectors,names

def images_to_sprite(data):

    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
    # Inverting the colors seems to look better for MNIST
    # data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

def sprite_image_generation(path,img_data):
    sprite = images_to_sprite(img_data)
    cv2.imwrite(os.path.join(path, 'sprite_24_classes.png'), sprite)
    scipy.misc.imsave(os.path.join(path, 'sprite.png'), sprite)

def tensorboard_execeution(path,feature_vectors):
    with tf.Session() as sess:
        features = tf.Variable(feature_vectors, name='features')
        saver = tf.train.Saver([features])

        sess.run(features.initializer)
        saver.save(sess, os.path.join(path, 'images_4_classes.ckpt'))

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = features.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join(path, 'metadata_4_classes.tsv')
        # Comment out if you don't want sprites
        # embedding.sprite.image_path = os.path.join(path, 'sprite_24_classes.png')
        # embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(path), config)

if __name__ == '__main__':

    feature_file = 'action_reco_features.pickle'

    if not (os.path.exists('embedding-logs')):
        os.mkdir('embedding-logs')
        path = os.getcwd() + '/embedding-logs'
    else:
        path = os.getcwd() + '/embedding-logs'

    feature_vectors, names = load_feature_files(path, feature_file)
#    img_data = load_image_files('data_path')
#    sprite_image_generation(path, img_data)
    tensorboard_execeution(path, feature_vectors)
    
    
