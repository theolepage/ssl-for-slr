import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import Callback

MOCO_DEFAULT_QUEUE_SIZE = 10000
MOCO_DEFAULT_INFO_NCE_TEMP = 0.07
MOCO_DEFAULT_EMBEDDING_DIM = 512
MOCO_DEFAULT_PROTO_NCE_LOSS_FACTOR = 0.25
MOCO_DEFAULT_N_CLUSTERS = 5000
MOCO_DEFAULT_CLUSTER_NEG_COUNT = 10000
MOCO_DEFAULT_EPOCHS_BEFORE_PROTO_NCE = 60

class MoCoModel(Model):
    '''
    Momentum contrastive learning (MoCo) for audio signals,
    implemented as a Keras model.

    "Self-supervised Text-independent Speaker Verification using Prototypical Momentum Contrastive Learning"
    Wei Xia, Chunlei Zhang, Chao Weng, Meng Yu, Dong Yu
    https://arxiv.org/pdf/2012.07178.pdf
    '''

    def __init__(self,
                 encoder_q,
                 encoder_k,
                 config,
                 weight_regularizer=0.0):
        super().__init__()

        self.config = config
        self.reg = regularizers.l2(weight_regularizer)

        self.enable_proto_nce = False

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.mlp = MLP(self.config['embedding_dim'])

        self.kmeans = None
        self.kmeans_temp = None
        # self.kmeans = faiss.Kmeans(self.config['embedding_dim'],
        #                            self.config['nb_clusters'])

        with tf.device("CPU:0"):
            queue_shape = [self.config['queue_size'], self.config['embedding_dim']]
            self.queue = tf.random.normal(queue_shape)

        update_model_weights_with_ema(self.encoder_q, self.encoder_k, 0.1)

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def call(self, X):
        return self.encoder_q(self.mlp(X))

    def train_step(self, data):
        X, _ = data # Discard labels provided by the dataset generator
        # X shape: (batch_size, frame_length, 40, 1)

        X_1_aug, X_2_aug = X

        with tf.GradientTape() as tape:
            Z_q = self.encoder_q(X_1_aug, training=True)
            Z_k = self.encoder_k(X_2_aug, training=False)
            # Out shape: (batch_size, encoded_dim)

            Z_q = self.mlp(Z_q, training=True)
            Z_k = self.mlp(Z_k, training=True)
            # Out shape: (batch_size, 512)

            loss, accuracy = moco_loss(Z_q, Z_k,
                                       self.queue,
                                       self.kmeans,
                                       self.kmeans_temp,
                                       self.config,
                                       self.enable_proto_nce)

        trainable_params = self.encoder_q.trainable_weights
        trainable_params += self.mlp.trainable_weights

        grads = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(grads, trainable_params))

        update_model_weights_with_ema(self.encoder_q, self.encoder_k)

        return { 'loss': loss, 'accuracy': accuracy, 'keys': Z_k }

    def test_step(self, data):
        X, _ = data # Discard labels provided by the dataset generator
        
        X_1_aug, X_2_aug = X

        Z_q = self.encoder_q(X_1_aug, training=False)
        Z_k = self.encoder_k(X_2_aug, training=False)

        Z_q = self.mlp(Z_q, training=False)
        Z_k = self.mlp(Z_k, training=False)

        loss, accuracy = moco_loss(Z_q, Z_k,
                                   self.queue,
                                   self.kmeans,
                                   self.kmeans_temp,
                                   self.config,
                                   self.enable_proto_nce)

        return { 'loss': loss, 'accuracy': accuracy }


@tf.function
def info_nce_loss(anchor, pos, neg, temp):
    # anchor: (B, C), pos: (B, C), neg: (K, C)

    batch_size = tf.shape(anchor)[0]

    # Determine loss
    l_pos = tf.einsum('nc,nc->n', anchor, pos) # Shape: (B)
    l_pos = tf.expand_dims(l_pos, axis=-1)  # Shape: (B, 1)
    l_neg = tf.einsum('nc,ck->nk', anchor, tf.transpose(neg)) # Shape: (B, queue_size)
    logits = tf.concat((l_pos, l_neg), axis=1) # Shape: (B, 1+queue_size)
    logits /= temp

    labels = tf.zeros(batch_size, dtype=tf.int32)
    loss = sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss = tf.reduce_mean(loss)

    # Determine accuracy
    logits_size = tf.shape(logits)[1]
    logits_softmax = tf.nn.softmax(logits, axis=0)
    pred_indices = tf.math.argmax(logits_softmax, axis=0, output_type=tf.int32)
    preds_acc = tf.math.equal(pred_indices, tf.zeros(logits_size, dtype=tf.int32))
    accuracy = tf.math.count_nonzero(preds_acc, dtype=tf.int32)
    accuracy /= logits_size

    return loss, accuracy


def compute_kmeans_temp(data, kmeans):
    _, I = kmeans.index.search(data, 1)

    res = []
    for cluster in range(kmeans.centroids.shape[0]):
        count = tf.math.count_nonzero(I == cluster)
        count = tf.cast(count, "float")
        Z_cluster = data[I == cluster]

        num = Z_cluster - kmeans.centroids[cluster]
        num = np.linalg.norm(num, axis=-1).sum()
        denom = count * np.log(count + 1e-6)
        res.append(num / denom)

    return np.array(res)


@tf.function
def proto_nce_loss(Z_q, kmeans, kmeans_temp, config):
    nb_negs = config['clustering_negs_count']
    batch_size = tf.shape(Z_q)[0]

    # Sample positive prototypes
    D, I = kmeans.index.search(Z_q, 1)
    I = I.flatten()
    pos = kmeans.centroids[I]

    l_pos = tf.einsum('nc,nc->n', Z_q, pos)
    l_pos = tf.expand_dims(l_pos, axis=-1)
    l_pos /= kmeans_temp[I]

    l_neg = []
    for i in range(batch_size):
        cluster_id = I[i]

        # Sample negative prototypes
        allowed_idx = np.arange(kmeans.centroids.shape[0])
        allowed_idx = np.delete(allowed_idx, np.where(allowed_idx == cluster_id))
        neg_idx = np.random.choice(allowed_idx, size=(nb_negs))
        neg = kmeans.centroids[neg_idx]

        l_neg_ = tf.einsum('c,ck->k', Z_q[i], tf.transpose(neg))
        l_neg_ /= kmeans_temp[neg_idx]
        l_neg.append(l_neg_)

    l_neg = tf.stack(l_neg)
    logits = tf.concat((l_pos, l_neg), axis=1)

    labels = tf.zeros(batch_size, dtype=tf.int32)
    loss = sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss = tf.reduce_mean(loss)

    return loss


@tf.function
def moco_loss(Z_q, Z_k, queue, kmeans, kmeans_temp, config, enable_proto_nce=False):
    loss, accuracy = info_nce_loss(Z_q, tf.stop_gradient(Z_k), queue, config['info_nce_temp'])

    if enable_proto_nce:
        loss += config['proto_nce_loss_factor'] \
                * proto_nce_loss(Z_q, kmeans, kmeans_temp, config)

    return loss, accuracy


def update_model_weights_with_ema(encoder_q, encoder_k, momentum=0.999):
    for v1, v2 in zip(encoder_q.variables, encoder_k.variables):
        v2.assign(momentum * v2 + (1.0 - momentum) * v1)


class MLP(Model):

    def __init__(self, dim):
        super().__init__()

        self.fc1 = Dense(512)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.fc2 = Dense(dim)

    def call(self, X):
        Z = self.fc1(X)
        Z = self.bn1(Z)
        Z = self.relu(Z)
        Z = self.fc2(Z)
        Z = tf.math.l2_normalize(Z, axis=-1)
        return Z


class MoCoUpdateCallback(Callback):
    def __init__(self, train_gen):
        super().__init__()

        self.train_gen = train_gen

    def on_batch_end(self, data, logs=None):
        keys = logs.pop('keys')

        self.model.queue = tf.concat([keys, self.model.queue], axis=0)
        self.model.queue = self.model.queue[:self.model.config['queue_size']]

    def on_epoch_end(self, epoch, logs=None):
        if epoch <= self.model.config['epochs_before_proto_nce']:
            return

        self.model.enable_proto_nce = True
        
        data = np.concatenate([self.encoder_q(self.mlp(X)) for X in self.train_gen])
        self.model.kmeans.train(X)

        self.model.kmeans_temp = (
            compute_kmeans_temp(data, self.model.kmeans)
        )