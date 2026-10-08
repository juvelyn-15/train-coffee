import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Rescaling,
    SpatialDropout2D,
)
from tensorflow.keras.metrics import BinaryAccuracy, F1Score, Precision, Recall
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

from data_loader import load_split
from experiments import EXPERIMENTS, TRAIN_COUNTS

MODEL_NAME = 'complex_cnn'
EXPERIMENT = os.environ.get('EXPERIMENT', 'E0')
SEED = int(os.environ.get('SEED', '42'))
SAMPLING = os.environ.get('SAMPLING', 'none')
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
BN_MOMENTUM = 0.9
BLOCK_DROPOUT = 0.0
HEAD_DROPOUT = 0.5
THRESHOLD = 0.5
PATIENCE = 8
USE_AUGMENTATION, class_weight = EXPERIMENTS[EXPERIMENT]

if SAMPLING not in ('none', 'under', 'over'):
    raise ValueError(f'Unknown SAMPLING: {SAMPLING}')
if SAMPLING != 'none' and class_weight is not None:
    raise ValueError('SAMPLING under/over must be combined with E0 or E1, not class weights')

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'
RUN_NAME = EXPERIMENT if SAMPLING == 'none' else f'{EXPERIMENT}_{SAMPLING}'
RUN_DIR = ARTIFACTS_DIR / MODEL_NAME / RUN_NAME / f'seed_{SEED}'
if (RUN_DIR / 'run.json').exists():
    raise ValueError(f'Completed run already exists: {RUN_DIR}')
RUN_DIR.mkdir(parents=True, exist_ok=True)

x_train, y_train, _ = load_split('train')
x_val, y_val, validation_metadata = load_split('val')

if USE_AUGMENTATION:
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode='nearest',
    )
else:
    datagen = ImageDataGenerator()


class SampledBatches(tf.keras.utils.PyDataset):
    def __init__(self, x, y, generator, mode, batch_size, seed):
        super().__init__()
        self.x = x
        self.y = y
        self.generator = generator
        self.mode = mode
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        labels = y.ravel().astype(int)
        self.normal = np.flatnonzero(labels == 0)
        self.defect = np.flatnonzero(labels == 1)
        self.indices = None
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.mode == 'under':
            normal = self.rng.choice(self.normal, len(self.defect), replace=False)
            indices = np.concatenate([normal, self.defect])
        else:
            defect = self.rng.choice(self.defect, len(self.normal), replace=True)
            indices = np.concatenate([self.normal, defect])
        self.indices = self.rng.permutation(indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        batch = np.sort(self.indices[start:start + self.batch_size])
        images = np.stack([self.generator.random_transform(np.array(self.x[i])) for i in batch])
        return images, self.y[batch]


if SAMPLING == 'none':
    train_generator = datagen.flow(
        x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
    )
else:
    train_generator = SampledBatches(x_train, y_train, datagen, SAMPLING, BATCH_SIZE, SEED)
val_generator = ImageDataGenerator().flow(
    x_val, y_val, batch_size=BATCH_SIZE, shuffle=False
)


def vgg_block(num_convs, num_filters):
    block = Sequential()
    for _ in range(num_convs):
        block.add(Conv2D(num_filters, (3, 3), padding='same', use_bias=False,
                         kernel_regularizer=l2(WEIGHT_DECAY)))
        block.add(BatchNormalization(momentum=BN_MOMENTUM))
        block.add(Activation('relu'))
    block.add(MaxPooling2D((2, 2)))
    if BLOCK_DROPOUT > 0:
        block.add(SpatialDropout2D(BLOCK_DROPOUT))
    return block
#2 gpu
strategy = tf.distribute.MirroredStrategy()
print("Number of GPUs:", strategy.num_replicas_in_sync)
with strategy.scope():
    model = Sequential()
    model.add(Input(shape=x_train.shape[1:]))
    model.add(Rescaling(1.0 / 255))
    conv_arch = ((2, 32), (2, 64), (2, 128), (2, 256))
    for num_convs, num_filters in conv_arch:
        model.add(vgg_block(num_convs, num_filters))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(Dropout(HEAD_DROPOUT))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        jit_compile=False,
        metrics=[BinaryAccuracy(name='accuracy', threshold=THRESHOLD),
                Precision(name='precision_defect', thresholds=THRESHOLD),
                Recall(name='recall_defect', thresholds=THRESHOLD),
                F1Score(name='f1_defect', threshold=THRESHOLD, average='micro')],
    )
checkpoint = ModelCheckpoint(
    RUN_DIR / 'best.keras', monitor='val_f1_defect', mode='max', save_best_only=True
)
early_stopping = EarlyStopping(
    monitor='val_f1_defect', mode='max', patience=PATIENCE
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', mode='min', factor=0.5, patience=3, min_lr=1e-6
)
start_time = time.monotonic()

H = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=2,
)
history = pd.DataFrame(H.history)
history.insert(0, 'phase', 'training')

training_seconds = time.monotonic() - start_time
history.index.name = 'epoch'
history.to_csv(RUN_DIR / 'history.csv')
model = load_model(RUN_DIR / 'best.keras', compile=False)
model.jit_compile = False
model.save(RUN_DIR / 'best.keras')
probabilities = model.predict(val_generator, verbose=0).ravel()
validation_predictions = validation_metadata.copy()
validation_predictions['p_defect'] = probabilities
validation_predictions.to_csv(RUN_DIR / 'validation_predictions.csv', index=False)
run = {
    'model': MODEL_NAME, 'experiment': EXPERIMENT, 'seed': SEED,
    'sampling': SAMPLING,
    'input_shape': list(x_train.shape[1:]), 'input_range': [0, 255],
    'threshold': THRESHOLD, 'batch_size': BATCH_SIZE,
    'augmentation': USE_AUGMENTATION, 'class_weights': class_weight,
    'train_counts': TRAIN_COUNTS,
    'best_epoch': int(history['val_f1_defect'].to_numpy().argmax()) + 1,
    'best_phase': str(history.iloc[history['val_f1_defect'].to_numpy().argmax()]['phase']),
    'best_validation_f1_defect': float(history['val_f1_defect'].max()), 'epochs_completed': len(history),
    'training_seconds': training_seconds, 'parameters': model.count_params(),
    'pretrained': False, 'optimizer': 'Adam',
    'learning_rate': LEARNING_RATE, 'max_epochs': EPOCHS,
    'weight_decay': WEIGHT_DECAY, 'bn_momentum': BN_MOMENTUM,
    'block_dropout': BLOCK_DROPOUT, 'head_dropout': HEAD_DROPOUT,
    'lr_schedule': 'ReduceLROnPlateau(val_loss, factor=0.5, patience=3)',
    'head_max_epochs': None, 'fine_tune_max_epochs': None,
    'fine_tune_learning_rate': None, 'fine_tune_stage': None,
    'early_stopping_patience': PATIENCE, 'selection_metric': 'val_f1_defect',
    'tensorflow_version': tf.__version__, 'keras_version': tf.keras.__version__,
}
(RUN_DIR / 'run.json').write_text(json.dumps(run, indent=2))