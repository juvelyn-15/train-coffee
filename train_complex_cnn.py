# 1. Import libraries
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
)
from tensorflow.keras.metrics import BinaryAccuracy, F1Score, Precision, Recall
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessing import DataPreprocessor

# 2. Training settings
MODEL_NAME = 'complex_cnn'
EXPERIMENT = os.environ.get('EXPERIMENT', 'E0')
SEED = int(os.environ.get('SEED', '42'))
SMOKE_TEST = os.environ.get('SMOKE_TEST', '0') == '1'
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
THRESHOLD = 0.5
USE_AUGMENTATION = EXPERIMENT in ('E1', 'E3')
USE_CLASS_WEIGHT = EXPERIMENT in ('E2', 'E3')
if EXPERIMENT not in ('E0', 'E1', 'E2', 'E3'):
    raise ValueError('EXPERIMENT must be E0, E1, E2 or E3')

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'
if SMOKE_TEST:
    ARTIFACTS_DIR = ARTIFACTS_DIR / 'smoke'
RUN_DIR = ARTIFACTS_DIR / MODEL_NAME / EXPERIMENT / f'seed_{SEED}'
if (RUN_DIR / 'run.json').exists():
    raise ValueError(f'Completed run already exists: {RUN_DIR}')
RUN_DIR.mkdir(parents=True, exist_ok=True)

# 3. Load data
preprocessor = DataPreprocessor()
file_table = preprocessor.build_file_table()
if SMOKE_TEST:
    file_table = file_table.groupby(['split', 'class_name'], sort=False).head(8)
x_train, y_train = preprocessor.load_split(file_table, 'train', memory_map=True)
x_val, y_val = preprocessor.load_split(file_table, 'val', memory_map=True)
print('Train:', x_train.shape, y_train.shape)
print('Validation:', x_val.shape, y_val.shape)

# 4. Data augmentation and class weights
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

train_generator = datagen.flow(
    x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
)
# The validation generator changes no pixels and includes the final batch.
val_generator = ImageDataGenerator().flow(
    x_val, y_val, batch_size=BATCH_SIZE, shuffle=False
)
counts = np.bincount(y_train.astype(int).ravel(), minlength=2)
class_weight = None
if USE_CLASS_WEIGHT:
    class_weight = {label: len(y_train) / (2 * int(count))
                    for label, count in enumerate(counts)}

# 5. Build model: VGG blocks, following Chapter 5.2, slides 6 and 8.
def vgg_block(num_convs, num_filters):
    block = Sequential()
    for _ in range(num_convs):
        block.add(Conv2D(num_filters, (3, 3), padding='same'))
        block.add(BatchNormalization())
        block.add(Activation('relu'))
    block.add(MaxPooling2D((2, 2)))
    block.add(Dropout(0.25))
    return block


model = Sequential()
model.add(Input(shape=(256, 256, 3)))
model.add(Rescaling(1.0 / 255))
conv_arch = ((2, 32), (2, 64), (2, 128), (2, 256))
for num_convs, num_filters in conv_arch:
    model.add(vgg_block(num_convs, num_filters))
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 6. Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    jit_compile=False,
    metrics=[BinaryAccuracy(name='accuracy', threshold=THRESHOLD),
             Precision(name='precision_defect', thresholds=THRESHOLD),
             Recall(name='recall_defect', thresholds=THRESHOLD),
             F1Score(name='f1_defect', threshold=THRESHOLD, average='micro')],
)
model.summary()
checkpoint = ModelCheckpoint(
    RUN_DIR / 'best.keras', monitor='val_f1_defect', mode='max', save_best_only=True
)
early_stopping = EarlyStopping(
    monitor='val_f1_defect', mode='max', patience=8
)
start_time = time.monotonic()

# 7. Train model
H = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=1 if SMOKE_TEST else EPOCHS,
    class_weight=class_weight,
    callbacks=[checkpoint, early_stopping],
    verbose=2,
)
history = pd.DataFrame(H.history)
history.insert(0, 'phase', 'training')

# 8. Save the best model and validation results; test is evaluated separately.
training_seconds = time.monotonic() - start_time
history.index.name = 'epoch'
history.to_csv(RUN_DIR / 'history.csv')
model = load_model(RUN_DIR / 'best.keras', compile=False)
model.jit_compile = False
# Retain the selected weights and preprocessing without unused optimizer slots.
model.save(RUN_DIR / 'best.keras')
probabilities = model.predict(val_generator, verbose=0).ravel()
validation_predictions = file_table[file_table.split == 'val'][['relative_path', 'class_name', 'label']].copy()
validation_predictions['p_defect'] = probabilities
validation_predictions.to_csv(RUN_DIR / 'validation_predictions.csv', index=False)
run = {
    'model': MODEL_NAME, 'experiment': EXPERIMENT, 'seed': SEED,
    'smoke_test': SMOKE_TEST,
    'input_shape': [256, 256, 3], 'input_range': [0, 255],
    'threshold': THRESHOLD, 'batch_size': BATCH_SIZE,
    'augmentation': USE_AUGMENTATION, 'class_weights': class_weight,
    'train_counts': counts.tolist(),
    'best_epoch': int(history['val_f1_defect'].to_numpy().argmax()) + 1,
    'best_phase': str(history.iloc[history['val_f1_defect'].to_numpy().argmax()]['phase']),
    'best_validation_f1_defect': float(history['val_f1_defect'].max()), 'epochs_completed': len(history),
    'training_seconds': training_seconds, 'parameters': model.count_params(),
    'pretrained': False, 'optimizer': 'Adam',
    'learning_rate': LEARNING_RATE, 'max_epochs': 50,
    'head_max_epochs': None, 'fine_tune_max_epochs': None,
    'fine_tune_learning_rate': None, 'fine_tune_stage': None,
    'early_stopping_patience': 8, 'selection_metric': 'val_f1_defect',
    'tensorflow_version': tf.__version__, 'keras_version': tf.keras.__version__,
}
(RUN_DIR / 'run.json').write_text(json.dumps(run, indent=2))
print('Saved:', RUN_DIR)
