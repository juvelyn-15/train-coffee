# 1. Import libraries
import json
import os
import time
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.metrics import BinaryAccuracy, F1Score, Precision, Recall
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_loader import load_split
from experiments import EXPERIMENTS, TRAIN_COUNTS

# 2. Training settings
MODEL_NAME = 'efficientnet_b4'
EXPERIMENT = os.environ.get('EXPERIMENT', 'E0')
SEED = int(os.environ.get('SEED', '42'))
BATCH_SIZE = 16
LEARNING_RATE = 0.001
THRESHOLD = 0.5
USE_AUGMENTATION, class_weight = EXPERIMENTS[EXPERIMENT]

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
# Avoid the failing NHWC/NCHW layout rewrite of stochastic-depth dropout.
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

ARTIFACTS_DIR = Path(__file__).resolve().parent / 'artifacts'
RUN_DIR = ARTIFACTS_DIR / MODEL_NAME / EXPERIMENT / f'seed_{SEED}'
if (RUN_DIR / 'run.json').exists():
    raise ValueError(f'Completed run already exists: {RUN_DIR}')
RUN_DIR.mkdir(parents=True, exist_ok=True)

# 3. Load data
x_train, y_train, _ = load_split('train')
x_val, y_val, validation_metadata = load_split('val')
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

# 5. Build model with ImageNet pretrained EfficientNetB4

# EfficientNet already includes its ImageNet input preprocessing.
base_network = EfficientNetB4(
    weights='imagenet', include_top=False, input_shape=(256, 256, 3)
)
base_network.trainable = False
model = Sequential([
    Input(shape=(256, 256, 3)), base_network,
    GlobalAveragePooling2D(), Dense(128, activation='relu'), Dropout(0.5),
    Dense(1, activation='sigmoid'),
])

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

# 7. Train the new head, then fine-tune the last backbone stage.
H = model.fit(
    train_generator, validation_data=val_generator,
    epochs=10,
    class_weight=class_weight, callbacks=[checkpoint], verbose=2,
)
history_head = pd.DataFrame(H.history)
history_head.insert(0, 'phase', 'head')

base_network.trainable = True
for layer in base_network.layers:
    layer.trainable = layer.name.startswith('block7')
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

# Changing trainable layers requires recompilation.
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    jit_compile=False,
    metrics=[BinaryAccuracy(name='accuracy', threshold=THRESHOLD),
             Precision(name='precision_defect', thresholds=THRESHOLD),
             Recall(name='recall_defect', thresholds=THRESHOLD),
             F1Score(name='f1_defect', threshold=THRESHOLD, average='micro')],
)
H_finetune = model.fit(
    train_generator, validation_data=val_generator,
    epochs=40,
    class_weight=class_weight, callbacks=[checkpoint, early_stopping], verbose=2,
)
history_finetune = pd.DataFrame(H_finetune.history)
history_finetune.insert(0, 'phase', 'fine_tune')
history = pd.concat([history_head, history_finetune], ignore_index=True)

# 8. Save the best model and validation results; test is evaluated separately.
training_seconds = time.monotonic() - start_time
history.index.name = 'epoch'
history.to_csv(RUN_DIR / 'history.csv')
model = load_model(RUN_DIR / 'best.keras', compile=False)
model.jit_compile = False
# Retain the selected weights and preprocessing without unused optimizer slots.
model.save(RUN_DIR / 'best.keras')
probabilities = model.predict(val_generator, verbose=0).ravel()
validation_predictions = validation_metadata.copy()
validation_predictions['p_defect'] = probabilities
validation_predictions.to_csv(RUN_DIR / 'validation_predictions.csv', index=False)
run = {
    'model': MODEL_NAME, 'experiment': EXPERIMENT, 'seed': SEED,
    'input_shape': [256, 256, 3], 'input_range': [0, 255],
    'threshold': THRESHOLD, 'batch_size': BATCH_SIZE,
    'augmentation': USE_AUGMENTATION, 'class_weights': class_weight,
    'train_counts': TRAIN_COUNTS,
    'best_epoch': int(history['val_f1_defect'].to_numpy().argmax()) + 1,
    'best_phase': str(history.iloc[history['val_f1_defect'].to_numpy().argmax()]['phase']),
    'best_validation_f1_defect': float(history['val_f1_defect'].max()), 'epochs_completed': len(history),
    'training_seconds': training_seconds, 'parameters': model.count_params(),
    'pretrained': True, 'optimizer': 'Adam',
    'learning_rate': LEARNING_RATE, 'max_epochs': 50,
    'head_max_epochs': 10, 'fine_tune_max_epochs': 40,
    'fine_tune_learning_rate': 1e-5, 'fine_tune_stage': 'block7',
    'early_stopping_patience': 8, 'selection_metric': 'val_f1_defect',
    'fine_tune_layers': [layer.name for layer in base_network.layers if layer.trainable],
    'tensorflow_version': tf.__version__, 'keras_version': tf.keras.__version__,
}
(RUN_DIR / 'run.json').write_text(json.dumps(run, indent=2))
print('Saved:', RUN_DIR)
