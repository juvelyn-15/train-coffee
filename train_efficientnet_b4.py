import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.metrics import BinaryAccuracy, F1Score, Precision, Recall
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

from data_loader import load_split
from experiments import EXPERIMENTS, TRAIN_COUNTS

# Usage: python train_efficientnet_b4.py E0 42
#   or:  $env:EXPERIMENT="E0"; $env:SEED="42"; python train_efficientnet_b4.py
MODEL_NAME = 'efficientnet_b4'
if len(sys.argv) == 3:
    EXPERIMENT = sys.argv[1]
    SEED = int(sys.argv[2])
else:
    EXPERIMENT = os.environ.get('EXPERIMENT', 'E3')
    SEED = int(os.environ.get('SEED', '42'))
BATCH_SIZE = 16
IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5

# Phase 1: train head only
HEAD_EPOCHS = 10
HEAD_LEARNING_RATE = 1e-3
HEAD_WEIGHT_DECAY = 1e-4

# Phase 2: fine-tune block7
BLOCK7_EPOCHS = 10
BLOCK7_LEARNING_RATE = 5e-6

# Phase 3: fine-tune block6 + block7
BLOCK6_7_EPOCHS = 15
BLOCK6_7_LEARNING_RATE = 1e-6

FINE_TUNE_WEIGHT_DECAY = 1e-5
HEAD_DROPOUT = 0.4
L2_STRENGTH = 1e-4


if EXPERIMENT not in EXPERIMENTS:
    raise ValueError(f'Unknown EXPERIMENT={EXPERIMENT!r}; choose from {sorted(EXPERIMENTS)}')
USE_AUGMENTATION, CLASS_WEIGHT = EXPERIMENTS[EXPERIMENT]

print('Model:', MODEL_NAME, '| Experiment:', EXPERIMENT, '| Seed:', SEED)
print('Augmentation:', USE_AUGMENTATION, '| Class weights:', CLASS_WEIGHT)

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / 'artifacts' / MODEL_NAME / EXPERIMENT / f'seed_{SEED}'
if (RUN_DIR / 'run.json').exists():
    raise ValueError(f'Completed run already exists: {RUN_DIR}')
RUN_DIR.mkdir(parents=True, exist_ok=True)

x_train, y_train, _ = load_split('train')
x_val, y_val, validation_metadata = load_split('val')
print('Train:', x_train.shape, y_train.shape)
print('Validation:', x_val.shape, y_val.shape)

if USE_AUGMENTATION:
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode='nearest',
    )
else:
    train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
)
val_generator = ImageDataGenerator().flow(
    x_val, y_val, batch_size=BATCH_SIZE, shuffle=False
)

base_network = EfficientNetB4(
    weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3)
)
base_network.trainable = False

model = Sequential([
    Input(shape=(*IMAGE_SIZE, 3)),
    base_network,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(128, activation='relu', kernel_regularizer=l2(L2_STRENGTH)),
    Dropout(HEAD_DROPOUT),
    Dense(1, activation='sigmoid'),
])

metrics = [
    BinaryAccuracy(name='accuracy', threshold=THRESHOLD),
    Precision(name='precision_defect', thresholds=THRESHOLD),
    Recall(name='recall_defect', thresholds=THRESHOLD),
    F1Score(name='f1_defect', threshold=THRESHOLD, average='micro'),
]

checkpoint = ModelCheckpoint(
    RUN_DIR / 'best.keras', monitor='val_f1_defect', mode='max',
    save_best_only=True, verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_f1_defect', mode='max', factor=0.5, patience=3,
    min_lr=1e-7, verbose=1,
)

start_time = time.monotonic()
history_parts = []

print('PHASE 1: train head for', HEAD_EPOCHS, 'epochs')
model.compile(
    optimizer=AdamW(learning_rate=HEAD_LEARNING_RATE, weight_decay=HEAD_WEIGHT_DECAY),
    loss='binary_crossentropy', jit_compile=False, metrics=metrics,
)
h1 = model.fit(
    train_generator, validation_data=val_generator, epochs=HEAD_EPOCHS,
    class_weight=CLASS_WEIGHT, callbacks=[checkpoint, reduce_lr], verbose=2,
)
h1 = pd.DataFrame(h1.history)
h1.insert(0, 'phase', 'head')
history_parts.append(h1)

base_network.trainable = True
for layer in base_network.layers:
    layer.trainable = layer.name.startswith('block7')
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

block7_layers = [layer.name for layer in base_network.layers if layer.trainable]
print('PHASE 2: fine-tune block7 for', BLOCK7_EPOCHS, 'epochs')
print('Trainable backbone layers:', len(block7_layers))

model.compile(
    optimizer=AdamW(learning_rate=BLOCK7_LEARNING_RATE, weight_decay=FINE_TUNE_WEIGHT_DECAY),
    loss='binary_crossentropy', jit_compile=False, metrics=metrics,
)
h2 = model.fit(
    train_generator, validation_data=val_generator, epochs=BLOCK7_EPOCHS,
    class_weight=CLASS_WEIGHT, callbacks=[checkpoint, reduce_lr], verbose=2,
)
h2 = pd.DataFrame(h2.history)
h2.insert(0, 'phase', 'fine_tune_block7')
history_parts.append(h2)

for layer in base_network.layers:
    layer.trainable = layer.name.startswith(('block6', 'block7'))
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

block6_7_layers = [layer.name for layer in base_network.layers if layer.trainable]
print('PHASE 3: fine-tune block6+7 for', BLOCK6_7_EPOCHS, 'epochs')
print('Trainable backbone layers:', len(block6_7_layers))

early_stopping = EarlyStopping(
    monitor='val_f1_defect', mode='max', patience=8, verbose=1,
)
model.compile(
    optimizer=AdamW(learning_rate=BLOCK6_7_LEARNING_RATE, weight_decay=FINE_TUNE_WEIGHT_DECAY),
    loss='binary_crossentropy', jit_compile=False, metrics=metrics,
)
h3 = model.fit(
    train_generator, validation_data=val_generator, epochs=BLOCK6_7_EPOCHS,
    class_weight=CLASS_WEIGHT, callbacks=[checkpoint, reduce_lr, early_stopping], verbose=2,
)
h3 = pd.DataFrame(h3.history)
h3.insert(0, 'phase', 'fine_tune_block6_7')
history_parts.append(h3)


history = pd.concat(history_parts, ignore_index=True)
history.index.name = 'epoch'
history.to_csv(RUN_DIR / 'history.csv')

# Reload best model and save validation predictions
model = load_model(RUN_DIR / 'best.keras', compile=False)
model.jit_compile = False
model.save(RUN_DIR / 'best.keras')

probabilities = model.predict(val_generator, verbose=0).ravel()
validation_predictions = validation_metadata.copy()
validation_predictions['p_defect'] = probabilities
validation_predictions.to_csv(RUN_DIR / 'validation_predictions.csv', index=False)

best_index = int(history['val_f1_defect'].to_numpy().argmax())
run = {
    'model': MODEL_NAME, 'experiment': EXPERIMENT, 'seed': SEED,
    'input_shape': [*IMAGE_SIZE, 3], 'input_range': [0, 255],
    'threshold': THRESHOLD, 'batch_size': BATCH_SIZE,
    'augmentation': USE_AUGMENTATION, 'class_weights': CLASS_WEIGHT,
    'train_counts': TRAIN_COUNTS,
    'best_epoch': best_index + 1,
    'best_phase': str(history.iloc[best_index]['phase']),
    'best_validation_f1_defect': float(history['val_f1_defect'].max()),
    'epochs_completed': len(history),
    'training_seconds': time.monotonic() - start_time,
    'parameters': model.count_params(),
    'pretrained': True, 'optimizer': 'AdamW',
    'head_learning_rate': HEAD_LEARNING_RATE,
    'block7_learning_rate': BLOCK7_LEARNING_RATE,
    'block6_7_learning_rate': BLOCK6_7_LEARNING_RATE,
    'head_weight_decay': HEAD_WEIGHT_DECAY,
    'fine_tune_weight_decay': FINE_TUNE_WEIGHT_DECAY,
    'head_dropout': HEAD_DROPOUT, 'l2_strength': L2_STRENGTH,
    'head_max_epochs': HEAD_EPOCHS,
    'block7_max_epochs': BLOCK7_EPOCHS,
    'block6_7_max_epochs': BLOCK6_7_EPOCHS,
    'fine_tune_stage': 'block7_then_block6+block7',
    'block7_layers': block7_layers, 'block6_7_layers': block6_7_layers,
    'reduce_lr_factor': 0.5, 'reduce_lr_patience': 3,
    'early_stopping_patience': 8, 'selection_metric': 'val_f1_defect',
    'tensorflow_version': tf.__version__, 'keras_version': tf.keras.__version__,
}
(RUN_DIR / 'run.json').write_text(json.dumps(run, indent=2))
print('Saved:', RUN_DIR)
