# config.py

# Data paths
TRAIN_PICKLE_PATH = "./prepared_data/train_data.pkl"
TEST_PICKLE_PATH = "./prepared_data/test_data.pkl"
CHECKPOINT_DIR = "checkpoints"
IMAGE_SAVE_PATH = "images/loss_plot.png"

# Model parameters
WINDOW_SIZE = 30
LATENT_DIM = 2
SELECTED_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
    'sensor_11', 'sensor_12', 'sensor_15'
]

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10

# Visualization parameters
MAX_TIME_FRACTION = 1.0
BASELINE_FRACTION = 0.1
SMOOTHING_WINDOW = None
NUM_PLOTS = 8
