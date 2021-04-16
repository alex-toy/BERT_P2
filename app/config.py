import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


INPUTS_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
INPUTS_FILE = os.path.join(INPUTS_FILE_PATH, 'training.1600000.processed.noemoticon.csv')


COLS = ["sentiment", "id", "date", "query", "user", "text"]
COLS_TO_KEEP = ["sentiment", "text"]


OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
OUTPUTS_FILE = os.path.join(OUTPUTS_DIR, 'outputs.csv')

CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../checkpoint'))

OUTPUTS_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))


EMB_DIM = 200

NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = 2

DROPOUT_RATE = 0.2

BATCH_SIZE = 32
NB_EPOCHS = 5