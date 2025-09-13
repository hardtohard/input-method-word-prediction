from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR/'data' / 'raw'
LOGS_DIR = ROOT_DIR /'logs'
PROCESSED_DIR = ROOT_DIR / 'data'/'processed'
MODULES_DIR = ROOT_DIR / 'models'


SEQ_LEN=5

BATCH_SIZE=128

EMBEDDING_DIM=128
HIDDEN_SIZE=256
