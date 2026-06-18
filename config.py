"""Central configuration for the SpeakEasy audio pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
TEMPLATES_DIR = DATA_DIR / "templates"
MANIFESTS_DIR = DATA_DIR / "manifests"

LETTERS_DIR = RAW_DIR / "letters"
WORDS_DIR = RAW_DIR / "words"

# ── Audio format ───────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
BIT_DEPTH = 16
CHANNELS = 1
AUDIO_FORMAT = "wav"

# ── Preprocessing ──────────────────────────────────────────────────────────
PRE_EMPHASIS_COEF = 0.97
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
WINDOW = "hamming"

# VAD thresholds (energy + zero-crossing rate)
VAD_ENERGY_THRESHOLD = 0.01
VAD_ZCR_THRESHOLD = 0.15
VAD_FRAME_MS = 25
VAD_HOP_MS = 10

# ── MFCC ───────────────────────────────────────────────────────────────────
N_MFCC = 13
N_MEL_FILTERS = 26
N_FFT = 512
FMIN = 0.0
FMAX = SAMPLE_RATE / 2

# ── Data augmentation ──────────────────────────────────────────────────────
AUGMENT_SNR_DB = [10, 15, 20, 30]
AUGMENT_SPEED_RANGE = (0.9, 1.1)
AUGMENT_PITCH_SEMITONES = (-2, 2)
AUGMENT_VOLUME_DB = (-3, 3)
AUGMENT_FACTOR = 8  # target 8× expansion for better real-world robustness

# ── Dataset split ──────────────────────────────────────────────────────────
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_SEED = 42

# ── Labels ─────────────────────────────────────────────────────────────────
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

DEFAULT_WORDS = [
    "apple", "book", "cat", "dog", "egg", "fish", "goat", "hat", "ice", "jam",
    "key", "lion", "map", "net", "owl", "pen", "queen", "rat", "sun", "tree",
    "up", "van", "web", "box", "yes", "zip", "ball", "car", "door", "eye",
    "fan", "game", "house", "ink", "jump", "kite", "leg", "moon", "nose", "open",
    "pig", "run", "sit", "top", "use", "walk", "xray", "yard", "zoo", "bird",
    "cake", "duck", "frog", "girl", "hand", "milk", "park", "rain", "star", "water",
]
