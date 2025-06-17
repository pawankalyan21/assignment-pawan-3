# assignment-pawan-3
# Jagatti Pawan Kalyan
# 700776779
import tensorflow as tf
import numpy as np
import time

print("✅ TensorFlow version:", tf.__version__)
print("🔍 GPU Available:", tf.config.list_physical_devices('GPU'))

# 1. Load dataset
path_to_file = tf.keras.utils.get_file("shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(path_to_file, 'rb').read().decode('utf-8')
vocab = sorted(set(text))
print(f"📄 Loaded text with {len(text)} characters and {len(vocab)} unique characters.")

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# 2. Create dataset
seq_length = 50
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)
BATCH_SIZE = 32
BUFFER_SIZE = 1000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# 3. Define smaller model
vocab_size = len(vocab)
embedding_dim = 128
rnn_units = 256

def build_model(vocab_size, embedding_dim, rnn_units):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True),
        tf.keras.layers.Dense(vocab_size)
    ])

model = build_model(vocab_size, embedding_dim, rnn_units)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# 4. Train quickly
EPOCHS = 5
print("\n🚀 Training for 1 epoch (fast mode)...")
start = time.time()
model.fit(dataset, epochs=EPOCHS)
print(f"✅ Done training in {time.time() - start:.2f} seconds.")

# 5. Text generation
def generate_text(model, start_string, temperature=1.0, num_generate=300):
    input_eval = tf.expand_dims([char2idx[s] for s in start_string], 0)
    text_generated = []

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :] / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

print("\n📝 Sample text:")
print(generate_text(model, start_string="ROMEO: "))
