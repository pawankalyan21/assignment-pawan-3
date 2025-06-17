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

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📂 Overview
This notebook includes two major sections:
1. **Character-Level Text Generation** using TensorFlow and LSTM networks.
2. **Basic NLP Preprocessing** using NLTK.

---

## 1️⃣ Character-Level Text Generation (TensorFlow)
**Objective**: Generate Shakespeare-like text using a recurrent neural network.

### Key Steps:
- **Dataset**: Loads Shakespeare text corpus from TensorFlow's dataset utility.
- **Preprocessing**: Text is tokenized into characters, then converted into sequences for training.
- **Model**:
  - Embedding Layer
  - LSTM Layer
  - Dense Output Layer
- **Training**: Trained for 5 epochs using `SparseCategoricalCrossentropy` as the loss.
- **Generation**: New text is generated based on a given start string using a temperature-controlled sampling method.

### Libraries Used:
- `tensorflow`
- `numpy`
- `time`

---

## 2️⃣ Basic NLP Preprocessing (NLTK)
**Objective**: Demonstrate basic preprocessing steps on a given sentence.

### Key Steps:
- **Tokenization** using `word_tokenize`
- **Stopword and Punctuation Removal**
- **Stemming** using `PorterStemmer`

### Example Sentence:
> "NLP techniques are used in virtual assistants like Alexa and Siri."

### Libraries Used:
- `nltk`
- `string`

---

## ✅ Requirements
Make sure to install or import the following packages:
```bash
pip install tensorflow nltk
```

Also download the necessary NLTK corpora:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📂 Overview
This notebook includes two major sections:
1. **Character-Level Text Generation** using TensorFlow and LSTM networks.
2. **Basic NLP Preprocessing** using NLTK.

---

## 1️⃣ Character-Level Text Generation (TensorFlow)
**Objective**: Generate Shakespeare-like text using a recurrent neural network.

### Key Steps:
- **Dataset**: Loads Shakespeare text corpus from TensorFlow's dataset utility.
- **Preprocessing**: Text is tokenized into characters, then converted into sequences for training.
- **Model**:
  - Embedding Layer
  - LSTM Layer
  - Dense Output Layer
- **Training**: Trained for 5 epochs using `SparseCategoricalCrossentropy` as the loss.
- **Generation**: New text is generated based on a given start string using a temperature-controlled sampling method.

### Libraries Used:
- `tensorflow`
- `numpy`
- `time`

---

## 2️⃣ Basic NLP Preprocessing (NLTK)
**Objective**: Demonstrate basic preprocessing steps on a given sentence.

### Key Steps:
- **Tokenization** using `word_tokenize`
- **Stopword and Punctuation Removal**
- **Stemming** using `PorterStemmer`

### Example Sentence:
> "NLP techniques are used in virtual assistants like Alexa and Siri."

### Libraries Used:
- `nltk`
- `string`

---

## ✅ Requirements
Make sure to install or import the following packages:
```bash
pip install tensorflow nltk
-----------------------------------------------------------------------------------------------------------------
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d = K.shape[1]  # Key dimension
    scores = np.dot(Q, K.T)                      # Step 1: Dot Product
    scaled_scores = scores / np.sqrt(d)          # Step 2: Scale
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=1, keepdims=True))  # Stability
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)         # Step 3: Softmax
    output = np.dot(attention_weights, V)        # Step 4: Multiply by V
    return attention_weights, output

# Example input
Q = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

attention_weights, output = scaled_dot_product_attention(Q, K, V)
print("Attention Weights:\n", attention_weights)
print("Output:\n", output)
---------------------------------------------------------------------------------------------------------------------------
from transformers import pipeline

# Load pre-trained sentiment-analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Input sentence
sentence = "Despite the high price, the performance of the new MacBook is outstanding."

# Get prediction
result = sentiment_analyzer(sentence)[0]

# Print output
print(f"Sentiment: {result['label']}")
print(f"Confidence Score: {result['score']:.4f}")
