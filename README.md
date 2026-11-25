# IMDB Movie Review Sentiment Analysis using Simple RNN

An end-to-end natural language processing project that classifies an IMDB movie review as positive or negative using a Simple Recurrent Neural Network (Simple RNN). The project covers text-to-index preprocessing, word embeddings, recurrent neural network training, early stopping, model serialization, notebook-based inference, and deployment with a Streamlit interface.

## Project Overview

Movie review platforms generate large volumes of unstructured text. This project converts raw review text into numerical sequences and trains a neural network to identify the sentiment expressed in a review.

The binary classification task uses the built-in **IMDB Movies Reviews** dataset:

- `0` — Negative review
- `1` — Positive review

### Resume Project Overview

Developed an end-to-end movie review sentiment classification system using a TensorFlow/Keras Simple RNN. Built a text preprocessing pipeline with a 10,000-word vocabulary, word-index encoding, 500-token sequence padding, and a 128-dimensional embedding layer. Trained a recurrent neural network with Adam optimization, binary cross-entropy loss, and early stopping, achieving approximately 82.06% validation accuracy. Serialized the trained model and developed notebook-based and Streamlit inference workflows for real-time positive/negative sentiment prediction.

## Key Features

- Loads and splits the 50,000-review IMDB dataset
- Limits the model vocabulary to the top 10,000 words
- Maps raw text to integer word indices
- Decodes integer sequences back into readable words
- Pads all reviews to a fixed length of 500 tokens
- Uses a trainable word embedding layer
- Builds a Simple RNN binary classification model
- Uses early stopping to restore the best validation weights
- Saves the trained model as an `.h5` artifact
- Provides sample notebook inference
- Provides an interactive Streamlit prediction interface
- Includes a separate notebook demonstrating one-hot encoding and word embeddings

## Dataset

**Source:** TensorFlow/Keras `imdb` dataset

- 25,000 training reviews
- 25,000 test reviews
- Maximum vocabulary size: 10,000 words
- Reviews are supplied as pre-tokenized integer sequences
- Labels are binary sentiment values

The dataset is loaded through:

```python
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
```

## NLP and Model Pipeline

### 1. Data Loading

The project loads the IMDB dataset and limits the vocabulary to 10,000 frequently used words. The first training review and label are inspected to verify the dataset structure.

### 2. Text-to-Integer Representation

The TensorFlow dataset stores each review as a sequence of integer word indices. The code retrieves the IMDB word index and reverses it to decode numeric sequences into readable text.

### 3. Sequence Padding

Review lengths vary, so all sequences are padded or truncated to a fixed length of 500 tokens:

```python
max_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
```

### 4. Word Embedding

The model uses an embedding layer that converts each vocabulary index into a 128-dimensional dense vector:

```python
model.add(Embedding(max_features, 128, input_length=max_length))
```

The embedding learns a distributed representation for each word during model training.

### 5. Simple RNN Architecture

| Layer | Configuration | Purpose |
|---|---|---|
| Embedding | 10,000 × 128 | Converts word indices into 128-dimensional vectors |
| SimpleRNN | 128 units, ReLU | Processes the review sequence and captures temporal patterns |
| Dense | 1 unit, Sigmoid | Outputs a probability between 0 and 1 |

The model was compiled using:

- Optimizer: Adam
- Loss: Binary cross-entropy
- Metric: Accuracy
- Batch size: 32
- Epochs: 10
- Early stopping patience: 5 epochs

The Simple RNN model contains:

- 1,280,000 embedding parameters
- 32,896 SimpleRNN parameters
- 129 Dense-layer parameters
- 1,313,025 trainable parameters

### 6. Model Training

Training used 20% of the training data for validation:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[earlyStopping]
)
```

Early stopping monitored validation loss and restored the best model weights.

### Recorded Training Results

The available training output showed:

| Metric | Recorded Value |
|---|---:|
| Peak validation accuracy | Approximately 82.06% |
| Highest recorded training accuracy | Approximately 91.68% |
| Peak validation loss observed | 0.4080 by epoch 2 after initial training |

The gap between later training and validation accuracy suggests possible overfitting. The notebook does not report a final held-out test-set evaluation, so the validation accuracy is the most reliable recorded performance metric in this repository.

## User Input Preprocessing

Raw review text is converted into model-ready input through the following steps:

1. Convert text to lowercase
2. Split it into individual words
3. Map each word to a Keras IMDB vocabulary index
4. Assign index `2` to unseen or unknown words
5. Add the vocabulary offset used by the IMDB representation
6. Pad the sequence to 500 tokens

```python
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    return sequence.pad_sequences([encoded_review], maxlen=500)
```

Unknown words are handled as out-of-vocabulary tokens, allowing the application to process text that was not encountered during training.

## Sentiment Prediction

The model predicts a probability from 0 to 1:

- Score greater than `0.5` — Positive
- Score less than or equal to `0.5` — Negative

Example:

```python
sentiment, score = predict_sentiment(
    "This movie was fantastic!! The acting was great and the plot was thrilling."
)
```

## Streamlit Web Application

The `main.py` file provides a Streamlit interface where a user can:

- Enter a movie review
- Click a **Classify** button
- View the predicted sentiment
- View the model confidence score

Run the application from this project directory:

```bash
cd "Deep Learning for NLP/RNN Project Implimentation"
streamlit run main.py
```

The app will open at a local URL such as `http://localhost:8501`.

## Project Structure

```text
RNN Project Implimentation/
├── main.py                        # Streamlit sentiment prediction application
├── prediction.ipynb               # Loads the trained model and tests inference
├── simpleRNN.ipynb                # Dataset preparation and Simple RNN training
├── wordEmbedding.ipynb            # One-hot encoding and embedding demonstration
├── simple_rnn_imdb.h5             # Serialized trained Simple RNN model
└── README.md                      # Project documentation
```

## Technologies and Libraries

- Python
- TensorFlow/Keras
- NumPy
- Streamlit
- IMDB dataset API
- Keras sequence preprocessing
- Jupyter Notebook

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install tensorflow numpy streamlit
```

The workspace may already include these libraries through its existing requirements file.

## Reproduce the Training

Open and run `simpleRNN.ipynb`:

1. Load the IMDB dataset
2. Inspect and decode a sample review
3. Pad sequences to 500 tokens
4. Build the Embedding + SimpleRNN + Dense architecture
5. Compile and train the model
6. Apply early stopping
7. Save the trained model as `simple_rnn_imdb.h5`

## Test Notebook Predictions

Open `prediction.ipynb` to:

- Load `simple_rnn_imdb.h5`
- Load the IMDB word index
- Preprocess example review text
- Generate a sentiment label and probability

## Word Embedding Experiment

`wordEmbedding.ipynb` demonstrates foundational text representation concepts:

- Converting sentences to one-hot representations
- Padding variable-length sequences
- Creating a Keras embedding matrix
- Exploring how words are represented as dense vectors

This notebook supports the learning workflow but is separate from the final IMDB sentiment model.

## Results Summary

| Component | Implementation | Outcome |
|---|---|---|
| Text preprocessing | 10,000-word vocabulary and 500-token padding | Fixed-length numerical review representation |
| Feature representation | 128-dimensional trainable embedding | Dense word vectors for the RNN |
| Recurrent model | SimpleRNN with 128 units | Sequential review classification |
| Training | Adam, binary cross-entropy, early stopping | Stable and serialized binary classifier |
| Validation performance | 20% validation split | Approximately 82.06% peak validation accuracy |
| Deployment | Streamlit and notebook inference | Real-time and interactive sentiment prediction |

## Skills Demonstrated

- Natural language preprocessing
- Text tokenization and sequence representation
- Word embeddings
- Recurrent neural networks
- Binary text classification
- TensorFlow/Keras model development
- Early stopping and validation monitoring
- Model serialization and loading
- Streamlit application development
- Machine learning inference pipelines

## Limitations and Future Improvements

- Evaluate the model on the separate 25,000-review IMDB test set
- Add accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix
- Compare Simple RNN with LSTM and GRU architectures
- Experiment with pretrained embeddings such as GloVe or Word2Vec
- Add bidirectional recurrent layers
- Tune sequence length, embedding size, dropout, learning rate, and batch size
- Add stemming or lemmatization as an optional preprocessing step
- Improve punctuation, capitalization, and HTML-token handling
- Handle unknown words more explicitly
- Add probability calibration and confidence thresholds
- Add automated tests for preprocessing and model inference
- Improve the Streamlit interface with examples, loading states, and input validation

## Author

This project was developed as part of a hands-on Deep Learning for NLP learning portfolio.
