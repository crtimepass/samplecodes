# 1. Classification using a Neural network

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess
df = pd.read_csv("breast_cancer.csv")
df.drop("id", axis=1, inplace=True)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
x, y = df.drop("diagnosis", axis=1), df["diagnosis"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation="relu", input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile and train
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(x_train, y_train, epochs=3)

# Predict with threshold 0.7
def predict_and_report(x, y, threshold=0.7):
    probs = model.predict(x)
    preds = (probs[:, 0] > threshold).astype(int)
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds))

# Train set report
predict_and_report(x_train, y_train)

# Test set report
predict_and_report(x_test, y_test)

"""# 2. Perform card classification using CNN"""

# Install necessary library to download dataset from KaggleHub
!pip install -q kagglehub

# Import libraries
import kagglehub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv

# Download dataset using KaggleHub (auto-downloads and extracts)
dataset_path = kagglehub.dataset_download('gpiosenka/cards-image-datasetclassification')
train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

# Check class names (card categories)
class_names = os.listdir(train_dir)
print("Classes:", class_names)

# Read and display one image
sample_img_path = os.path.join(train_dir, "ace of clubs", "001.jpg")
img = cv.imread(sample_img_path)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # Convert BGR (OpenCV format) to RGB
plt.title("Sample Image")
plt.axis("off")
plt.show()

# Optional: Remove images with invalid extensions
valid_exts = [".jpg", ".jpeg", ".png", ".bmp"]
for folder in os.listdir(train_dir):
    for filename in os.listdir(os.path.join(train_dir, folder)):
        ext = os.path.splitext(filename)[-1]
        if ext.lower() not in valid_exts:
            print("Invalid extension:", ext)

# Load training and validation datasets (90/10 split)
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.1,
    subset="both",
    seed=42,
    image_size=(32, 32),
    batch_size=32,
    label_mode="int"
)

# View a sample batch of images with labels
plt.figure(figsize=(10, 10))
for imgs, labels in train_ds.take(1):
    for i in range(32):
        plt.subplot(8, 4, i + 1)
        plt.imshow(imgs[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.show()

# Define CNN model with preprocessing and augmentation
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Rescaling(1./127.0, offset=-1),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1)

# Train the model
model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stop])

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(32, 32),
    batch_size=32,
    label_mode="int"
)

# Predict on test dataset
predictions = model.predict(test_ds)
predicted_labels = tf.argmax(predictions, axis=1)

# Show actual vs predicted images
actual_imgs, actual_labels = next(iter(test_ds))
plt.figure(figsize=(10, 10))
for i in range(len(actual_imgs)):
    plt.subplot(8, 4, i + 1)
    plt.imshow(actual_imgs[i].numpy().astype("uint8"))
    plt.title(f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[actual_labels[i]]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

"""# 3. Performing classification of pet’s images with residual network"""

# Install FastAI
!pip install -q git+https://github.com/fastai/fastai.git

# Imports
from fastai.vision.all import *

# Load dataset
path = untar_data(URLs.PETS)
files = get_image_files(path/"images")

# Define labels using regex
pattern = r'^(.*)_\d+.jpg$'

# Create dataloaders
dls = ImageDataLoaders.from_name_re(path, files, pattern, item_tfms=Resize(224))
dls.show_batch()

# Build and train model
learn = vision_learner(dls, resnet18, metrics=error_rate, model_dir="/kaggle/working/model")
learn.fit(3)

# Evaluate
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
interp.print_classification_report()

"""# 4. Predict the character sequence of a custom textual paragraph using RNN"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Text and vocab
text = "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers, where's the peck of pickled peppers Peter Piper picked?"
chars = sorted(set(text))
vocab = {ch: i for i, ch in enumerate(chars)}
inv_vocab = {i: ch for ch, i in vocab.items()}

# Create sequences
def create_seq(text, seq_len=5):
    x, y = [], []
    for i in range(len(text) - seq_len):
        x.append([vocab[c] for c in text[i:i+seq_len]])
        y.append(vocab[text[i+seq_len]])
    return np.array(x), np.array(y)

seq_len = 5
x, y = create_seq(text, seq_len)
x = tf.one_hot(x, len(vocab))
y = tf.one_hot(y, len(vocab))

# Build model
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(seq_len, len(vocab))),
    Dense(len(vocab), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=500, verbose=0)

# Generate text
seed = "Peter picked"
output = seed
for _ in range(50):
    inp = tf.one_hot([vocab[c] for c in output[-seq_len:]], len(vocab))
    pred = model.predict(tf.expand_dims(inp, 0), verbose=0)
    next_char = inv_vocab[np.argmax(pred)]
    output += next_char
    print(output)



"""# 5. Perform a time series prediction using LSTM"""

# Install required package
!pip install -q openpyxl

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import r2_score

# Load and preprocess data
df = pd.read_csv("AirPassengers.csv", parse_dates=["date"])
df["#Passengers"] = df["value"]
df["Date"] = df["date"].dt.date
ts = pd.Series(df["#Passengers"].values, index=df["Date"])
plt.plot(ts)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[["#Passengers"]])

# Create sequences
def create_dataset(data, steps=15):
    X, y = [], []
    for i in range(len(data) - steps):
        X.append(data[i:i+steps, 0])
        y.append(data[i+steps, 0])
    return np.array(X), np.array(y)

# Train/test split
train_size = int(len(data_scaled) * 0.7)
train, test = data_scaled[:train_size], data_scaled[train_size:]

X_train, y_train = create_dataset(train)
X_test, y_test = create_dataset(test)

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build and train model
model = Sequential([
    LSTM(20, input_shape=(1, 15)),
    Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=0)

# Predict and evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print("Train R2:", r2_score(y_train, train_pred))
print("Test R2:", r2_score(y_test, test_pred))

# Inverse transform for plotting
train_pred_inv = scaler.inverse_transform(train_pred)
test_pred_inv = scaler.inverse_transform(test_pred)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(train_pred_inv, label="Train Prediction")
plt.plot(range(len(train_pred_inv), len(train_pred_inv)+len(test_pred_inv)), test_pred_inv, label="Test Prediction")
plt.legend()
plt.title("LSTM Time Series Forecast")
plt.show()



"""# 7. Perform machine translation using custom sentences to translate from English to Hindi"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Data
src_texts = ["hello world", "how are you", "good morning"]
tgt_texts = ["नमस्ते दुनिया", "आप कैसे हैं", "सुप्रभात"]
tgt_in = [f"<sos> {t}" for t in tgt_texts]
tgt_out = [f"{t} <eos>" for t in tgt_texts]

# Vocab building
def build_vocab(texts, special_tokens=[]):
    vocab = {tok: i+1 for i, tok in enumerate(special_tokens)}
    for sent in texts:
        for word in sent.split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1
    return vocab

src_vocab = build_vocab(src_texts)
tgt_vocab = build_vocab(tgt_texts, special_tokens=["<sos>", "<eos>"])
rev_tgt_vocab = {i: w for w, i in tgt_vocab.items()}

# Convert to sequences
def to_seq(texts, vocab):
    return [[vocab.get(w, 0) for w in s.split()] for s in texts]

enc_seqs = pad_sequences(to_seq(src_texts, src_vocab), padding='post')
dec_in_seqs = pad_sequences(to_seq(tgt_in, tgt_vocab), padding='post')
dec_out_seqs = pad_sequences(to_seq(tgt_out, tgt_vocab), padding='post')

# Model
embed_dim, units = 64, 128
src_vocab_size = len(src_vocab) + 1
tgt_vocab_size = len(tgt_vocab) + 1
enc_len, dec_len = enc_seqs.shape[1], dec_in_seqs.shape[1]

enc_input = Input(shape=(enc_len,))
dec_input = Input(shape=(dec_len,))

enc_emb = Embedding(src_vocab_size, embed_dim)(enc_input)
enc_out, h, c = LSTM(units, return_sequences=True, return_state=True)(enc_emb)

dec_emb = Embedding(tgt_vocab_size, embed_dim)(dec_input)
dec_out, _, _ = LSTM(units, return_sequences=True, return_state=True)(dec_emb, initial_state=[h, c])

attn = Attention()([dec_out, enc_out])
context = Concatenate()([dec_out, attn])
out = Dense(tgt_vocab_size, activation='softmax')(context)

model = Model([enc_input, dec_input], out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([enc_seqs, dec_in_seqs], np.expand_dims(dec_out_seqs, -1), epochs=150, batch_size=2, verbose=0)

# Inference
def translate(inp):
    seq = pad_sequences(to_seq([inp], src_vocab), maxlen=enc_len, padding='post')
    dec_seq = np.zeros((1, dec_len))
    dec_seq[0, 0] = tgt_vocab["<sos>"]

    result = []
    for i in range(1, dec_len):
        preds = model.predict([seq, dec_seq], verbose=0)
        token = np.argmax(preds[0, i-1])
        word = rev_tgt_vocab.get(token, "<unk>")
        if word == "<eos>":
            break
        result.append(word)
        dec_seq[0, i] = token
    return " ".join(result)

# Output
for s in src_texts:
    print(f'Input: "{s}"')
    print(f'Predicted: "{translate(s)}"\n')



"""# 8. Create a RAG model which fine tunes a
LLM with any external knowledge source
and create a app which can answer
questions from than knowledge source
"""

# Install libraries
!pip install -q sentence-transformers transformers

from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import codecs

# Load external knowledge (text file)
with codecs.open('cat-facts.txt', 'r', encoding='utf-8', errors='ignore') as f:
    docs = f.readlines()

# Load models
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map='auto', torch_dtype=torch.float16)
gen = pipeline("text-generation", model=llm, tokenizer=tokenizer, max_new_tokens=200)

# Embed and store all knowledge chunks
db = [(text, embed_model.encode(text)) for text in docs]

# Similarity + retrieval
def retrieve(query, top_n=3):
    q_vec = embed_model.encode(query)
    ranked = sorted(db, key=lambda x: torch.nn.functional.cosine_similarity(
        torch.tensor(q_vec), torch.tensor(x[1]), dim=0), reverse=True)
    return [text for text, _ in ranked[:top_n]]

# Ask
query = input("Ask a question: ")
context = retrieve(query)

# Prompt and response
prompt = f"You are a helpful chatbot.\nUse only this context:\n" + \
         "".join(f" - {c}" for c in context) + f"\nQuestion: {query}\nAnswer:"
response = gen(prompt)[0]['generated_text']
print("\nAnswer:", response[len(prompt):].strip())

"""# 9. Perform tokenization and next sentence prediction with Bert"""

!pip install numpy==1.26.4

import numpy as np
import pandas as pd
import os

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

text = """Artificial Intelligence is transforming industries at a rapid pace.
From healthcare to finance, AI systems are streamlining operations, improving outcomes,
and opening up new possibilities.
The core of modern AI is machine learning, where algorithms learn patterns from data to make predictions or decisions.
With the rise of big data, cloud computing, and powerful hardware, the capabilities of AI continue to expand."""

word_text = text.split(" ")
encoding = tokenizer.encode(text)
print("Token IDs:", encoding)
tokens = tokenizer.convert_ids_to_tokens(encoding)
print(tokens[0:11])

sentences = [
    ("Machine learning is a subset of AI.", "AI encompasses many fields, including machine learning.", 1),
    ("Python is a popular programming language.", "Pizza is a delicious Italian dish.", 0),
    ("Cloud computing enables remote data access.", "Data can be accessed from anywhere via the cloud.", 1),
    ("Quantum computing is still experimental.", "Dogs love playing fetch.", 0),
    ("Cybersecurity is essential for data protection.", "Hackers exploit vulnerabilities in software.", 1),
    ("Tech companies are investing in AI.", "Water boils at 100 degrees Celsius.", 0)
]

from datasets import Dataset
dataset = Dataset.from_dict({
    "sentence1": [s[0] for s in sentences],
    "sentence2": [s[1] for s in sentences],
    "label": [s[2] for s in sentences]
})

train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding="max_length")

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="/working/results",
    learning_rate=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="/working/logs",
    logging_steps=10,
    save_strategy="epoch",
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_args,
    compute_metrics=compute_metrics
)

trainer.train()
eval_results = trainer.evaluate()
print(f"\nEvaluation Results: {eval_results}")

"""# 10. Generate a sine wave patten with GAN"""

# Install torch in Colab
!pip install torch

import torch
from torch import nn
import matplotlib.pyplot as plt
import math

# Generate real sine wave data
n = 1024
x = 2 * math.pi * torch.rand(n)
y = torch.sin(x)
real_data = torch.stack([x, y], dim=1)

plt.plot(x, y, ".")
plt.title("Real Sine Wave")
plt.show()

# DataLoader
batch_size = 32
loader = torch.utils.data.DataLoader(real_data, batch_size=batch_size, shuffle=True)

# Discriminator Model
D = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1), nn.Sigmoid()
)

# Generator Model
G = nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(),
    nn.Linear(32, 2)
)

# Optimizers & Loss
opt_D = torch.optim.Adam(D.parameters(), lr=1e-3)
opt_G = torch.optim.Adam(G.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# Train Loop
for epoch in range(301):
    for real in loader:
        # Train D
        z = torch.randn(batch_size, 2)
        fake = G(z).detach()

        inputs = torch.cat([real, fake])
        labels = torch.cat([torch.ones(batch_size, 1), torch.zeros(batch_size, 1)])

        D.zero_grad()
        loss_D = loss_fn(D(inputs), labels)
        loss_D.backward()
        opt_D.step()

        # Train G
        z = torch.randn(batch_size, 2)
        G.zero_grad()
        loss_G = loss_fn(D(G(z)), torch.ones(batch_size, 1))
        loss_G.backward()
        opt_G.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | D Loss: {loss_D:.4f} | G Loss: {loss_G:.4f}")

# Generate & Plot
with torch.no_grad():
    gen = G(torch.randn(n, 2))
plt.plot(x, y, ".", label="Real")
plt.plot(gen[:, 0], gen[:, 1], ".", label="Generated", alpha=0.7)
plt.legend(); plt.title("Real vs. Generated Sine Data"); plt.show()

