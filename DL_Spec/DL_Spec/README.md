# Deep Learning Specialisation — Self Project (Oct'24–Dec'24)

This mini-project demonstrates core deep learning architectures and training tricks:
- Convolutional Neural Networks (CNN) with BatchNorm and Dropout
- Recurrent Networks (RNN) and LSTMs with Dropout
- Transformer encoder classifier with Dropout (LayerNorm inside PyTorch layer)
- Tiny real-world-ish demos: toy vision classification, toy sentiment, and a tiny seq2seq chatbot

CPU-only friendly, fast to run on small synthetic/toy datasets.

## Setup

Create a Python 3.9+ virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Project layout

- `src/models/`
  - `cnn.py` — Simple CNN with BatchNorm/Dropout for 28x28 grayscale
  - `rnn_lstm.py` — SimpleRNN and SimpleLSTM for text classification
  - `transformer.py` — Tiny Transformer encoder classifier
- `src/trainers/simple_trainer.py` — Minimal trainer used by vision demo
- `src/utils/data_utils.py` — Synthetic MNIST-like data, toy sentiment/chat datasets, vocab
- `apps/`
  - `vision_demo.py` — Trains CNN for a couple of epochs on synthetic images
  - `sentiment_demo.py` — Runs LSTM and Transformer on toy sentiment
  - `chatbot_demo.py` — Trains a tiny seq2seq to parrot simple responses and does greedy decode
- `tests/smoke_test.py` — Instantiates each model and runs a single optimization step

## Try it

- Vision (CNN):
```bash
python -m apps.vision_demo
```

- Sentiment (LSTM + Transformer):
```bash
python -m apps.sentiment_demo
```

- Chatbot (seq2seq):
```bash
python -m apps.chatbot_demo
```

- Smoke tests:
```bash
python -m pytest -q
```

## Notes
- For speed, datasets are tiny and synthetic; accuracy is not the goal here.
- Dropout and BatchNorm are demonstrated in CNN, Dropout in RNN/LSTM/Transformer.
- You can tweak model sizes and epochs as you like to experiment.
