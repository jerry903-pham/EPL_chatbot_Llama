# Premier League AI Assistant

A Streamlit app for Premier League match prediction and team analysis, with a local chat interface powered by Ollama (Phi-3). Individual academic project.

## Features

- **Match prediction** using a Random Forest model
- **Team analysis** and **visualizations** (top scoring teams, cards distribution, head-to-head)
- **Bulk prediction** from an uploaded CSV file
- **Chat**: a locally hosted Phi-3 model (via Ollama) extracts team names from natural-language questions, and the app routes them to the prediction or analysis functions using keyword matching

## How it works

- **Model:** scikit-learn `RandomForestClassifier` (300 trees, max depth 15)
- **Features (20):** engineered match features (bookmaker odds), team features (3-match rolling form), and referee features
- **Data:** 380 Premier League matches from the 2024/25 season (`epl_matches.csv`)
- The Random Forest makes the predictions. The LLM is only used for chat and team-name extraction.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The **Chat** tab needs [Ollama](https://ollama.com) installed and running:

```bash
ollama pull phi3:mini
```

The other tools work without Ollama.

`model_package.pkl` was created with scikit-learn 1.7.2. To retrain it with your own version:

```bash
python train_model.py
```

## Limitations

- Small, single-season dataset, so results vary a lot between train/test splits.
- Chat routing is keyword-based, and a small local model can misread team names.
- Academic project. Not production software and not betting advice.
