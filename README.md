# Run Smarter: Predict Every Split, Conquer Every Peak

Most race prediction tools today rely on simple averages or past race times. They tell you how fast you might run — but not how the terrain will affect you.

## Smart Peak Splits changes that.
It’s an intelligent system that learns from your real running data — including elevation, heart rate, and terrain profile — to give you accurate, personalized split predictions for any route.

By analyzing your GPX files from past runs, Smart Peak Splits builds a performance model that understands how you actually run: how you slow down on climbs, recover on descents, and adapt to elevation gain.
Then, when you upload a new route, it predicts your pace and effort for every kilometer — creating a realistic, data-driven race plan made just for you.

In short: Smart Peak Splits helps runners predict their real performance, not just their finish time — bringing terrain and physiology into the equation.

```
smart-peak-splits/
├── app.py                          # Streamlit app — drag & drop GPX and get predictions
│
├── data/
│   ├── raw/                        # Original GPX files (user uploads or training data)
│   ├── processed/                  # Parsed and cleaned data (CSV splits, features)
│   ├── models/                     # Trained models (.pkl, .pt, etc.)
│   └── input/                      # GPX files used for testing or demo predictions
│
├── src/
│   ├── __init__.py
│   ├── extract_features.py         # Read and parse GPX files into DataFrames
│   ├── feature_engineering.py      # Compute splits, elevation gain, pace, HR, etc.
│   ├── train_model.py              # Train ML or neural network models
│   ├── predict_route.py            # Predict pace/HR for a new route GPX
│   ├── orchestrator.py             # End-to-end pipeline (training + prediction)
│   └── utils.py                    # Helper functions (math, conversions, etc.)
│
├── notebooks-research/
│   ├── model_selection.ipynb       # Model experiments and evaluation
│   ├── data_exploration.ipynb      # Data visualization and insights
│   └── feature_tests.ipynb         # Feature engineering experiments
│
├── output/
│   ├── predictions/                # Final predicted splits (CSV)
│   └── plots/                      # Graphs (pace vs elevation, HR trends)
│
├── requirements.txt
└── README.md
```
