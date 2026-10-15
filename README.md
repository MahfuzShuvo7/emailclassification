# Email Spam Classification with Linear SVM

A Python notebook that classifies email text as **spam** or **legitimate email** using TF-IDF and a Linear Support Vector Machine (SVM). It also explains predictions by showing the words or phrases that most strongly support the model's decision.

This is a supervised learning project: the model learns from emails that already have a known label.

## Features

- Automatic dataset download from Kaggle.
- Checks for missing values, duplicate entries, and class counts.
- Email length analysis, class distribution plots, and word clouds.
- Text cleaning and conversion into numerical features with TF-IDF.
- Linear SVM training, parameter tuning, and evaluation.
- Twelve illustrative emails with expected labels, predictions, and explanations.

## Dataset

Source: [Spam Email Classification on Kaggle](https://www.kaggle.com/datasets/shalmamuji/spam-email-classification).

The notebook downloads the dataset using `kagglehub` and selects a CSV containing these columns:

| Column | Meaning |
| --- | --- |
| `email` | Email text, renamed to `message` in the notebook |
| `label` | `ham` for legitimate email or `spam` for unwanted email |

The saved notebook outputs report:

| Dataset detail | Count |
| --- | ---: |
| Original entries | 10,000 |
| Legitimate entries | 8,500 |
| Spam entries | 1,500 |
| Duplicate entries before cleaning | 8,223 |
| Entries after duplicate removal and missing-value handling | 1,777 |
| Training emails | 1,421 |
| Testing emails | 356 |

These counts describe the saved run; downloading a newer dataset version may produce different results. The dataset is downloaded separately and is not included in this repository.

## How It Works

1. **Load and inspect the emails.** Check labels, missing values, duplicates, email lengths, and common words.
2. **Clean the text.** Remove duplicate entries and rows missing text or labels. Convert text to lowercase; remove URLs, email addresses, the `subject:` heading, numbers, and punctuation. Remove common English words, reduce words to their basic form, and normalize spaces.
3. **Prepare the labels.** Convert `ham` to `0` and `spam` to `1`.
4. **Split the data.** Use 80% for training and 20% for testing, keeping approximately the same class proportions. The split uses `random_state=42`.
5. **Convert text into numbers.** TF-IDF assigns numerical values to words and word pairs. The vocabulary is limited to 5,000 features and is learned from the training data within the pipeline.
6. **Train a Linear SVM.** `LinearSVC` learns to separate the numerical representations of spam and legitimate emails.
7. **Evaluate and tune.** Report accuracy, precision, recall, F1 score, and a confusion matrix. Try `C` values of `0.1`, `1`, and `10` using five-fold cross-validation.
8. **Predict and explain.** Use the tuned model to classify example emails and display the strongest supporting words or phrases.

The character counts, word counts, line counts, and subject-presence checks are used for exploration. The classifier itself uses the TF-IDF representation of cleaned email text.

## Getting Started

### Requirements

- Python 3 with pip.
- Jupyter Notebook, JupyterLab, or VS Code with the Python and Jupyter extensions.
- Internet access for the initial package, dataset, and NLTK language-resource downloads.

### Run the Notebook

```bash
git clone https://github.com/MahfuzShuvo7/emailclassification.git
cd emailclassification
```

Open [`emailclassification.ipynb`](emailclassification.ipynb), select a Python kernel, and run the cells from top to bottom.

The first cell installs the project libraries into the notebook's Python environment:

```python
%pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn kagglehub
```

The later setup cells download NLTK stopwords and WordNet resources. Dataset loading is automatic; no Google Colab upload is required. Restart the kernel if newly installed packages are not recognized, then run the notebook again.

To run with Jupyter Notebook from a terminal, install and launch it:

```bash
python -m pip install notebook
python -m notebook emailclassification.ipynb
```

## Classify Your Own Email

After running the training, tuning, and final prediction cells:

```python
email = "Hi team, please review the project report before tomorrow's meeting."

print("Prediction:", predict_spam(email))
print("Why:", explain_prediction(email))
```

`predict_spam()` returns either `Spam Email` or `Legitimate Email`. `explain_prediction()` returns up to five words or phrases supporting that prediction.

## How Prediction Explanations Work

For each recognized word or phrase present in an email, the notebook multiplies its TF-IDF value by its learned SVM weight. It ranks the contributions that favor the predicted label and prints the strongest ones.

These explanations come from the trained model rather than a manually written list of spam keywords. They use cleaned words, so the displayed terms can differ from the original wording. If no words match the vocabulary, the explanation states that the prediction comes from the model's baseline bias.

The explanation shows the strongest supporting signals, not the entire decision calculation or proof that an email is malicious. Predictions also depend on other word contributions and the model's baseline bias; an explanation can accompany an incorrect prediction.

## Saved Results and Limitations

The saved run reports **100% accuracy, precision, recall, and F1** on the 356-email test set. The tuned `C` value is `0.1`. The twelve illustrative examples produce **11 matches out of 12**; the assignment reminder is incorrectly classified as spam.

The test-set report evaluates the initial pipeline. The example predictions and explanations use the tuned model, `grid.best_estimator_`.

These results describe this dataset and saved run, not guaranteed performance on real inboxes. Exact duplicate rows are removed, but the notebook does not group similar email templates or check for duplicates created by text cleaning before splitting. Removing numbers, links, and punctuation can also discard useful clues.

The final cross-validation section uses parameters selected earlier, so it is not an independent assessment of the entire tuning process. A stronger evaluation would group related emails, tune only within training data, and evaluate the final model on a separate unseen dataset.

## Libraries

| Library | Purpose |
| --- | --- |
| pandas and NumPy | Data tables and numerical calculations |
| Matplotlib and Seaborn | Charts |
| WordCloud | Common-word visualizations |
| NLTK | Common English words and word-form reduction |
| scikit-learn | TF-IDF, SVM, data splitting, tuning, and evaluation |
| kagglehub | Dataset download |

## Project File

- [`emailclassification.ipynb`](emailclassification.ipynb): dataset loading, exploration, cleaning, training, evaluation, and explained predictions.
