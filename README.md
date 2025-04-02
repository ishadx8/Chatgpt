# AI Text Detection using Machine Learning

## 📌 Project Overview
This project implements a simple AI text detection system using machine learning. It differentiates between human-written and AI-generated text by analyzing linguistic features and applying a Random Forest classifier.

## 🛠️ Features
- **Text Preprocessing:** Tokenization, stopword removal, and case normalization.
- **Feature Engineering:** Extracts word count, character count, lexical diversity, and readability score.
- **TF-IDF Vectorization:** Converts processed text into numerical representations.
- **Machine Learning Model:** Trained using a Random Forest classifier.
- **AI Text Detection Function:** Predicts whether a given text is AI-generated or human-written.

## 📂 Dataset
A simulated dataset containing human and AI-generated sentences:
| Text | Label |
|------|--------|
| This is a human-written answer with natural variations. | Human |
| AI-generated responses often have structured formatting. | AI |

## 🔧 Installation
Ensure you have Python and the necessary dependencies installed:

```bash
pip install pandas nltk textstat scikit-learn
```

Download necessary NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🚀 Usage
1. Clone the repository:
```bash
git clone https://github.com/your-repo.git
cd your-repo
```
2. Run the script:
```bash
python ai_text_detector.py
```
3. Test AI text detection:
```python
print(detect_ai_text("This response is generated based on deep learning models."))
```

## 📊 Model Performance
The Random Forest classifier achieves an accuracy of around **X%** on the test set (update with actual results).

## 📜 License
This project is open-source and available under the MIT License.

## 🤝 Contributing
Feel free to submit pull requests or open issues to improve the model!

