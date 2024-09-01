# AI-Text-Detector

**AI-Text-Detector** is a Python-based application designed to analyze and determine whether a given text is AI-generated or human-written. It uses advanced NLP models and various metrics to assess text characteristics like perplexity, burstiness, entropy, grammar, and spelling. The project is built using PyTorch, spaCy, and other cutting-edge NLP tools.

## Features

- **Perplexity Calculation**: Measures the fluency of the text based on GPT-2's language model.
- **Burstiness Calculation**: Evaluates the variation in sentence lengths to detect AI-generated patterns.
- **Entropy Calculation**: Analyzes the randomness in word distribution within the text.
- **N-Gram Diversity**: Checks for repetitive patterns using N-Gram analysis.
- **Grammar and Spelling Scores**: Provides insights into the grammatical accuracy and spelling correctness of the text.
- **Human Score Prediction**: Uses a fine-tuned BERT model to predict whether the text is more likely AI-generated or human-written.

## Requirements

- Python 3.8+
- PyTorch
- spaCy
- Transformers
- TextBlob
- SpellChecker
- scikit-learn

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ai-text-detector.git
   cd ai-text-detector
   ```

2. **Set Up the Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Necessary spaCy Models**:
   ```bash
   python -m spacy download en_core_web_trf
   ```

## Usage

To run the application and evaluate a sample text, execute:

```bash
python app.py
```

Replace the text in the `text` variable within `app.py` with any text you wish to analyze.

### Docker Usage

1. **Build the Docker Image**:
   ```bash
   docker build -t ai-text-detector .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -d -p 8000:8000 ai-text-detector
   ```

## Contributing

We welcome contributions! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project utilizes the [Hugging Face Transformers](https://github.com/huggingface/transformers) library for NLP tasks.
- Special thanks to the contributors of [spaCy](https://github.com/explosion/spaCy) and [PyTorch](https://pytorch.org/).

## Contact

For any inquiries or feedback, please reach out at [your_email@example.com](mailto:adarshsingh254001@gmail.com) or create an issue on the repository.

---

### Example ReadMe in Markdown Format:

```markdown
# AI-Text-Detector

**AI-Text-Detector** is a Python-based application designed to analyze and determine whether a given text is AI-generated or human-written. It uses advanced NLP models and various metrics to assess text characteristics like perplexity, burstiness, entropy, grammar, and spelling. The project is built using PyTorch, spaCy, and other cutting-edge NLP tools.

## Features

- **Perplexity Calculation**: Measures the fluency of the text based on GPT-2's language model.
- **Burstiness Calculation**: Evaluates the variation in sentence lengths to detect AI-generated patterns.
- **Entropy Calculation**: Analyzes the randomness in word distribution within the text.
- **N-Gram Diversity**: Checks for repetitive patterns using N-Gram analysis.
- **Grammar and Spelling Scores**: Provides insights into the grammatical accuracy and spelling correctness of the text.
- **Human Score Prediction**: Uses a fine-tuned BERT model to predict whether the text is more likely AI-generated or human-written.

## Requirements

- Python 3.8+
- PyTorch
- spaCy
- Transformers
- TextBlob
- SpellChecker
- scikit-learn

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ai-text-detector.git
   cd ai-text-detector
   ```

2. **Set Up the Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Necessary spaCy Models**:
   ```bash
   python -m spacy download en_core_web_trf
   ```

## Usage

To run the application and evaluate a sample text, execute:

```bash
python app.py
```

Replace the text in the `text` variable within `app.py` with any text you wish to analyze.

### Docker Usage

1. **Build the Docker Image**:
   ```bash
   docker build -t ai-text-detector .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -d -p 8000:8000 ai-text-detector
   ```

## Contributing

We welcome contributions! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project utilizes the [Hugging Face Transformers](https://github.com/huggingface/transformers) library for NLP tasks.
- Special thanks to the contributors of [spaCy](https://github.com/explosion/spaCy) and [PyTorch](https://pytorch.org/).

## Contact

For any inquiries or feedback, please reach out at [your_email@example.com](mailto:your_email@example.com) or create an issue on the repository.
```

This `README.md` provides clear instructions for users and developers on how to set up and use your project while adhering to the Apache 2.0 license. Make sure to replace placeholders like `yourusername` and `your_email@example.com` with your actual details before publishing.
