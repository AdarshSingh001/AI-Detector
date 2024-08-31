import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
import json
import math
import spacy
from textblob import TextBlob 

# Initialize models and tools
nlp = spacy.load('en_core_web_trf')

def calculate_perplexity(text):
    model_id = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)

    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity

def calculate_burstiness(text):
    sentences = text.split('.')
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence]

    avg_length = sum(sentence_lengths) / len(sentence_lengths)
    variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
    burstiness = variance ** 0.5

    return burstiness

def calculate_entropy(text):
    words = text.split()
    word_freq = Counter(words)
    probs = [freq / len(words) for freq in word_freq.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    return entropy

def calculate_ngram_diversity(text, n=3):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    ngram_freq = Counter(ngrams)
    ngram_diversity = len(ngram_freq) / len(words)
    return ngram_diversity

def calculate_expanded_grammar_score(text):
    doc = nlp(text)
    
    num_errors = 0
    
    for token in doc:
        if token.dep_ == "amod" and token.head.pos_ != "NOUN":
            num_errors += 1
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            if token.tag_ == "NN" and token.head.tag_ not in ["VBZ", "VBP"]:
                num_errors += 1
        if token.dep_ == "advmod" and token.head.pos_ not in ["VERB", "ADJ"]:
            num_errors += 1
    
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    grammar_score = max(100 - (num_errors / num_sentences) * 100, 0)
    
    return grammar_score

def calculate_spelling_score(text):
    blob = TextBlob(text)
    words = text.split()
    corrected_text = str(blob.correct())
    correct_words = len([word for word, corrected in zip(words, corrected_text.split()) if word == corrected])

    # Spelling Score: percentage of correctly spelled words
    spelling_score = (correct_words / len(words)) * 100 if len(words) > 0 else 100

    return spelling_score

def calculate_metrics(text):
    perplexity = calculate_perplexity(text)
    burstiness = calculate_burstiness(text)
    entropy = calculate_entropy(text)
    ngram_diversity = calculate_ngram_diversity(text)
    grammar_score = calculate_expanded_grammar_score(text)
    spelling_score = calculate_spelling_score(text)

    human_score = (
        0.3 * (100 - perplexity) +
        0.25 * (100 - burstiness * 10) +
        0.25 * entropy +
        0.2 * ngram_diversity * 100 
    )

    results = {
        "Human Score": max(min(human_score, 100), 0),
        "Grammar Score": grammar_score,
        "Spelling Score": spelling_score
    }
    
    return json.dumps(results, indent=4)

def main():
    text = "This is a sample text to analyze if it is AI-generated or human-written."
    results_json = calculate_metrics(text)
    print(f"Results:\n{results_json}")

if __name__ == "__main__":
    main()