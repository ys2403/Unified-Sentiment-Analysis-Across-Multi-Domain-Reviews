![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green.svg)

# Unified Sentiment Analysis Across Multi-Domain Reviews

## View the Project
You can view the code and resources used in this project [here](https://github.com/ys2403/Unified-Sentiment-Analysis-Across-Multi-Domain-Reviews).

---

## Project Overview

This project focuses on building an interactive **Sentiment Analysis System** across various domains using both traditional and deep learning models. By leveraging datasets from movie reviews, Yelp reviews, and Twitter tweets, the project aims to provide meaningful insights into user sentiments, helping businesses make informed, data-driven decisions.

### Dataset

The datasets used in this project include:
- **Movie Reviews**: Provides detailed evaluations with a diverse linguistic style.
- **Yelp Reviews**: Informal customer feedback with slang and local dialects.
- **Tweets**: Short, often abbreviated messages requiring handling of hashtags and mentions.

The dataset is included in this repository as `train_150k.txt`, which can be found in the code folder.

---

## Objectives

The primary goal is to create a **unified sentiment analysis model** capable of accurately classifying sentiments across multiple domains. The project explores traditional models like Logistic Regression as well as advanced deep learning models, including CNNs and LSTMs, to improve prediction accuracy and adaptability.

---

## Benefits of the Project

This project provides valuable insights and benefits, such as:
- **Cross-Domain Sentiment Analysis**: It improves the understanding of user sentiments across different domains, which can help businesses in better decision-making and targeted marketing.
- **Model Comparisons and Insights**: By experimenting with various models, we identified the best-performing techniques and learned the trade-offs between accuracy, speed, and interpretability.
- **Skill Development**: Through this project, I learned and implemented different NLP and deep learning models, enhancing my understanding of their applications and limitations in real-world sentiment analysis tasks.

---

## Key Steps and Methodologies

### Data Preprocessing

1. **Text Cleaning**: Removal of special characters and standardizing text to lowercase to make it suitable for analysis.
2. **Tokenization**: Splitting text into individual tokens for model input.
3. **Padding and Truncation**: Ensuring uniform sequence lengths for neural networks to improve model performance.

### Models Implemented

1. **Logistic Regression with TF-IDF**:
   - A simple and interpretable model that serves as the baseline.
   - Achieved a baseline accuracy of **81.25%**.

2. **Convolutional Neural Network (CNN)**:
   - Explored various CNN configurations to capture features in text data.
   - Achieved a test accuracy of **84.49%** with dense configurations and increased vocabulary size.

3. **CNN + Word2Vec Embeddings**:
   - Combined CNN with Word2Vec embeddings to capture semantic relationships in text.
   - Improved accuracy by focusing on meaningful words and achieved **83.38%**.

4. **LSTM + Word2Vec**:
   - Added an LSTM layer for capturing sequential dependencies in the text data.
   - Reached the highest test accuracy of **85.21%**, demonstrating the model's ability to capture complex language features.

---

## Performance Summary

| Model                  | Test Accuracy |
|------------------------|---------------|
| Logistic Regression    | 81.25%        |
| Basic CNN              | 81.89%        |
| CNN (increased vocab)  | 84.28%        |
| Dense CNN              | 84.49%        |
| CNN + Word Cloud       | 84.64%        |
| Word2Vec               | 79.28%        |
| Word2Vec + CNN         | 83.38%        |
| Dense CNN + Word2Vec   | 83.92%        |
| LSTM + Word2Vec        | 85.21%        |

---

## Model Insights

Each model contributed unique insights:
- **Logistic Regression** provided a simple baseline to measure the effectiveness of advanced models.
- **CNN Variants** allowed us to experiment with capturing spatial hierarchies in text, proving effective with increased vocabulary.
- **Word2Vec Embeddings** enhanced context understanding by placing semantically similar words close in the embedding space.
- **LSTM** further boosted accuracy by capturing dependencies in sequential text data, handling complex patterns better than CNN alone.

---

## Output and Evaluation

The best model, **LSTM + Word2Vec**, achieved an accuracy of **85.21%** on the test set, outperforming simpler models like Logistic Regression and CNN. This model demonstrated high precision, recall, and F1-scores, proving to be robust in cross-domain sentiment analysis. The performance of each model was evaluated based on metrics like accuracy, precision, recall, and F1-score to ensure consistency and reliability.

---

## Conclusion

This project demonstrates that, with proper tuning, traditional models can rival more advanced architectures like BERT for sentiment analysis on mixed-domain datasets. Through this exploration, we found that CNNs combined with Word2Vec and LSTMs offered the best balance of accuracy and efficiency.

**Impact**: By improving sentiment analysis across various domains, this project provides valuable insights for businesses aiming to understand customer sentiments more accurately.

---


