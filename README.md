# 📊 A Comparative Study on Machine Learning Models for DDoS Detection in IoT Networks

This project analyzes and compares various machine learning models for detecting Distributed Denial of Service (DDoS) attacks within IoT environments. The study aims to identify the most effective model for classifying network traffic as normal or indicative of a DDoS attack, contributing to the broader effort of enhancing IoT security.

## 📂 Project Structure

- **📖 Introduction**  
  - **🔍 Background**: Discusses the growing threat of DDoS attacks in IoT networks.
  - **🎯 Motivation**: Explains the limitations of traditional detection techniques and the need for adaptive ML-based security.
  - **💡 Contribution**: Summarizes the study's focus on comparing machine learning models for DDoS detection in IoT environments.

- **📚 Literature Review**  
  - Provides an overview of traditional DDoS detection methods and the transition to machine learning approaches.
  - Discusses supervised, unsupervised, and semi-supervised learning methods for DDoS detection.

- **⚙️ Methodology**  
  - **📊 Dataset Description**: Details the IoT network traffic dataset used for training and testing the models.
  - **🧠 Model Description**: Introduces the machine learning models evaluated: XGBoost, K-Nearest Neighbors (KNN), Stochastic Gradient Descent (SGD), and Naïve Bayes.
  - **📈 Evaluation Metrics**: Describes the performance metrics used, including accuracy, precision, recall, and F1 score.

- **📉 Results**  
  - Presents the performance of each model based on selected metrics. XGBoost performed the best, followed by KNN, SGD, and Naïve Bayes.

- **🗣️ Discussion**  
  - Analyzes why certain models perform better than others in the context of IoT DDoS detection.

- **📝 Conclusion and Future Work**  
  - Summarizes key findings and suggests directions for future research, including model optimization, hybrid approaches, and exploration of deep learning techniques.

## 💻 Technologies Used

- **Machine Learning Models**: XGBoost, K-Nearest Neighbors (KNN), Stochastic Gradient Descent (SGD), Naïve Bayes
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score

## 📊 Results Summary

| 📈 **Model**              | 🎯 **Accuracy** | ✅ **Precision** | 🔄 **Recall** | 📊 **F1 Score** |
|---------------------------|----------------|------------------|---------------|-----------------|
| **XGBoost**               | 0.998186       | 0.998184        | 0.998186      | 0.998180        |
| **K-Nearest Neighbors**   | 0.995587       | 0.995586        | 0.995587      | 0.995546        |
| **Stochastic Gradient Descent** | 0.982564 | 0.983273        | 0.982564      | 0.982291        |
| **Naïve Bayes**           | 0.910000       | 0.966957        | 0.910000      | 0.921368        |

## 🚀 Future Work

- **🧬 Hybrid Models**: Investigate models that combine Naïve Bayes and XGBoost for faster and more accurate classification.
- **🤖 Deep Learning Techniques**: Explore CNNs and RNNs for advanced anomaly detection in network traffic.
- **🛠️ Real-World Testing**: Implement the models within actual IoT infrastructures to validate their performance in operational settings.
- **🔒 Ethical and Regulatory Compliance**: Ensure data privacy and mitigate biases in future models to adhere to data protection regulations (e.g., GDPR).


---

**👤 Prepared by**: Sushil Shakya  
**👨‍🏫 Supervised by**: Dr. Robert Abbas
