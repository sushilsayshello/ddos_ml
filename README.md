# A Comparative Study on Machine Learning Models for DDoS Detection in IoT Networks

This project aims to analyze and compare various machine learning models in detecting Distributed Denial of Service (DDoS) attacks within IoT environments. The study focuses on identifying the most effective model for classifying network traffic as normal or indicative of a DDoS attack, contributing to the broader effort of enhancing IoT security.

## Project Structure

- **Introduction**  
  - **Background**: Discusses the growing threat of DDoS attacks in IoT networks.
  - **Motivation**: Explains the limitations of traditional detection techniques and the need for adaptive ML-based security.
  - **Contribution**: Summarizes the study's focus on comparing machine learning models for DDoS detection in IoT environments.

- **Literature Review**  
  - Overview of traditional DDoS detection methods and the role of machine learning.
  - Examines supervised, unsupervised, and semi-supervised learning approaches for DDoS detection.

- **Methodology**  
  - **Dataset Description**: Details the IoT network traffic dataset used for training and testing the models.
  - **Model Description**: Introduces the machine learning models evaluated in this study: XGBoost, K-Nearest Neighbors (KNN), Stochastic Gradient Descent (SGD), and Naïve Bayes.
  - **Evaluation Metrics**: Describes the performance metrics used to evaluate the models, including accuracy, precision, recall, and F1 score.

- **Results**  
  - Presents the performance of each model based on the selected metrics. XGBoost was found to perform best, followed by KNN, SGD, and Naïve Bayes.

- **Discussion**  
  - Analyzes why certain models perform better than others in the context of IoT DDoS detection.

- **Conclusion and Future Work**  
  - Summarizes key findings and provides suggestions for future work, including model optimization, hybrid approaches, and the exploration of deep learning techniques.

## Technologies Used

- **Machine Learning Models**: XGBoost, K-Nearest Neighbors (KNN), Stochastic Gradient Descent (SGD), Naïve Bayes
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score

## Results Summary

| Model                   | Accuracy | Precision | Recall | F1 Score |
|-------------------------|----------|-----------|--------|----------|
| XGBoost                 | 95.2%    | 94.8%     | 95.0%  | 94.9%    |
| K-Nearest Neighbors     | 89.7%    | 88.9%     | 90.1%  | 89.5%    |
| Stochastic Gradient Descent | 87.5% | 86.7%     | 87.3%  | 87.0%    |
| Naïve Bayes             | 85.3%    | 84.5%     | 86.0%  | 85.2%    |

## Future Work

- **Hybrid Models**: Integrate the strengths of multiple models for enhanced detection.
- **Deep Learning Techniques**: Explore CNNs and RNNs for more advanced anomaly detection.
- **Real-World Testing**: Validate model performance in operational IoT environments.
- **Ethical and Regulatory Compliance**: Ensure data privacy and bias mitigation in future models.

## References

For detailed references, please refer to the References section in the project document.

---

**Prepared by**: Sushil Shakya  
**Supervised by**: Dr. Robert Abbas
