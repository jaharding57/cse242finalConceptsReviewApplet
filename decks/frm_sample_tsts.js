const frm_sample_tsts = [
  {
    "q": "What is the i.i.d. assumption in machine learning?",
    "a": "It assumes data samples are drawn independently from the same distribution."
  },
  {
    "q": "Why is it misleading to evaluate a model on training data?",
    "a": "It can result in overfitting and misleading performance estimates."
  },
  {
    "q": "Define overfitting and underfitting.",
    "a": "Overfitting is when a model captures noise; underfitting is when it fails to capture patterns."
  },
  {
    "q": "Which regularization method promotes sparsity and why?",
    "a": "L1 regularization (Lasso) encourages sparsity by penalizing the absolute values of coefficients."
  },
  {
    "q": "What is the maximum likelihood estimate of h given 5 heads in 8 coin flips?",
    "a": "The MLE is h = 5/8 = 0.625."
  },
  {
    "q": "What is the naive assumption in Naive Bayes?",
    "a": "Features are conditionally independent given the class."
  },
  {
    "q": "What data does the Perceptron algorithm converge on?",
    "a": "Linearly separable data."
  },
  {
    "q": "What is the dual form of the optimal SVM weight vector?",
    "a": "w = \u03a3 \u03b1_i y_i x_i, where \u03b1_i are dual variables."
  },
  {
    "q": "How do you perform PCA with kernel methods?",
    "a": "Use the kernel trick to compute principal components in feature space via eigen-decomposition."
  },
  {
    "q": "True or False: Logistic loss is better than L2 loss for classification?",
    "a": "True. Logistic loss is designed for classification problems."
  },
  {
    "q": "True or False: Less training data reduces overfitting?",
    "a": "False. Less data may increase overfitting due to insufficient generalization."
  },
  {
    "q": "Should boosting stop if a weak learner achieves zero error?",
    "a": "No. Boosting continues to combine multiple learners for robustness."
  },
  {
    "q": "One similarity and one difference between feature selection and PCA?",
    "a": "Both reduce dimensionality; PCA transforms features, selection chooses from existing ones."
  },
  {
    "q": "What does the kernel trick do?",
    "a": "It enables operations in high-dimensional space without explicitly computing coordinates."
  },
  {
    "q": "What does bias-variance decomposition describe?",
    "a": "It quantifies how model complexity impacts bias (underfitting) and variance (overfitting)."
  },
  {
    "q": "What is the curse of dimensionality?",
    "a": "High-dimensional data becomes sparse, making learning difficult and distances less meaningful."
  },
  {
    "q": "What does PCA do?",
    "a": "Reduces dimensionality by projecting data onto axes of highest variance."
  },
  {
    "q": "What happens in the E-step and M-step of EM for GMM?",
    "a": "E-step estimates responsibilities; M-step updates means, covariances, and weights."
  },
  {
    "q": "Difference between KNN and K-means?",
    "a": "KNN is a supervised classifier; K-means is an unsupervised clustering algorithm."
  },
  {
    "q": "What is Bayes' theorem and how is it used?",
    "a": "Posterior = (Likelihood \u00d7 Prior) / Evidence; updates belief after seeing data."
  },
  {
    "q": "What is MLE of spam rate given EMAIL = {S, N, N, S, N, S, S, S}?",
    "a": "5 spam out of 8 \u2192 MLE = 5/8 = 0.625."
  },
  {
    "q": "Supervised vs Unsupervised learning?",
    "a": "Supervised uses labeled data; unsupervised finds patterns without labels."
  },
  {
    "q": "What does the Universal Approximation Theorem say?",
    "a": "A neural network with one hidden layer can approximate any continuous function."
  },
  {
    "q": "How do you make SVM data more separable?",
    "a": "Use kernels to project data into a higher-dimensional space."
  },
  {
    "q": "What is discounting in RL?",
    "a": "Future rewards are multiplied by \u03b3 < 1 to prioritize immediate outcomes."
  },
  {
    "q": "What is entropy in decision trees?",
    "a": "A measure of uncertainty or impurity in a data split."
  },
  {
    "q": "What is Q-value in MDP?",
    "a": "The expected utility of taking an action from a state under policy \u03c0."
  },
  {
    "q": "Why use EM instead of gradient descent for GMMs?",
    "a": "EM handles latent variables efficiently and ensures monotonic likelihood improvement."
  },
  {
    "q": "Difference between L1 and L2 regularization?",
    "a": "L1 produces sparse models; L2 keeps all weights small."
  },
  {
    "q": "What is collapsing variance in EM?",
    "a": "Occurs when a Gaussian collapses to a point, leading to infinite likelihood."
  },
  {
    "q": "Why is exploration necessary in RL?",
    "a": "To discover new states and actions that might lead to better policies."
  },
  {
    "q": "How does Q-learning differ from value learning?",
    "a": "Q-learning directly learns action-values without needing a model of the environment."
  }
];