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
  },
  {
    "q": "Why is ReLU preferred over Sigmoid in deep networks?",
    "a": "ReLU avoids the vanishing gradient problem for positive inputs, making it more effective in deep networks."
  },
  {
    "q": "What is the formula for parameter counting in Naive Bayes (binary case)?",
    "a": "num_params = (k - 1) + n × (k × (v - 1)); for boolean features and binary class, total = 1 + 2n."
  },
  {
    "q": "Why is a soft-margin SVM preferred in real-world applications?",
    "a": "Because real-world data is noisy, and soft margins allow some misclassifications while still learning a robust decision boundary."
  },
  {
    "q": "Which SVM kernel has the highest bias?",
    "a": "The linear kernel, because it assumes a simple decision boundary and may underfit complex data."
  },
  {
    "q": "What does the kernel trick enable in SVMs?",
    "a": "It allows a model to operate in a high-dimensional feature space without explicitly computing the transformation, using valid kernel functions."
  },
  {
    "q": "Which SVM kernel has the highest bias?",
    "a": "The linear kernel, because it assumes a simple decision boundary and may underfit complex data."
  },
  {
    "q": "How do you initialize the parameters in a Gaussian Mixture Model?",
    "a": "Assign equal priors to each cluster, randomly or heuristically initialize means, and use identity or data-based covariances."
  },
  {
    "q": "What are common convergence criteria in the EM algorithm for GMMs?",
    "a": "Set a maximum number of iterations or stop when log-likelihood or parameter changes fall below a threshold."
  },
  {
    "q": "What happens during the E-step and M-step of the EM algorithm?",
    "a": "E-step: compute responsibilities based on how likely each Gaussian generated a point. M-step: update the means, covariances, and priors using those responsibilities."
  },
  {
    "q": "If two GMM components start with the same mean but different covariances, what happens?",
    "a": "Different covariances cause different responsibilities, leading the component means to shift apart in the M-step."
  },
  {
  "q": "What happens when a GMM component collapses onto a single data point?",
  "a": "Its variance becomes nearly zero, making the component assign infinite likelihood to that point, which can destabilize the model."
  },
  {
    "q": "How do we calculate expected error for naive bayes classifier?",
    "a": "We sum across the losing probabilities."
  },
  {
    "q": "Redundant information and naive bayes? Do models like log. reg. struggle similarly?",
    "a": "Duplicate information distorts the posterior. Not necessarily; distinct weights and learns the dual effect with weighting across both."
  },
  {
    "q": "Why is Q-learning better than value learning?",
    "a": "Q-learning allows you to learn optimal actions directly."
  },
  {
  "q": "How does temporal difference learning work, and when would you use it?",
  "a": "TDL updates value estimates using the difference between successive predictions. It's used in online or episodic tasks where learning happens during interaction."
  },
  {
    "q": "What is policy iteration in reinforcement learning?",
    "a": "An algorithm that alternates between evaluating a policy and improving it until convergence to the optimal policy."
  },
  {
    "q": "What is Q-learning?",
    "a": "A model-free reinforcement learning algorithm that learns the optimal action-value function by iteratively updating Q-values using the Bellman equation."
  },
  {
    "q": "What is the difference between model-free and model-based reinforcement learning?",
    "a": "Model-free methods learn directly from experience without building a model of the environment, while model-based methods learn a model and plan actions using it."
  },
  {
    "q": "Under what conditions does Q-learning converge to the optimal policy?",
    "a": "If the agent explores enough, uses a learning rate that is small enough but not decaying too quickly, Q-learning converges to the optimal policy regardless of action selection."
  },
  {
    "q": "What do TDL, Q-learning, and model-free RL have in common? What makes them especially useful?",
    "a": "They learn from experience without needing the transition model T(s′|s,a); they update value estimates directly from interactions."
  },
  {
    "q": "What is the difference between supervised and unsupervised learning?",
    "a": "Supervised learning uses labeled data to learn mappings from inputs to outputs, commonly for classification and regression. Unsupervised learning uses unlabeled data to discover structure, such as clusters or latent variables."
  },
  {
    "q": "What is the curse of dimensionality?",
    "a": "In high-dimensional spaces, data becomes sparse and distances lose meaning, making learning difficult. With few examples, this can lead to overfitting, poor generalization, and inefficient computation."
  },
  {
    "q": "What is the purpose of discounting in reinforcement learning?",
    "a": "The discount factor in reinforcement learning reduces the value of future rewards, reflecting their uncertainty or lesser importance. It balances immediate vs. long-term gains in the agent's decision-making."
  },
  {
  "q": "What does the C parameter control in Support Vector Machines (SVM)?",
  "a": "The C parameter controls the trade-off between margin width and classification errors: a small C allows a wider margin with more misclassifications, while a large C prioritizes correct classification with a narrower margin."
  },
  {
  "q": "What's the best first step to reduce overfitting to noise in an SVM with an RBF kernel?",
  "a": "Decrease the C parameter — this softens the margin, allowing more classification errors and reducing sensitivity to noise."
  },
  {
    "q": "Why is decreasing γ (gamma) not the best first step to reduce overfitting in an SVM with an RBF kernel?",
    "a": "Decreasing γ increases the width of the RBF kernel, making the decision boundary smoother. While this reduces model complexity, it's less direct than decreasing C, which explicitly allows more margin violations and penalizes overfitting. Adjusting C typically yields more immediate regularization effects."
  },
  {
    "q": "Before fitting k-NN, which preprocessing step is most critical to prevent features like square footage from overwhelming the distance calculation?",
    "a": "Standardize each feature to mean 0 and variance 1 — this ensures that all features contribute equally to the distance computation."
  },
  {
    "q": "What does the decision boundary look like when using k = 1 in k-NN?",
    "a": "The decision boundary is highly jagged and prone to overfitting noise, since each training point becomes its own region of influence."
  },
  {
  "q": "What happens when you set k = N (number of training points) in k-NN classification?",
  "a": "The classifier predicts the majority class for all inputs, ignoring features entirely. This causes extreme underfitting because it acts like a constant classifier."
  },
  {
    "q": "Why shouldn't you pick the value of K in K-means clustering that gives the lowest SSD on training data?",
    "a": "Because increasing K always decreases SSD — the minimum SSD occurs when K = N (one point per cluster), which leads to overfitting and poor generalization."
  },
  {
    "q": "Which K-means initialization strategy is most sensitive to extreme outliers in the data?",
    "a": "Furthest-first (k-means++ style) initialization — it can select outliers as initial centroids, distorting clustering results."
  },
  {
    "q": "Decision trees: if an attribute perfectly predicts the label, what happens to the entropy after splitting on it?",
    "a": "Entropy drops to zero in each subset — the labels become pure, meaning there's no uncertainty left."
  },
  {
    "q": "Which reinforcement learning approach is best when the environment's dynamics (T) and rewards (R) are unknown, but sample trajectories can be collected?",
    "a": "Model-free Q-learning with ε-greedy exploration — it learns action-values directly from experience without modeling T or R, and ε-greedy ensures exploration of unseen actions."
  },
  {
    "q": "Which RL method uses the update rule Q(s,a) ← (1–α)Q(s,a) + α[r + γ·maxₐ′ Q(s′,a′)]?",
    "a": "Off-policy Q-learning — it updates Q-values using the best possible next action, regardless of the action actually taken."
  },
  {
    "q": "How does Q-learning differ from SARSA in terms of policy learning?",
    "a": "Q-learning is off-policy — it learns the value of the optimal policy regardless of the actions taken. SARSA is on-policy — it learns the value of the policy actually being followed, including its exploration behavior."
  },
  {
    "q": "When is model-based RL preferred over model-free Q-learning?",
    "a": "Model-based RL is preferred when sample efficiency is critical or a reliable model of transitions and rewards can be learned — it enables planning using learned dynamics. Q-learning is better when transitions and rewards are unknown and too complex to model accurately."
  },
  {
    "q": "Why can't value iteration be used when the transition model is unknown?",
    "a": "Value iteration requires full knowledge of the transition function T(s, a, s′) and reward function R. When these are unknown, model-free methods like Q-learning must be used instead."
  },
  {
    "q": "How does TD(0) differ from Q-learning in what it learns?",
    "a": "TD(0) learns value functions V(s) for states, while Q-learning learns action-values Q(s, a) for state–action pairs."
  },
  {
    "q": "Why is ε-greedy exploration commonly used with Q-learning?",
    "a": "ε-greedy balances exploration and exploitation by mostly choosing the best-known action but occasionally trying random ones — this is essential for Q-learning to learn optimal Q-values in unknown environments."
  },
  {
    "q": "Why is GlobalAveragePooling2D (GAP) sometimes preferred over Flatten → Dense for image classification?",
    "a": "Because GAP reduces the number of parameters and acts as a built-in regularizer, making it well-suited for smaller datasets. Reducing parameters lowers computational cost and helps prevent overfitting."
  },
  {
    "q": "If a model is overfitting, will increasing the number of parameters help generalization?",
    "a": "No — increasing model complexity typically worsens overfitting. Reducing parameters or using regularization is more effective in this case."
  },
  {
    "q": "Conceptually, how do we go from maximizing margin to minimizing norm of w?",
    "a": "Maximizing margin = minimizing 1/margin = minimizing ||w|| under the constraints that all instances are correctly classified."
  },
  {
    "q": "In SVMs, what are 2 interesting properties about the optimal w?",
    "a": "i) opt w orthogonal to H. ii) w^T.x = {-1,1} only for s.v.s iii)"
  },
  {
    "q": "In SVMs, one more interesting properties about optimal w?",
    "a": "formula for w can be represented as a lin. combo. of the data: w*=sum(alpha_i*y_i*x_i); alpha_i only non-zero for the s.v.s"
  },


];