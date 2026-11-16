

<div align="center">

# Dual-Individual Genetic Algorithm with L2 Regularization and Dropout for Enhanced Model Generalization

</div>

<div align="center">

**Authors:** Tran Thuy Nga Truong¹ and Jooyong Kim*

**Affiliations:**

**¹** Department of Smart Wearables Engineering
**\*** Department of Materials Science and Engineering
**Institution:** Soongsil University, Seoul 156-743, Korea
**Correspondence:** jykim@ssu.ac.kr | Tel: +82-10-8720-0631

</div>

# Abstract:

In our previous study, we proposed the Dual-Individual Genetic Algorithm (DIGA), an evolutionary framework that operates with only two individuals: a Leader and a Follower. This design simplifies the evolutionary process while maintaining genetic diversity and preventing premature convergence. To further enhance generalization and mitigate overfitting, we integrate two complementary regularization techniques, L2 regularization and Dropout, directly into the fitness evaluation and forward propagation stages of DIGA. L2 regularization penalizes large weights, while Dropout randomly deactivates neurons during training, preventing complex co-adaptation among features. We conduct extensive experiments on a binary classification task with 12,288 input features, systematically evaluating the individual and combined effects of L2 and Dropout across a wide range of hyperparameter configurations (λ from 0.0 to 1e-6, and keep_prob from 0.5 to 1.0). We ran 30 seeds and computed the mean and standard deviation of the metrics across three models: L2 alone, dropout alone, and L2 combined with dropout. The results show that both L2 and dropout improve generalization on the training and test sets. Although combining L2 with dropout does not yield a dramatic improvement over L2 or dropout alone, it still generalizes better than vanilla DIGA. These findings confirm that integrating regularization techniques into the evolutionary search process effectively balances accuracy, generalization, and stability. Future work will focus on accelerating convergence speed and scaling DIGA to handle large-scale datasets through batch-wise training strategies. (The source code is available in the supplement.)



# How to run

The program can be run directly on Google Colab or Visual Studio Code (Version: 1.145.2).



