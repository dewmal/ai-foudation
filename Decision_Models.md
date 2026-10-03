**AI Foundation Program**

# Building a Simple Decision Model From Scratch

*A step-by-step guide with Python functions and NumPy*

_From weighted scores to a small trainable binary classifier_

$$
INPUT  →  MODEL  →  SCORE OR PROBABILITY  →  DECISION
$$

### Learning goal

Build a decision model in stages: represent features as arrays, combine them with weights, apply a threshold, then learn weights from labelled examples using logistic regression. Every calculation is implemented with Python functions and NumPy.

No scikit-learn, TensorFlow, or PyTorch is used.

## 1. Introduction

A decision model evaluates information and arrives at a conclusion. It receives input values, processes them, and produces a decision. We will use student results to make the ideas concrete.

$$
Input  →  Mathematical model  →  Score  →  Decision
$$

The three inputs in our running example are exam mark, assignment mark, and attendance. In machine learning, input measurements like these are called features.

### What we will build

- A manually weighted score and a pass/fail threshold.

- NumPy functions that handle one student or many students.

- A logistic-regression model that learns weights from labelled examples.

## 2. Installing and Importing NumPy

NumPy is a Python library for numerical computation. Its arrays and vector operations let us express calculations compactly.

```bash
pip install numpy
```

```python
import numpy as np
```

We will use NumPy arrays, dot products, matrix operations, and mathematical functions throughout the guide.

## 3. Representing Input Data

For one student, the feature values are:

| Feature | Value |
| --- | --- |
| Exam | 70 |
| Assignment | 65 |
| Attendance | 85 |

Store the values in a NumPy array. Keep the feature order consistent everywhere: exam, assignment, attendance.

```python
import numpy as np

student = np.array([
    70,
    65,
    85
])

print(student)
```

$$
X = [70, 65, 85]ᵀ
$$

A single student is represented by a feature vector. Later, a dataset will use one row per student.

## 4. Adding Weights

Features can contribute different amounts to a score. For this first model, we choose the weights ourselves:

- Exam: 60%

- Assignment: 30%

- Attendance: 10%

```python
weights = np.array([
    0.6,
    0.3,
    0.1
])
```

$$
z = x₁w₁ + x₂w₂ + x₃w₃
$$

For this student, the calculation is:

$$
z = 70(0.6) plus 65(0.3) plus 85(0.1) = 42 + 19.5 + 8.5 = 70
$$

## 5. Using NumPy Dot Product

The dot product multiplies matching entries and adds the results. It performs the weighted sum in one operation.

```python
student = np.array([70, 65, 85])
weights = np.array([0.6, 0.3, 0.1])

score = np.dot(student, weights)
print(score)
```

```text
70.0
```

Here, np.dot(student, weights) is the compact form of the weighted-sum equation. The same operation appears in many machine-learning models and neural-network layers.

## 6. Creating a Decision Function

A function gives the score calculation a name so that we can reuse it with different students and weights.

```python
def calculate_score(features, weights):
    score = np.dot(features, weights)
    return score
```

```python
student = np.array([70, 65, 85])
weights = np.array([0.6, 0.3, 0.1])

score = calculate_score(student, weights)
print(score)
```

$$
Inputs: X (features), W (weights)     Output: z = X · W
$$

## 7. Adding a Threshold

A score becomes a category when we define a threshold. In this example, a score of 50 or above means Pass; a lower score means Fail.

$$
Decision = 1 (Pass) when score ≥ 50; otherwise 0 (Fail)
$$

```python
def make_decision(score, threshold=50):
    if score >= threshold:
        return 1
    return 0
```

```python
def calculate_score(features, weights):
    return np.dot(features, weights)

def make_decision(score, threshold=50):
    if score >= threshold:
        return 1
    return 0

student = np.array([70, 65, 85])
weights = np.array([0.6, 0.3, 0.1])
score = calculate_score(student, weights)
decision = make_decision(score)

print("Score:", score)
print("Decision:", decision)
```

```text
Score: 70.0
Decision: 1
```

The decision value 1 represents Pass. This is a hand-designed scoring rule: we supplied both the weights and threshold.

## 8. Combining Scoring and Decision

We can place both steps in one prediction function.

```python
def predict(features, weights, threshold=50):
    score = np.dot(features, weights)
    if score >= threshold:
        return 1
    return 0

student = np.array([70, 65, 85])
weights = np.array([0.6, 0.3, 0.1])
prediction = predict(student, weights)
print(prediction)
```

$$
X  →  XW  →  Threshold  →  Decision
$$

## 9. Adding Bias

A bias shifts the score before the decision. The model score becomes:

$$
z = XW plus b
$$

- X: input features

- W: weights

- b: bias

- z: model score

```python
def calculate_score(features, weights, bias):
    return np.dot(features, weights) + bias

student = np.array([70, 65, 85])
weights = np.array([0.6, 0.3, 0.1])
bias = -10
score = calculate_score(student, weights, bias)
print(score)
```

$$
z = 70(0.6) plus 65(0.3) plus 85(0.1) minus 10 = 60
$$

Bias changes where the model’s decision boundary falls.

## 10. Working With Multiple Students

A dataset stores one student per row and one feature per column.

```python
X = np.array([
    [70, 65, 85],
    [40, 45, 60],
    [90, 80, 95],
    [50, 55, 70]
])
```

$$
Rows = students     Columns = exam, assignment, attendance
$$

## 11. Calculating Scores for Multiple Students

NumPy applies the same weights to every row in one matrix-vector operation.

```python
weights = np.array([0.6, 0.3, 0.1])
scores = np.dot(X, weights)
print(scores)
```

$$
Z = XW
$$

Instead of writing a loop for each student, NumPy computes all four scores together.

## 12. Making Multiple Decisions

A comparison creates one Boolean result for each score. Convert the Booleans to integers when you want 1/0 labels.

```text
decisions = scores >= 50
print(decisions)
```

```text
[ True False  True  True]
```

```text
decisions = (scores >= 50).astype(int)
print(decisions)
```

```text
[1 0 1 1]
```

Here, 1 means Pass and 0 means Fail.

## 13. A Prediction Function for Multiple Records

```python
def predict(X, weights, bias=0, threshold=50):
    scores = np.dot(X, weights) + bias
    predictions = (scores >= threshold).astype(int)
    return predictions

X = np.array([
    [70, 65, 85],
    [40, 45, 60],
    [90, 80, 95],
    [50, 55, 70]
])
weights = np.array([0.6, 0.3, 0.1])
predictions = predict(X, weights)
print(predictions)
```

## 14. The Main Limitation

So far, we selected the weights ourselves. Machine learning changes this: the model estimates useful weights and bias from examples that include the correct outcome.

```python
X = np.array([
    [80, 75, 90],
    [30, 40, 50],
    [70, 65, 85],
    [45, 40, 55],
    [90, 85, 95],
    [35, 45, 40]
])
y = np.array([1, 0, 1, 0, 1, 0])
```

$$
X = input features     y = correct answers (labels)
$$

The training objective is to find values for W and b that make the predictions agree with the labelled examples.

## 15. Why We Need a Probability Function

A linear score z = XW plus b can be any real number. For binary classification, we map it to a value from 0 to 1 with the sigmoid function.

$$
σ(z) = 1 / (1 plus e⁻ᶻ)
$$

Large positive scores map close to 1; large negative scores map close to 0. We can interpret the result as the model’s estimated probability of class 1.

## 16. Implementing Sigmoid From Scratch

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid(0))
```

```text
0.5
```

## 17. The Forward Pass

A forward pass sends inputs through the model to produce probabilities. First calculate the linear score, then apply sigmoid.

$$
Step 1: z = XW plus b     Step 2: ŷ = σ(z)
$$

```python
def forward(X, weights, bias):
    z = np.dot(X, weights) + bias
    probabilities = sigmoid(z)
    return probabilities
```

The symbol ŷ (“y-hat”) represents the model output.

## 18. Example Forward Pass

This example uses feature values between 0 and 1. Scaling the marks consistently helps keep the inputs on a comparable range during training.

```python
X = np.array([
    [0.8, 0.75, 0.9],
    [0.3, 0.4, 0.5]
])
weights = np.array([0.2, 0.4, 0.1])
bias = 0
probabilities = forward(X, weights, bias)
print(probabilities)
```

## 19. Converting Probabilities Into Decisions

Use 0.5 as the decision threshold: probability at least 0.5 maps to class 1; a lower probability maps to class 0.

$$
ŷ ≥ 0.5 → 1     ŷ < 0.5 → 0
$$

```python
def predict(X, weights, bias):
    probabilities = forward(X, weights, bias)
    predictions = (probabilities >= 0.5).astype(int)
    return predictions
```

This probability threshold is a different decision stage from the earlier score threshold of 50. The first model scores marks on a 0–100 scale; sigmoid maps its input to a 0–1 probability. In a trained model, use the same feature scaling during training and prediction.

## 20. Measuring Model Error

Training needs a measure of how far predictions are from the correct labels. Binary cross-entropy is a common loss for binary classification.

$$
L = −(1/m) Σᵢ [ yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ) ]
$$

- y: correct label

- ŷ: predicted probability

- m: number of training examples

Training aims to minimize the loss.

## 21. Implementing the Loss

```python
def binary_cross_entropy(y, y_hat):
    epsilon = 1e-8
    loss = -np.mean(
        y * np.log(y_hat + epsilon)
        + (1 - y) * np.log(1 - y_hat + epsilon)
    )
    return loss
```

The tiny epsilon keeps the logarithm away from zero and avoids log(0).

## 22. Learning the Weights

The model adjusts parameters in response to its error. Gradients indicate the direction and size of the change that reduces loss.

$$
dW = (1/m) Xᵀ(ŷ − y)       db = (1/m) Σ(ŷ − y)
$$

## 23. Calculating Gradients

```python
def calculate_gradients(X, y, y_hat):
    m = len(y)
    error = y_hat - y
    dw = np.dot(X.T, error) / m
    db = np.sum(error) / m
    return dw, db
```

## 24. Updating the Parameters

The learning rate α controls the size of each update.

$$
W ← W − αdW       b ← b − αdb
$$

```python
def update_parameters(weights, bias, dw, db, learning_rate):
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db
    return weights, bias
```

## 25. The Training Function

Training repeats the same cycle: produce probabilities, measure loss, calculate gradients, and update the parameters. An epoch is one full pass through the training examples.

```python
def train(X, y, learning_rate=0.1, epochs=1000):
    number_of_features = X.shape[1]
    weights = np.zeros(number_of_features)
    bias = 0.0

    for epoch in range(epochs):
        y_hat = forward(X, weights, bias)
        loss = binary_cross_entropy(y, y_hat)
        dw, db = calculate_gradients(X, y, y_hat)
        weights, bias = update_parameters(
            weights, bias, dw, db, learning_rate
        )

        if epoch % 100 == 0:
            print("Epoch:", epoch, "Loss:", loss)

    return weights, bias
```

## 26. Complete Training Example

The following complete implementation combines the functions into a small logistic-regression model. Each helper has one job, which makes the training process easier to inspect.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias)

def binary_cross_entropy(y, y_hat):
    epsilon = 1e-8
    return -np.mean(
        y * np.log(y_hat + epsilon)
        + (1 - y) * np.log(1 - y_hat + epsilon)
    )

def calculate_gradients(X, y, y_hat):
    m = len(y)
    error = y_hat - y
    dw = np.dot(X.T, error) / m
    db = np.sum(error) / m
    return dw, db

def update_parameters(weights, bias, dw, db, learning_rate):
    weights -= learning_rate * dw
    bias -= learning_rate * db
    return weights, bias

def train(X, y, learning_rate=0.1, epochs=1000):
    weights = np.zeros(X.shape[1])
    bias = 0.0
    for epoch in range(epochs):
        y_hat = forward(X, weights, bias)
        loss = binary_cross_entropy(y, y_hat)
        dw, db = calculate_gradients(X, y, y_hat)
        weights, bias = update_parameters(
            weights, bias, dw, db, learning_rate
        )
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return weights, bias

def predict(X, weights, bias):
    probabilities = forward(X, weights, bias)
    return (probabilities >= 0.5).astype(int)
```

## 27. Prepare Training Data

Normalize percentage marks to values from 0 to 1 by dividing by 100. Keep feature order fixed as exam, assignment, attendance.

```python
X = np.array([
    [0.80, 0.75, 0.90],
    [0.30, 0.40, 0.50],
    [0.70, 0.65, 0.85],
    [0.45, 0.40, 0.55],
    [0.90, 0.85, 0.95],
    [0.35, 0.45, 0.40],
    [0.75, 0.70, 0.80],
    [0.25, 0.35, 0.45]
])
y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
```

## 28. Train the Model

```text
weights, bias = train(
    X, y, learning_rate=0.5, epochs=2000
)

print("Learned weights:")
print(weights)
print("Learned bias:")
print(bias)
```

Unlike the first model, these parameters are learned from examples rather than entered as fixed percentages.

## 29. Make a New Prediction

A new student has exam 72%, assignment 68%, and attendance 82%. Normalize those values using the same scale used for training.

```python
new_student = np.array([[0.72, 0.68, 0.82]])

probability = forward(new_student, weights, bias)
prediction = predict(new_student, weights, bias)

print("Probability:", probability)
print("Prediction:", prediction)
```

A prediction of [1] means Pass; [0] means Fail. The probability provides more detail about the model output before applying the decision threshold.

## 30. What Happens During Training

$$
Forward pass  →  Loss  →  Gradient  →  Parameter update  →  Repeat
$$

The model sends the inputs through XW + b and sigmoid, compares its predicted probabilities with the known labels, calculates how the parameters contributed to the error, and adjusts the weights and bias. Repeating this cycle is the core training idea.

## 31. Role of Each Function

| Function | Purpose |
| --- | --- |
| sigmoid() | Maps raw scores to values between 0 and 1. |
| forward() | Calculates XW + b and applies sigmoid. |
| binary_cross_entropy() | Measures the mismatch between labels and probabilities. |
| calculate_gradients() | Finds how weights and bias should change. |
| update_parameters() | Applies the learning-rate adjustment. |
| train() | Repeats forward pass, loss, gradients, and updates. |
| predict() | Converts probabilities into 0/1 decisions. |

## Key Takeaways

- A decision model maps input features to a score, probability, or category.

- A dot product combines features with weights.

- A threshold turns a numeric result into a decision.

- A trainable model learns weights and bias from labelled examples.

- NumPy arrays make it practical to calculate many examples together.

$$
INPUTS  →  FORWARD PASS  →  LOSS  →  GRADIENTS  →  UPDATE  →  PREDICTION
$$

## 32. Decision Trees

A decision tree reaches an outcome by asking a sequence of questions. Each answer selects a branch, and the final branch ends at a decision. Unlike the weighted model earlier in this guide, a tree does not need to combine every feature into one score.

### How to Build a Decision Tree

Start with labelled examples and turn the prediction task into a sequence of questions. For the student example, the target is Pass or Fail and the possible features include exam mark and attendance.

1. Define the target decision. Choose the outcome the tree should predict, such as Pass or Fail. These are the class labels.

1. Gather labelled examples. For each past student, record the features (exam mark and attendance) and the known outcome.

1. Choose a question that splits the examples. For this teaching example, ask whether the exam mark is at least 50. A Yes/No question divides the records into two groups.

1. Repeat the process within each branch. For students whose exam mark is at least 50, ask whether attendance is at least 75%. Each new question creates another split.

1. Stop and assign an outcome at each leaf. When a branch is decisive or no useful question remains, label its endpoint Pass or Fail.

1. Review the tree on examples it did not use to build it. Check whether its decisions are sensible on new cases. A tree that keeps splitting can memorize its training examples, so limit its depth when needed.

When a computer learns the tree from data, it compares candidate questions and selects splits that separate the labels well. Measures such as Gini impurity or information gain help score those splits. In the hand-built example below, we choose the questions and thresholds ourselves.

Example: decide whether a student passes using an exam threshold of 50 and an attendance threshold of 75%.

```text
Exam mark >= 50?
|-- No: Fail
`-- Yes: Attendance >= 75%?
    |-- No: Fail
    `-- Yes: Pass
```

### How to read the tree

- Start at the top question, called the root.

- Follow the branch that matches the student’s answer. A “No” answer to either question leads to Fail.

- A final outcome such as Pass or Fail is called a leaf.

This is a hand-written decision tree. In machine learning, a tree-learning algorithm can choose useful questions and split points from labelled training data. The shared idea is that the model processes input information and produces a decision; the path through a tree is simply a different way to do that.


## 33. Using Decision Trees in Practice

A decision tree is useful when a prediction can be explained as a path of questions. It can be hand-built from explicit policy rules, or learned from labelled examples by a tree-learning algorithm. In either case, each record starts at the root, follows the branch matching its feature values, and ends at a leaf containing a class or numeric prediction.

### Decision model and decision tree

A decision model is the broad idea: a system that uses inputs and a method to produce an outcome. A weighted score with a threshold is one kind of decision model. A decision tree is another kind. The weighted model combines several inputs in one calculation, while a tree asks one question at a time and selects a branch. Trees often make the reasoning easier to follow; weighted models can express many small contributions compactly. A decision tree is therefore a type of decision model, not a separate category from all decision models.

| Aspect | Weighted score model | Decision tree |
| --- | --- | --- |
| How it decides | Multiplies features by weights, adds them, then applies a threshold or output rule. | Tests feature conditions in sequence and follows branches to a leaf. |
| Example rule | Score = 0.6 Exam + 0.3 Assignment + 0.1 Attendance; Pass if score is at least 50. | If Exam is below 50, Fail; otherwise check Attendance. |
| What is learned or specified | Weights may be chosen by a person or learned from examples. | Questions and split values may be written by a person or learned from examples. |
| Explanation | Shows a combined score and each feature contribution. | Shows the exact sequence of questions that led to the answer. |
| Typical strength | Smoothly combines many signals and is compact. | Easy to explain when the tree is small and rules are meaningful. |
| Main caution | Weights and scale choices can be hard to interpret; the threshold can hide trade-offs. | Deep trees can become long and memorize training records. |

### How to use a decision tree to make a prediction

1. Prepare one case as a record with the same feature names and units used to define the tree. For example: Exam = 70 and Attendance = 85%.
2. Start at the root question. Do not skip a question just because another feature seems more important.
3. Evaluate the condition using the case value. For Exam >= 50, 70 >= 50 is True, so follow the Yes branch.
4. Continue at the next question on that branch. Attendance >= 75% is True for 85%, so follow Yes again.
5. Return the leaf prediction, Pass. The path can be stated as: Exam >= 50 (Yes), Attendance >= 75% (Yes), therefore Pass.
6. Apply the same path for every new case. Keep missing or invalid values explicit; define a safe rule for them instead of silently treating them as zero.

A prediction is not automatically a certainty. If a trained classification tree stores class proportions at leaves, a leaf may provide an estimated probability as well as the most common class. That probability is an estimate based on the examples reaching the leaf.

### Representing and running a small tree with Python

A small tree can be represented with nested dictionaries. Each internal node stores a feature, a comparison value, and its Yes and No branches. A leaf stores a prediction. This example implements the already-defined student tree; it does not learn the tree from data.

```python
tree = {
    "feature": "exam",
    "threshold": 50,
    "yes": {
        "feature": "attendance",
        "threshold": 75,
        "yes": {"prediction": "Pass"},
        "no": {"prediction": "Fail"},
    },
    "no": {"prediction": "Fail"},
}

def predict_one(node, student):
    if "prediction" in node:
        return node["prediction"]

    value = student[node["feature"]]
    if value >= node["threshold"]:
        return predict_one(node["yes"], student)
    return predict_one(node["no"], student)

student = {"exam": 70, "attendance": 85}
print(predict_one(tree, student))  # Pass
```

The function checks for a leaf first. If the current node is a question, it reads the matching value and recursively follows one branch. Recursion ends when the function reaches a leaf. For a larger application, validate that required features are present and that values use the expected units.

### How a learning algorithm builds a tree

To learn a tree, give the algorithm a training table containing input features and known target labels. At each node, the algorithm considers candidate questions, such as Exam >= 50 or Attendance >= 75%. It estimates how well each question separates the target classes, selects a useful split, and repeats the process separately for the resulting groups.

For a classification task, one common measure is Gini impurity. If a group contains classes with proportions p_1, ..., p_k, its Gini impurity is:

$$
Gini = 1 - Σᵢ pᵢ²
$$

A group with only one class has Gini 0, meaning it is pure. A candidate split is judged by the weighted impurity of its child groups; a useful split makes this value lower than the parent impurity. Information gain based on entropy is another common split criterion. For numeric features, the learner tests possible thresholds; for categorical features, it tests category groupings according to the algorithm.

A simple training loop is:

1. Begin with all training rows at the root.
2. List candidate feature questions and thresholds for those rows.
3. Calculate the split quality, such as weighted Gini impurity.
4. Choose the best valid split and send rows to the corresponding child nodes.
5. Repeat for each child until a stopping rule is met.
6. Assign each leaf the most common class among its training rows; for regression, assign a numeric summary such as the mean.
7. Evaluate the finished tree on separate validation data, then reduce its size if it does not generalize well.

A hand-written tree uses questions selected by a person. A learned tree uses an algorithm to search for them. The dictionary example above teaches prediction by a fixed tree; it is not a full tree-training algorithm.

### Controlling tree size and checking quality

A very deep tree can create a special path for nearly every training example. It may achieve excellent results on the examples it saw while making poor predictions on new ones. Limit this risk by setting a maximum depth, requiring a minimum number of examples in a leaf, or pruning branches that do not improve validation performance.

Split the data into training and evaluation sets before tuning the tree. Build the tree using training data, choose size controls using validation data, and report final performance on data held aside for evaluation. For classification, inspect accuracy along with measures such as precision, recall, and a confusion matrix, especially when one outcome is rare or mistakes have unequal costs. Check important subgroups too, because overall performance can hide uneven errors.

### Decision-tree use cases

Decision trees are useful when the decision can be expressed through understandable conditions, or when interactions between features matter. Examples include:

- **Student support:** flag students who may need help using attendance, assignment completion, and assessment results. Treat the output as a prompt for support, not as an automatic final judgment.
- **Loan triage:** route applications for review based on documented eligibility criteria and financial information. High-impact lending decisions require careful validation, fairness review, and appropriate human oversight.
- **Equipment maintenance:** estimate whether a machine needs inspection from temperature, vibration, operating hours, and fault history.
- **Customer support routing:** direct a request to a team using issue type, account status, and urgency.
- **Medical decision support:** organize risk indicators to help clinicians review cases. The tree should support qualified professionals and be validated carefully before use.
- **Quality control:** classify products for pass, rework, or inspection based on measured properties.

A tree is a good fit when people need to inspect the route to a result and the number of conditions can stay manageable. If relationships are subtle, the data is noisy, or accuracy is the main priority, compare it with other methods and validate each on representative unseen data.

### Practical checklist

- Define the prediction target and what each label means.
- Use features available at the moment the decision must be made; avoid information that leaks the answer.
- Keep feature definitions, units, and missing-value handling consistent.
- Separate training, validation, and final evaluation data.
- Set depth or leaf-size limits and inspect whether branches make sense.
- Measure errors that matter for the real task, including subgroup differences.
- Document the tree, its intended use, and when a person should review the outcome.
