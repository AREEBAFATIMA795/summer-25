import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Load features
df = pd.read_csv('features.csv')
X = df.drop('label', axis=1).values
y = df['label'].values
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Load your dataset
df = pd.read_csv('features.csv')
X = df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values.astype(np.float32).reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define input size
input_size = X.shape[1]

# Initialize weights and biases
W1 = tf.Variable(tf.random.normal([input_size, 16], stddev=0.1))
b1 = tf.Variable(tf.zeros([16]))

W2 = tf.Variable(tf.random.normal([16, 8], stddev=0.1))
b2 = tf.Variable(tf.zeros([8]))

W3 = tf.Variable(tf.random.normal([8, 1], stddev=0.1))
b3 = tf.Variable(tf.zeros([1]))

# Activation functions
def relu(x):
    return tf.maximum(0.0, x)

def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

# Forward pass
def forward(X):
    z1 = tf.matmul(X, W1) + b1
    a1 = relu(z1)
    z2 = tf.matmul(a1, W2) + b2
    a2 = relu(z2)
    z3 = tf.matmul(a2, W3) + b3
    output = sigmoid(z3)
    return output

# Binary cross-entropy loss
def loss_fn(y_true, y_pred):
    eps = 1e-7
    return -tf.reduce_mean(y_true * tf.math.log(y_pred + eps) + (1 - y_true) * tf.math.log(1 - y_pred + eps))

# Optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = forward(X_train)
        loss = loss_fn(y_train, y_pred)

    gradients = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])
    optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2, W3, b3]))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.numpy():.4f}")

# Evaluation
y_test_pred = forward(X_test).numpy()
y_test_labels = (y_test_pred > 0.5).astype(int)
acc = accuracy_score(y_test, y_test_labels)
print(f"Test Accuracy: {acc:.4f}")

# Get predicted probabilities and labels
y_pred_prob = forward(X_test).numpy().flatten()
y_pred_labels = (y_pred_prob > 0.5).astype(int)

# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred_labels)

# Plot Confusion Matrix
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Compute ROC and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc = roc_auc_score(y_test, y_pred_prob)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()