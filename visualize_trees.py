import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load model
with open('improved_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load training data for feature names
train = pd.read_csv('data/train.csv')
X_train = train.drop(columns=['nct_id', 'success'])

# Visualize the tree (limit depth for readability)
plt.figure(figsize=(25, 15))
plot_tree(model,  # ← Direct model, NO .estimators_[0]
          feature_names=X_train.columns,
          class_names=['Failure', 'Success'],
          filled=True,
          rounded=True,
          fontsize=9,
          max_depth=4)  # Show top 4 levels

plt.title('Decision Tree Visualization (Top 4 Levels)', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/decision_tree_better.png', dpi=300, bbox_inches='tight')
# plt.show()