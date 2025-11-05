import matplotlib.pyplot as plt

# Check class distribution in training data
class_counts = train_generator.classes
plt.hist(class_counts, bins=len(train_generator.class_indices))
plt.xticks(range(len(train_generator.class_indices)), list(train_generator.class_indices.keys()))
plt.title("Class Distribution in Training Data")
plt.show()
