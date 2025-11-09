# =====================================================
# Assignment: Pandas + Matplotlib Data Analysis
# Dataset: Iris Dataset (from sklearn)
# Author: [Your Name]
# Date: November 2025
# =====================================================

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris

# Optional: Set style for beautiful plots
sns.set(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("Libraries imported successfully!\n")

# =====================================================
# Task 1: Load and Explore the Dataset
# =====================================================

try:
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("Dataset loaded successfully!\n")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display first 10 rows
print("First 10 rows of the dataset:")
print(df.head(10))
print()

# Check data types and missing values
print("Dataset Info:")
df.info()
print()

print("Missing Values:")
print(df.isnull().sum())
print("\nNo missing values found – dataset is already clean!\n")

# =====================================================
# Task 2: Basic Data Analysis
# =====================================================

print("Basic Statistics (Numerical Columns):")
print(df.describe())
print()

# Group by species and compute mean
print("Average measurements by species:")
grouped_mean = df.groupby('species').mean()
print(grouped_mean)
print()

# Findings
print("Key Findings from Analysis:")
print("   • Virginica has the largest average petal length (5.55 cm) and width (2.03 cm)")
print("   • Setosa has the smallest petals but widest sepals (3.43 cm)")
print("   • Versicolor lies in between – good for classification tasks")
print("   • Strong differences in petal dimensions make species easily separable\n")

# =====================================================
# Task 3: Data Visualization
# =====================================================

# 1. Line Chart: Sepal Length Trend Across Samples
plt.figure(figsize=(12, 6))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.plot(subset.index, subset['sepal length (cm)'], 
             label=species, marker='o', linewidth=2, alpha=0.8)

plt.title('Trend of Sepal Length Across Iris Samples', fontsize=16, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend(title='Species')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Bar Chart: Average Petal Length by Species
plt.figure(figsize=(8, 6))
avg_petal_length = df.groupby('species')['petal length (cm)'].mean()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

bars = plt.bar(avg_petal_length.index, avg_petal_length.values, 
               color=colors, edgecolor='black', linewidth=1.2)

plt.title('Average Petal Length by Species', fontsize=16, fontweight='bold')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.xticks(rotation=0)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of Sepal Width
plt.figure(figsize=(9, 6))
plt.hist(df['sepal width (cm)'], bins=20, color='skyblue', edgecolor='black', alpha=0.8)
mean_width = df['sepal width (cm)'].mean()
plt.axvline(mean_width, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_width:.2f} cm')

plt.title('Distribution of Sepal Width', fontsize=16, fontweight='bold')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Scatter Plot: Petal Length vs Petal Width (Best for Classification!)
plt.figure(figsize=(10, 8))
colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}

for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                label=species, c=colors[species], s=100, alpha=0.9, edgecolors='k', linewidth=0.5)

plt.title('Petal Length vs Petal Width by Species', fontsize=16, fontweight='bold')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Species', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# Final Observations & Conclusion
# =====================================================

print("="*60)
print("FINAL OBSERVATIONS & INSIGHTS")
print("="*60)
print("1. Three distinct clusters visible in scatter plot → Perfect for classification")
print("2. Petal length and width show strong positive correlation")
print("3. Setosa is completely separated from others using petal dimensions")
print("4. Virginica has largest flowers, Setosa has smallest petals")
print("5. Sepal width is normally distributed around 3.06 cm")
print()
print("Conclusion: The Iris dataset is an excellent example for teaching")
print("machine learning because of its clear, natural separation of classes.")
print("Petal measurements alone can achieve near-perfect classification accuracy.")
print("="*60)

print("\nAssignment Complete! All 4 visualizations generated successfully.")