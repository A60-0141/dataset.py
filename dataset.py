import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore_data():
    # Load the Iris dataset from sklearn
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target

    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(iris_df.head())
    
    # Display the structure of the dataset
    print("\nDataset Info:")
    print(iris_df.info())
    
    # Check for missing values
    print("\nMissing values in the dataset:")
    print(iris_df.isnull().sum())

    return iris_df

# Task 2: Basic Data Analysis
def basic_data_analysis(iris_df):
    # Basic statistics of numerical columns
    print("\nBasic Statistics of the dataset:")
    print(iris_df.describe())
    
    # Group by 'species' and calculate the mean for each numerical column
    grouped = iris_df.groupby('species').mean()
    print("\nGrouped by species (mean values):")
    print(grouped)

# Task 3: Data Visualization
def data_visualization(iris_df):
    # Line chart (Simulated Time-Series)
    iris_df['sepal length (cm)'] = iris_df['sepal length (cm)'].cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(iris_df.index, iris_df['sepal length (cm)'], label="Sepal Length", color='blue')
    plt.title("Trend of Sepal Length Over Time")
    plt.xlabel("Index")
    plt.ylabel("Sepal Length (cm)")
    plt.legend()
    plt.show()

    # Bar chart (Average Petal Length per Species)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='species', y='petal length (cm)', data=iris_df)
    plt.title("Average Petal Length per Species")
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.show()

    # Histogram (Sepal Length Distribution)
    plt.figure(figsize=(10, 6))
    sns.histplot(iris_df['sepal length (cm)'], bins=15, kde=True)
    plt.title("Distribution of Sepal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Frequency")
    plt.show()

    # Scatter plot (Sepal Length vs Petal Length)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', data=iris_df)
    plt.title("Relationship Between Sepal Length and Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.show()

# Main function to call all tasks
def main():
    iris_df = load_and_explore_data()
    basic_data_analysis(iris_df)
    data_visualization(iris_df)

if __name__ == "__main__":
    main()
