import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def read_data(file_path):
    """Reads the CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def display_info(df):
    """Displays information about the DataFrame."""
    print(df)
    print('-----------------------------------')
    info = df.info()
    description = df.describe(include='all')
    print('-----------------------------------')
    print(description)

def check_missing_values(df):
    """Checks for missing and NaN values in the DataFrame."""
    print('Checking existence of null values')
    print(df.isnull().sum())
    print('-----------------------------------')
    print('Checking existence of NaN values')
    print(df.isna().sum())

def display_duplicates(df):
    """Displays duplicate rows in the DataFrame."""
    duplicate_rows = df[df.duplicated()]
    print('-----------------------------------')
    print("Duplicate Rows except first occurrence:")
    print(duplicate_rows)
    print('-----------------------------------')
    print("\nNumber of duplicate rows:", len(duplicate_rows))

def plot_pie_chart(df):
    """Plots a pie chart for the distribution of Heart Disease."""
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 10))
    df['HeartDisease'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No Heart Disease', 'Heart Disease'], startangle=90)
    plt.title('Balance of Heart Disease in the Dataset')
    plt.savefig('balance-heart-disease.png')

def plot_histograms(df, numerical_vars):
    """Plots histograms for numerical variables."""
    num_vars = len(numerical_vars)
    num_rows = (num_vars + 1) // 2
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(16, 4 * num_rows))
    sns.set(style='whitegrid')

    for i, var in enumerate(numerical_vars):
        row, col = divmod(i, 2)
        plot = sns.histplot(data=df, x=var, hue='HeartDisease', multiple='stack', kde=False, ax=axes[row, col], element='step', common_norm=False)
        axes[row, col].set_title(f'Histogram for {var} with Heart Disease')
        axes[row, col].set_xlabel(var)
        axes[row, col].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('numerical-hist-plots.png')

def plot_scatter_plots(df, numerical_vars):
    """Plots scatter plots for numerical variables against HeartDisease."""
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
    axes = axes.flatten()

    for i, column in enumerate(numerical_vars):
        sns.scatterplot(data=df, x=df.index, y=column, hue=df['HeartDisease'], ax=axes[i])
        axes[i].set_title(f'{column} vs. HeartDisease')
        axes[i].set_xlabel('Index')

    plt.tight_layout()
    plt.savefig('numerical-scatterplots.png')

def plot_categorical_pie_charts(df, categories):
    """Plots pie charts for categorical variables."""
    sns.set(style="whitegrid")
    fig, subplots = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

    for i, category in enumerate(categories):
        data = df[df['HeartDisease'] == 1][category].value_counts()
        percentages = [(count / sum(data)) * 100 for count in data]

        sns.color_palette("pastel")
        sns.set_palette("pastel")
        subplots[i // 2, i % 2].pie(percentages, labels=data.index, autopct='%1.1f%%', startangle=90, explode=[0.1] * len(data))
        subplots[i // 2, i % 2].set_title(f'Distribution of {category} in Positive Heart Disease Cases')

    main_title = 'Categorical Variables in Positive Heart Disease Cases'
    plt.suptitle(main_title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('categorical-pie-charts.png')

def plot_sex_distribution(df):
    """Plots the distribution of males and females."""
    sex_distribution = df['Sex'].value_counts()
    print("Population of Males and Females:")
    print(sex_distribution)
    

    plt.figure(figsize=(10, 10))
    
    sex_distribution.plot(kind='bar', color=['blue', 'pink'])
    plt.title('Distribution of Males and Females in the Dataset')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    plt.savefig('sex-participant-imbalance.png')

def preprocess_data(df, numerical_vars):
    """Preprocesses the data by normalizing numerical variables and converting categorical variables."""
    normalized_df = df.copy()
    scaler = MinMaxScaler()
    normalized_df[numerical_vars] = scaler.fit_transform(normalized_df[numerical_vars])

    # Convert categorical variables to numerical
    normalized_df['Sex'] = normalized_df['Sex'].replace({'M': 0, 'F': 1}).astype(np.uint8)
    normalized_df['ChestPainType'] = normalized_df['ChestPainType'].replace({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}).astype(np.uint8)
    normalized_df['RestingECG'] = normalized_df['RestingECG'].replace({'Normal': 0, 'ST': 1, 'LVH': 2}).astype(np.uint8)
    normalized_df['ST_Slope'] = normalized_df['ST_Slope'].replace({'Up': 0, 'Flat': 1, 'Down': 2}).astype(np.uint8)
    normalized_df['ExerciseAngina'] = normalized_df['ExerciseAngina'].replace({'N': 0, 'Y': 1}).astype(np.uint8)

    return normalized_df

def save_to_csv(df, output_file):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(output_file, index=False)

def main(file_path):
    heart_df = read_data(file_path)

    display_info(heart_df)
    check_missing_values(heart_df)
    display_duplicates(heart_df)

    plot_pie_chart(heart_df)

    numerical_vars = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    plot_histograms(heart_df, numerical_vars)
    plot_scatter_plots(heart_df, numerical_vars)

    categories = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    plot_categorical_pie_charts(heart_df, categories)

    plot_sex_distribution(heart_df)

    normalized_heart_df = preprocess_data(heart_df, numerical_vars)

    # Save the final DataFrame to a CSV file
    save_to_csv(normalized_heart_df, 'normalized_heart_data.csv')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <heart_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)
