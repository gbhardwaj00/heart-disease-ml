import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def read_data(file_path):
    """Reads the CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def print_df(df):
    """Displays information about the DataFrame."""
    print(df)

def plot_histogram_normalzied(df):
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    # Plot histograms for each variable against HeartDisease
    for i, column in enumerate(df.columns[:-1]):  # Exclude HeartDisease column
        sns.histplot(data=df, x=column, hue="HeartDisease", multiple="stack", ax=axes[i], palette="Set2", kde=False)
        axes[i].set_title(f'{column} vs. HeartDisease')
        axes[i].set_xlabel(column)

    plt.tight_layout()
    plt.savefig('histograms-normalized.png')

def plot_correlation_heatmap(df):
    correlation_matrix = df.corr()
    # Plotting the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix of Heart Failure Dataset")
    plt.savefig('correlation-matrix-normalized.png')

def print_best_features(df):
    # Separating the features and the target variable
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    bestfeatures = SelectKBest(score_func=chi2, k=11)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Attributes','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))

def gradient_boosting_clf(X_train, X_valid, y_train, y_valid):
    GBCmodel = GradientBoostingClassifier(n_estimators=50, max_depth=2, min_samples_leaf=0.1)
    GBCmodel.fit(X_train, y_train)

    print(f"Training Score: {GBCmodel.score(X_train, y_train)}")
    print(f"Validation Score: {GBCmodel.score(X_valid, y_valid)}")

    # Predicting and evaluating on the test set
    y_pred_gbc = GBCmodel.predict(X_valid)
    accuracy_gbc = accuracy_score(y_valid, y_pred_gbc)
    report_gbc = classification_report(y_valid, y_pred_gbc)

    print(f"Accuracy of the Gradient Boosting Classifier model: {accuracy_gbc}")
    print("Classification Report:")
    print(report_gbc)

def decision_tree_clf(X_train, X_valid, y_train, y_valid):
    DTModel = DecisionTreeClassifier(max_depth=4, min_samples_leaf=4)
    DTModel.fit(X_train, y_train)

    print(f"Training Score: {DTModel.score(X_train, y_train)}")
    print(f"Validation Score: {DTModel.score(X_valid, y_valid)}")

    # Predicting and evaluating on the test set
    y_pred_dt = DTModel.predict(X_valid)
    accuracy_dt = accuracy_score(y_valid, y_pred_dt)
    report_dt = classification_report(y_valid, y_pred_dt)

    print(f"Accuracy of the Decision Tree model: {accuracy_dt}")
    print("Classification Report:")
    print(report_dt)

def mlp_clasifier(X_train, X_valid, y_train, y_valid):
    MLPModel = MLPClassifier(hidden_layer_sizes=50, max_iter=1000)
    MLPModel.fit(X_train, y_train)

    print(f"Training Score: {MLPModel.score(X_train, y_train)}")
    print(f"Validation Score: {MLPModel.score(X_valid, y_valid)}")

    # Predicting and evaluating on the test set
    y_pred_mlp = MLPModel.predict(X_valid)
    accuracy_mlp = accuracy_score(y_valid, y_pred_mlp)
    report_mlp = classification_report(y_valid, y_pred_mlp)

    print(f"Accuracy of the MLP Classifier model: {accuracy_mlp}")
    print("Classification Report:")
    print(report_mlp)

def random_forest_clf(X_train, X_valid, y_train, y_valid):
    RFModel = RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_leaf=5)
    RFModel.fit(X_train, y_train)

    print(f"Training Score: {RFModel.score(X_train, y_train)}")
    print(f"Validation Score: {RFModel.score(X_valid, y_valid)}")

    # Predicting and evaluating on the test set
    y_pred_rf = RFModel.predict(X_valid)
    accuracy_rf = accuracy_score(y_valid, y_pred_rf)
    report_rf = classification_report(y_valid, y_pred_rf)

    print(f"Accuracy of the Random Forest Classifier model: {accuracy_rf}")
    print("Classification Report:")
    print(report_rf)

def knn_clf(X_train, X_valid, y_train, y_valid):
    KNNModel = KNeighborsClassifier(n_neighbors=10)
    KNNModel.fit(X_train, y_train)

    print(f"Training Score: {KNNModel.score(X_train, y_train)}")
    print(f"Validation Score: {KNNModel.score(X_valid, y_valid)}")

    # Predicting and evaluating on the validation set
    y_pred_knn = KNNModel.predict(X_valid)
    accuracy_knn = accuracy_score(y_valid, y_pred_knn)
    report_knn = classification_report(y_valid, y_pred_knn)

    print(f"Accuracy of the K Neighbors Classifier model: {accuracy_knn}")
    print("Classification Report:")
    print(report_knn)

def main(file_path):
    normalized_df = read_data(file_path)
    print_df(normalized_df)

    plot_histogram_normalzied(normalized_df)

    plot_correlation_heatmap(normalized_df)

    print_best_features(normalized_df)

    # Not dropping cholrestrol, age as it shows very high correaltion when it is low
    # Dropping Sex as we had dispropotionate gender counts in our data
    X = normalized_df.drop(['HeartDisease', 'Sex', 'RestingECG'], axis=1)
    y = normalized_df['HeartDisease']
    
    # Creating ML models
    # Splitting the dataset into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

    # Gradient Boosting Classifier
    gradient_boosting_clf(X_train, X_valid, y_train, y_valid)

    # Decision Tree Classifier
    decision_tree_clf(X_train, X_valid, y_train, y_valid)

    # MLP Classifier
    mlp_clasifier(X_train, X_valid, y_train, y_valid)

    # Random Forest Classifier
    random_forest_clf(X_train, X_valid, y_train, y_valid)

    # KNN Classifier
    knn_clf(X_train, X_valid, y_train, y_valid)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <normalized_heart_data.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)