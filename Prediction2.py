import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from platform import python_version

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import NuSVC
from sklearn.tree import ExtraTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


# Libraries and Python version
library = {
    "Pandas": pd,
    "Matplotlib": matplotlib,
    "Seaborn": sns,
    "NumPy": np,
    "Scikit-Learn": sklearn,
    }

# Libraries version
print("Library Version:\n")
print(f"{'':-^20} | {'':-^10}")
print(f"{'Library':^20} | {'Version':^10}")
print(f"{'':-^20} | {'':-^10}")

for nome, library in sorted(library.items()):
    print(f"{nome:<20} | {library.__version__:>10}")

# Python Version    
print()
print(f"Python Version: {python_version()}")

# Importing df 
df = pd.read_csv('C:/Users/kleon/OneDrive/Documents/Dissertation/Dissertation-Repository/merged_injury_workload.csv')

# Preview
print("\nDataset preview:")
print(df.head())

# =========================
#  Data Cleaning/ Rounding
# =========================
# Apply rounding where applicable (only if columns exist in new dataset)
for col in ["Player_Weight", "Player_Height", "Training_Intensity"]:
    if col in df.columns:
        df[col] = df[col].round(2)



# View df_vg | Visualizar df_vg
df.head(5)

# Creating DataFrame with Dtype, Unique, and Null information
df_info = pd.DataFrame(df.dtypes, columns=['Dtype'])
df_info['Unique'] = df.nunique().values
df_info['Null'] = df.isnull().sum().values
df_info

# Df Describe 
with pd.option_context(
    "display.float_format",
    "{:.2f}".format,
    "display.max_columns",
    None,
):
    print(df.describe())

# Calculate the Body Mass Index (BMI)
df['BMI'] = df['Player_Weight'] / (df['Player_Height'] / 100) ** 2

# Defining gaps for BMI classification
gaps = [-float('inf'), 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
categories = ['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III']

# Create "BMI_Classification" column
df['BMI_Classification'] = pd.cut(df['BMI'], bins=gaps, labels=categories, right=False)

df.head(1)

# Finding the youngest and oldest age among athletes
print('Player Age Min: {}'.format(df.Player_Age.min()))
print('Player Age Max: {}'.format(df.Player_Age.max()))

# Creating columns with grouping
df["Age_Group"] = pd.cut(
    df["Player_Age"],
    bins=[18, 22, 26, 30, 34, df["Player_Age"].max()],
    labels=["18-22", "23-26", "27-30", "31-34", "35+"],
    include_lowest=True,
)

df.head(5)

def plot_histogram_kde_and_boxplot(dataframe, column, color_column):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Remove grid and spines
    for ax in axs:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Plot histogram (subplot 1)
    sns.histplot(data=dataframe, x=column, bins=20, color='skyblue', edgecolor='black', kde=True, ax=axs[0])

    # Add labels
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].set_title(f'{column} Histogram', weight='bold', size=13)

    # Plot KDE (subplot 2)
    sns.kdeplot(data=dataframe, x=column, color='skyblue', fill=True, hue=color_column, palette={0: 'green', 1: 'red'}, ax=axs[1])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].set_title(f'{column} Density', weight='bold', size=13)  

    # Plot boxplot (subplot 3)
    sns.boxplot(data=dataframe[column], orient='h', ax=axs[2])

    # Add labels
    axs[2].set_xlabel('')
    axs[2].set_ylabel('')
    axs[2].set_title(f'{column} Boxplot', weight='bold', size=13)
   
    # Adjust layout
    plt.tight_layout()

    # Display figure
    plt.show()

    
def plot_dual_chart(dataframe, column1, column2, cat_order=None, y_limit1=None, y_limit2=None):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Remove grid and spines
    for ax in axs:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Plot histogram
    sns.histplot(data=dataframe, x=column1, bins=20, color='skyblue', edgecolor='black', kde=True, ax=axs[0])
    axs[0].set_title(f'{column1} Histogram', weight='bold', size=13)
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')

    # Define y limit
    if y_limit1 is None:
        y_limit1 = dataframe[column1].max() * 1.1  
    axs[0].set_ylim(top=y_limit1)

    # Plot two sets of bars
    ax = sns.countplot(data=dataframe, x=column2, hue='Likelihood_of_Injury', palette={0: 'green', 1: 'red'}, ax=axs[1], linewidth=2, order=cat_order)

    # Add labels
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].set_title(f'{column2} x Likelihood_of_Injury', weight='bold', size=13)

    # Rotate x-axis labels
    axs[1].tick_params(axis='x', rotation=0)

    # Remove background grid
    axs[1].grid(False)

    # Add legend
    axs[1].legend()

    # Define upper limit
    if y_limit2 is None:
        y_limit2 = dataframe[column2].value_counts().max() * 1.1  # Max value multiplied by 1.1 to ensure a margin
    axs[1].set_ylim(top=y_limit2)

    # Add values on top of each bar
    for p in axs[1].patches:
        height = p.get_height()
        if not np.isnan(height):
            axs[1].annotate(str(int(height)), (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points', color='black', weight='bold', size=13)
        else:
            axs[1].annotate("0", (p.get_x() + p.get_width() / 2., 0),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points', color='black', weight='bold', size=13)

    # Adjust layout
    plt.tight_layout()

    # Display figure
    plt.show()

plot_dual_chart(df, 'Player_Age', 'Age_Group', cat_order=["18-22", "23-26", "27-30", "31-34", "35+"], y_limit1=130, y_limit2=160)

plot_dual_chart(df, 'BMI', 'BMI_Classification', y_limit1=180, y_limit2=350)

plot_histogram_kde_and_boxplot(df, 'Player_Weight', 'Likelihood_of_Injury')

plot_histogram_kde_and_boxplot(df, 'Player_Height', 'Likelihood_of_Injury')

plot_histogram_kde_and_boxplot(df, 'Training_Intensity', 'Likelihood_of_Injury')

plot_histogram_kde_and_boxplot(df, 'Recovery_Time', 'Likelihood_of_Injury')

plot_histogram_kde_and_boxplot(df, 'Previous_Injuries', 'Likelihood_of_Injury')

# Count 'Likelihood_of_Injury' | Contagem 'Likelihood_of_Injury'
li_count = df['Likelihood_of_Injury'].value_counts()

# Plot pie chart | Plotar o gráfico de pizza
plt.figure(figsize=(5, 5))
plt.pie(li_count, labels=li_count.index, autopct='%1.1f%%', startangle=140, colors=['#8CFCA4', '#FF6961'])
plt.title('Distribution of Likelihood_of_Injury', weight='bold', size=13)
plt.axis('equal') 
plt.show()

# Categorical columns
one_hot_cols = [
    "BMI_Classification",
    "Age_Group",
]

# Selecting only categorical columns from the DataFrame 
df_categorical = df[one_hot_cols]

# Applying OneHotEncoder
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df_categorical)

# Obtaining names of the features generated by OneHotEncoder
one_hot_feature_names = encoder.get_feature_names_out(one_hot_cols)

# Creating a DataFrame with transformed features
df_encoded = pd.DataFrame(encoded_data.toarray(), columns=one_hot_feature_names)

# Joining DataFrames
df_final = pd.concat([df, df_encoded], axis=1)

# Dropping categorical columns
df_final.drop(columns=['BMI_Classification', 'Age_Group'], inplace=True)

# Visualizing the first few rows of the final DataFrame
df_final.head()

# Calculating correlation matrix
correlation_matrix = df_final.corr()

# Plotting heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlation Matrix', weight='bold', size=13)
plt.show()

# Calculating correlation matrix
correlation_matrix = df_final.corr()

# Selecting only 'Likelihood_of_Injury' column from the correlation matrix 
correlation_with_likelihood = correlation_matrix['Likelihood_of_Injury']

# Removing the correlation with the 'Likelihood_of_Injury' column
correlation_with_likelihood = correlation_with_likelihood.drop('Likelihood_of_Injury')

# Sorting correlations in descending order
correlation_with_likelihood = correlation_with_likelihood.sort_values(ascending=False)

# Plotting correlation bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_likelihood.index, y=correlation_with_likelihood.values, palette='coolwarm')
plt.xticks(rotation=90, ha='center')  
plt.xlabel('')
plt.ylabel('')
plt.box(False) 
plt.title('Correlation of Columns with Likelihood_of_Injury', weight='bold', size=13)
plt.tight_layout()
plt.show()

# Drop columns starting with "Age_Group"
df_final = df_final.loc[:, ~df_final.columns.str.startswith('Age_Group')]

# Drop BMI column
df_final = df_final.drop(columns=['BMI'])

df_final.head(1)

# Features
X = df_final.drop('Likelihood_of_Injury', axis=1)

# Target variable
y = df_final['Likelihood_of_Injury']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

# Dictionary models
models = {
    "LGBMClassifier": LGBMClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "ExtraTreesClassifier": ExtraTreesClassifier(),
    "NuSVC": NuSVC(probability=True),
    "ExtraTreeClassifier": ExtraTreeClassifier(),
    }

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    
    print(f"Model: {model_name}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print("-" * 50)

# Create figure and axes
fig, axes = plt.subplots(1, len(models), figsize=(15, 3.5))

# Plot confusion matrix for each model
for ax, (model_name, model) in zip(axes, models.items()):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{model_name}", weight='bold', size=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

# Adjust layout and show figure
plt.tight_layout()
plt.show()

# Plot Inverted ROC Curve
plt.figure(figsize=(8, 6))
for model_name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 0]  # Probabilities of belonging to the negative class
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    tnr = 1 - fpr  # True Negative Rate (TNR)
    fnr = 1 - tpr  # False Negative Rate (FNR)
    
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fnr, tnr, label=f'{model_name} (AUC = {auc:.2f})')

# Plot the diagonal line
plt.plot([0, 1], [0, 1], linestyle='--', color='black')

plt.xlabel('False Negative Rate (FNR)')
plt.ylabel('True Negative Rate (TNR)')
plt.title('Inverted ROC Curve', weight='bold', size=13)
plt.legend()

plt.box(False)  # Remove plot borders
plt.show()