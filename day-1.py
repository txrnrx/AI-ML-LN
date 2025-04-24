# 1. Import the dataset and explore basic info (nulls, data types)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1.1 Load the dataset
df = pd.read_csv('/Titanic-Dataset.csv')  # Adjust path if needed

# 1.2 Display dataset shape
print("1.1 Shape of the dataset:", df.shape)

# 1.3 Show data types and non-null counts
print("\n1.2 Data types and non-null counts:")
print(df.info())

# 1.4 Check missing values
print("\n1.3 Missing values before handling:")
print(df.isnull().sum())


# 2. Handle missing values using mean/median/imputation

# 2.1 Fill missing 'Age' values with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# 2.2 Fill missing 'Embarked' values with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 2.3 Drop 'Cabin' column due to excessive missing values
df.drop(columns='Cabin', inplace=True)

# 2.4 Drop any remaining rows with nulls
df.dropna(inplace=True)

# 2.5 Confirm all missing values handled
print("\n2.1 Missing values after handling:")
print(df.isnull().sum())


# 3. Convert categorical features into numerical using encoding

# 3.1 Apply one-hot encoding to 'Sex' and 'Embarked'
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 3.2 View resulting columns
print("\n3.1 Columns after encoding:")
print(df.columns)


# 4. Normalize/standardize the numerical features

# 4.1 Identify numerical columns to scale
num_cols = ['Age', 'Fare']
scaler = StandardScaler()

# 4.2 Apply standardization
df[num_cols] = scaler.fit_transform(df[num_cols])

# 4.3 Show standardized column stats
print("\n4.1 Standardized statistics:")
print(df[num_cols].describe())


# 5. Visualize outliers using boxplots and remove them

# 5.1 Display boxplots to visualize outliers
plt.figure(figsize=(12, 5))
for i, col in enumerate(num_cols):
    plt.subplot(1, 2, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# 5.2 Remove outliers using IQR method
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# 5.3 Show final dataset shape
print("\n5.1 Final dataset shape after outlier removal:", df.shape)
