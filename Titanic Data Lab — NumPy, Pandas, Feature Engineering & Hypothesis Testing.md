# Titanic Data Lab
## NumPy + Pandas from Zero to Feature Engineering and Hypothesis Testing

> **Dataset:** Titanic passenger dataset from the Seaborn repository  
> **Main Goal:** Learn how raw data becomes useful information and eventually useful features for machine learning.

---

# 1. What Are We Going to Learn?

In this lesson we will use one real dataset to understand the complete beginner data-analysis workflow:

```text
Raw Data
   ↓
Load Data
   ↓
Understand Data
   ↓
NumPy Operations
   ↓
Pandas Operations
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Form Hypotheses
   ↓
Test Hypotheses
   ↓
Prepare Data for Machine Learning
```

By the end, students should understand:

- What NumPy is
- What a NumPy array is
- Vectorized operations
- Filtering arrays
- Aggregations
- Missing values
- What pandas is
- Series and DataFrames
- Loading CSV files
- Inspecting datasets
- Selecting rows and columns
- Filtering data
- Sorting data
- Grouping and aggregation
- Missing-value handling
- Exploratory Data Analysis
- Feature engineering
- Categorical encoding
- Data leakage
- Statistical hypotheses
- Null and alternative hypotheses
- Chi-square tests
- t-tests
- Mann–Whitney U tests
- How statistics helps us investigate assumptions before machine learning

---

# 2. Download the Dataset

If you are using Google Colab:

```python
!wget https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
```

Check whether the file exists:

```python
!ls
```

You should see:

```text
titanic.csv
```

---

# 3. Install / Import the Libraries

We mainly need:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

For statistical hypothesis testing:

```python
from scipy import stats
```

---

# 4. Before pandas — Understand NumPy

## 4.1 What is NumPy?

**NumPy** stands for:

> Numerical Python

NumPy helps Python work efficiently with numerical data.

Instead of processing values one by one, NumPy allows us to operate on many values together.

---

# 5. Python List vs NumPy Array

Start with a normal Python list.

```python
ages = [22, 38, 26, 35, 35]

print(ages)
print(type(ages))
```

Create a NumPy array:

```python
ages_array = np.array(ages)

print(ages_array)
print(type(ages_array))
```

Output looks similar to:

```text
[22 38 26 35 35]
```

But this is no longer a Python `list`.

It is a:

```text
numpy.ndarray
```

---

# 6. Why NumPy?

Imagine increasing every age by 1.

With a normal Python list:

```python
ages = [22, 38, 26, 35, 35]

new_ages = []

for age in ages:
    new_ages.append(age + 1)

print(new_ages)
```

With NumPy:

```python
ages = np.array([22, 38, 26, 35, 35])

new_ages = ages + 1

print(new_ages)
```

Output:

```text
[23 39 27 36 36]
```

This concept is called **vectorization**.

Instead of manually looping through every value, we perform an operation on the entire array.

---

# 7. NumPy Array Properties

Create an array:

```python
numbers = np.array([10, 20, 30, 40, 50])
```

Check its shape:

```python
numbers.shape
```

Check number of dimensions:

```python
numbers.ndim
```

Check data type:

```python
numbers.dtype
```

Check number of elements:

```python
numbers.size
```

---

# 8. NumPy Indexing

```python
numbers = np.array([10, 20, 30, 40, 50])
```

First element:

```python
numbers[0]
```

Second element:

```python
numbers[1]
```

Last element:

```python
numbers[-1]
```

---

# 9. NumPy Slicing

```python
numbers = np.array([10, 20, 30, 40, 50])
```

First three:

```python
numbers[0:3]
```

Output:

```text
[10 20 30]
```

Another example:

```python
numbers[2:]
```

Output:

```text
[30 40 50]
```

---

# 10. Mathematical Operations

```python
numbers = np.array([10, 20, 30, 40, 50])
```

Addition:

```python
numbers + 10
```

Multiplication:

```python
numbers * 2
```

Division:

```python
numbers / 10
```

Square:

```python
numbers ** 2
```

---

# 11. NumPy Aggregation

Aggregation means reducing many numbers into a summary value.

```python
numbers = np.array([10, 20, 30, 40, 50])
```

Mean:

```python
np.mean(numbers)
```

Median:

```python
np.median(numbers)
```

Minimum:

```python
np.min(numbers)
```

Maximum:

```python
np.max(numbers)
```

Sum:

```python
np.sum(numbers)
```

Standard deviation:

```python
np.std(numbers)
```

---

# 12. Boolean Operations

This is one of the most important concepts in data analysis.

```python
ages = np.array([22, 38, 16, 35, 12, 40])
```

Ask:

> Which passengers are adults?

```python
ages >= 18
```

Output:

```text
[ True  True False  True False  True]
```

NumPy created a **boolean mask**.

---

# 13. Boolean Filtering

Use that mask to get the actual values.

```python
ages[ages >= 18]
```

Result:

```text
[22 38 35 40]
```

Now:

```python
ages[ages < 18]
```

Result:

```text
[16 12]
```

This concept will become extremely important in pandas.

---

# 14. Multiple Conditions

```python
ages = np.array([5, 15, 22, 38, 65, 80])
```

Find people between 18 and 60:

```python
ages[(ages >= 18) & (ages <= 60)]
```

Important:

Use:

```python
&
```

instead of Python's:

```python
and
```

for NumPy array conditions.

---

# 15. NumPy `where`

Sometimes we want to create categories.

```python
ages = np.array([5, 15, 22, 38, 65, 80])

category = np.where(ages >= 18, "Adult", "Child")

print(category)
```

Possible result:

```text
['Child' 'Child' 'Adult' 'Adult' 'Adult' 'Adult']
```

This is our first simple example of **feature engineering**.

We converted:

```text
age
```

into:

```text
age category
```

---

# 16. Missing Data in NumPy

NumPy represents many numerical missing values using:

```python
np.nan
```

Example:

```python
ages = np.array([22, 38, np.nan, 35, 28])
```

Try:

```python
np.mean(ages)
```

You may receive:

```text
nan
```

Because a missing value exists.

Instead use:

```python
np.nanmean(ages)
```

Also:

```python
np.nanmedian(ages)
```

```python
np.nanmin(ages)
```

```python
np.nanmax(ages)
```

---

# 17. Move to the Real Titanic Dataset

Now we move from small examples to real data.

Import pandas:

```python
import pandas as pd
```

Load the dataset:

```python
df = pd.read_csv("titanic.csv")
```

---

# 18. What is Pandas?

Pandas is a Python library designed for working with structured data.

The two most important pandas structures are:

```text
Series
DataFrame
```

---

# 19. What is a Series?

A Series is basically one column of data.

Example:

```python
df["age"]
```

Check its type:

```python
type(df["age"])
```

Result:

```text
pandas.core.series.Series
```

---

# 20. What is a DataFrame?

A DataFrame is a table containing rows and columns.

Our `df` variable contains the Titanic DataFrame.

```python
type(df)
```

---

# 21. First Look at the Dataset

Never start analysing data before looking at it.

```python
df.head()
```

Shows the first five rows.

Try:

```python
df.head(10)
```

---

# 22. Look at the Last Rows

```python
df.tail()
```

---

# 23. Random Sample

Sometimes `head()` gives a misleading first impression.

Use:

```python
df.sample(5)
```

This shows random passengers.

---

# 24. Dataset Shape

```python
df.shape
```

The result gives:

```text
(rows, columns)
```

For example:

```text
(891, 15)
```

Meaning approximately:

```text
891 passengers
15 variables
```

---

# 25. Column Names

```python
df.columns
```

You should see columns similar to:

```text
survived
pclass
sex
age
sibsp
parch
fare
embarked
class
who
adult_male
deck
embark_town
alive
alone
```

---

# 26. What Do These Columns Mean?

| Column | Meaning |
|---|---|
| `survived` | Whether passenger survived |
| `pclass` | Passenger class |
| `sex` | Sex |
| `age` | Age |
| `sibsp` | Number of siblings/spouses aboard |
| `parch` | Number of parents/children aboard |
| `fare` | Ticket fare |
| `embarked` | Port of embarkation |
| `class` | Passenger class as text |
| `who` | Man / woman / child |
| `adult_male` | Whether adult male |
| `deck` | Cabin deck |
| `embark_town` | Full embarkation town |
| `alive` | Yes/No representation of survival |
| `alone` | Whether passenger travelled alone |

---

# 27. `info()` — One of the Most Important Commands

```python
df.info()
```

This tells us:

- column names
- number of rows
- number of non-null values
- missing values
- data types
- approximate memory usage

When receiving an unfamiliar dataset, one of the first commands should be:

```python
df.info()
```

---

# 28. Data Types

```python
df.dtypes
```

Common pandas types:

```text
int64
float64
object
bool
category
```

Examples:

```text
age        → float
fare       → float
survived   → integer
sex        → object/string
```

---

# 29. Numerical Summary

```python
df.describe()
```

This provides values including:

```text
count
mean
std
min
25%
50%
75%
max
```

---

# 30. Include Categorical Variables

```python
df.describe(include="all")
```

Now pandas also provides information about categorical columns.

---

# 31. Selecting One Column

```python
df["age"]
```

---

# 32. Selecting Multiple Columns

```python
df[["age", "sex", "fare"]]
```

Important difference:

```python
df["age"]
```

returns a Series.

While:

```python
df[["age"]]
```

returns a DataFrame.

---

# 33. Row Selection with `iloc`

`iloc` works using numerical positions.

First row:

```python
df.iloc[0]
```

First five rows:

```python
df.iloc[0:5]
```

Specific rows and columns:

```python
df.iloc[0:5, 0:4]
```

---

# 34. Row Selection with `loc`

`loc` works using labels and conditions.

Example:

```python
df.loc[0, "age"]
```

Multiple columns:

```python
df.loc[0:5, ["age", "sex", "survived"]]
```

---

# 35. Filtering Data

Find passengers who survived:

```python
df[df["survived"] == 1]
```

Passengers who did not survive:

```python
df[df["survived"] == 0]
```

---

# 36. Filter by Sex

```python
df[df["sex"] == "female"]
```

---

# 37. Multiple Conditions

Female passengers in first class:

```python
df[
    (df["sex"] == "female") &
    (df["pclass"] == 1)
]
```

Passengers older than 50 who survived:

```python
df[
    (df["age"] > 50) &
    (df["survived"] == 1)
]
```

---

# 38. OR Conditions

Passengers who were first class **or** second class:

```python
df[
    (df["pclass"] == 1) |
    (df["pclass"] == 2)
]
```

---

# 39. `isin()`

A cleaner way:

```python
df[df["pclass"].isin([1, 2])]
```

---

# 40. Sorting Data

Oldest passengers:

```python
df.sort_values("age", ascending=False).head(10)
```

Cheapest fares:

```python
df.sort_values("fare").head(10)
```

---

# 41. Count Unique Values

```python
df["sex"].value_counts()
```

Passenger classes:

```python
df["pclass"].value_counts()
```

Survival:

```python
df["survived"].value_counts()
```

---

# 42. Convert Counts to Percentages

```python
df["survived"].value_counts(normalize=True)
```

Multiply by 100:

```python
df["survived"].value_counts(normalize=True) * 100
```

Now we have percentages.

---

# 43. Survival Rate

Because:

```text
0 = did not survive
1 = survived
```

we can use:

```python
df["survived"].mean()
```

Why?

For:

```text
0, 1, 1, 0, 1
```

the mean is:

```text
3 / 5 = 0.60
```

That is the proportion of `1`s.

Therefore:

```python
df["survived"].mean() * 100
```

gives the survival percentage.

---

# 44. GroupBy — Extremely Important

Suppose our question is:

> Did survival differ between males and females?

Use:

```python
df.groupby("sex")["survived"].mean()
```

Now by passenger class:

```python
df.groupby("pclass")["survived"].mean()
```

---

# 45. Multiple GroupBy Variables

```python
df.groupby(["sex", "pclass"])["survived"].mean()
```

This answers a much more interesting question:

> What was survival like for each combination of sex and passenger class?

---

# 46. Multiple Aggregations

```python
df.groupby("pclass")["age"].agg([
    "count",
    "mean",
    "median",
    "min",
    "max"
])
```

---

# 47. Group Fare Statistics

```python
df.groupby("pclass")["fare"].agg([
    "count",
    "mean",
    "median",
    "min",
    "max"
])
```

Ask students:

> Why is the mean fare different from the median fare?

This introduces the concept of **skewed distributions** and **outliers**.

---

# 48. Cross Tabulation

A cross-tabulation compares categorical variables.

```python
pd.crosstab(df["sex"], df["survived"])
```

Convert counts into percentages:

```python
pd.crosstab(
    df["sex"],
    df["survived"],
    normalize="index"
) * 100
```

---

# 49. Survival by Passenger Class

```python
pd.crosstab(
    df["pclass"],
    df["survived"],
    normalize="index"
) * 100
```

---

# 50. Missing Values

Real-world data is rarely clean.

Check missing values:

```python
df.isnull().sum()
```

Alternative:

```python
df.isna().sum()
```

---

# 51. Missing-Value Percentage

```python
missing_percentage = (
    df.isna().sum() / len(df) * 100
)

missing_percentage.sort_values(ascending=False)
```

This is much more useful than simply counting missing values.

---

# 52. Visualize Missingness

```python
missing_percentage.sort_values().plot(
    kind="barh",
    figsize=(8, 6)
)

plt.xlabel("Missing Values (%)")
plt.title("Missing Data in Titanic Dataset")
plt.show()
```

---

# 53. Why is Missing Data Important?

Imagine:

```text
Age missing
Cabin/deck missing
Embarkation missing
```

We cannot simply assume:

```text
missing = useless
```

We need to ask:

> Why might this value be missing?

Missingness itself can sometimes contain information.

---

# 54. Strategy 1 — Drop Missing Values

Example:

```python
df.dropna()
```

But be careful.

This may remove many passengers.

Check:

```python
len(df)
```

and:

```python
len(df.dropna())
```

Ask:

> How much data did we lose?

---

# 55. Drop Rows Missing Age Only

```python
df.dropna(subset=["age"])
```

---

# 56. Strategy 2 — Fill Missing Values

Median age:

```python
df["age"].median()
```

Create a working copy:

```python
data = df.copy()
```

Fill missing age:

```python
data["age"] = data["age"].fillna(
    data["age"].median()
)
```

Check:

```python
data["age"].isna().sum()
```

---

# 57. Why Median Instead of Mean?

Compare:

```python
df["age"].mean()
```

and:

```python
df["age"].median()
```

The median is often more resistant to extreme values.

This does **not** mean median is always correct.

Imputation should depend on:

- the variable
- distribution
- missingness mechanism
- analysis goal

---

# 58. Better Age Imputation

Instead of giving every passenger the same median age, calculate group-level median age.

For example:

```python
data = df.copy()

group_age_median = data.groupby(
    ["pclass", "sex"]
)["age"].transform("median")

data["age"] = data["age"].fillna(
    group_age_median
)
```

Now age is estimated using passenger class and sex.

---

# 59. Missingness as a Feature

Before filling age:

```python
data = df.copy()

data["age_missing"] = data["age"].isna().astype(int)
```

Then:

```python
data[["age", "age_missing"]].head(10)
```

Now the model can potentially learn whether **age being missing** contains useful information.

---

# 60. Fill Embarked

Check:

```python
data["embarked"].value_counts()
```

Get the mode:

```python
data["embarked"].mode()
```

Fill:

```python
data["embarked"] = data["embarked"].fillna(
    data["embarked"].mode()[0]
)
```

---

# 61. What Should We Do with `deck`?

Check missing percentage:

```python
data["deck"].isna().mean() * 100
```

Possible options:

```text
1. Drop the variable
2. Create an "Unknown" category
3. Investigate whether missingness itself is meaningful
```

Example:

```python
data["deck_known"] = data["deck"].notna().astype(int)
```

This converts:

```text
deck known?
```

into:

```text
0 / 1
```

---

# 62. Duplicates

Check:

```python
data.duplicated().sum()
```

But be careful.

Two Titanic passengers can theoretically have identical values across the columns we happen to have.

Therefore:

```python
duplicated()
```

does not automatically mean the record is incorrect.

---

# 63. Exploratory Data Analysis

Now we start asking questions.

This is called:

> **EDA — Exploratory Data Analysis**

EDA means exploring a dataset before building models.

---

# 64. Question 1 — How Many Survived?

```python
data["survived"].value_counts()
```

Percent:

```python
data["survived"].value_counts(normalize=True) * 100
```

---

# 65. Plot Survival

```python
data["survived"].value_counts().plot(
    kind="bar"
)

plt.title("Titanic Survival")
plt.xlabel("Survived")
plt.ylabel("Passengers")
plt.show()
```

---

# 66. Question 2 — Survival and Sex

```python
data.groupby("sex")["survived"].mean()
```

Visualize:

```python
data.groupby("sex")["survived"].mean().plot(
    kind="bar"
)

plt.ylabel("Survival Rate")
plt.title("Survival Rate by Sex")
plt.show()
```

---

# 67. Question 3 — Passenger Class and Survival

```python
data.groupby("pclass")["survived"].mean()
```

Plot:

```python
data.groupby("pclass")["survived"].mean().plot(
    kind="bar"
)

plt.ylabel("Survival Rate")
plt.title("Survival Rate by Passenger Class")
plt.show()
```

---

# 68. Question 4 — Age Distribution

```python
data["age"].hist(
    bins=30,
    figsize=(8, 5)
)

plt.xlabel("Age")
plt.ylabel("Passengers")
plt.title("Passenger Age Distribution")
plt.show()
```

---

# 69. Survivors vs Non-Survivors Age

```python
data[data["survived"] == 1]["age"].describe()
```

Compare:

```python
data[data["survived"] == 0]["age"].describe()
```

Or:

```python
data.groupby("survived")["age"].agg([
    "mean",
    "median",
    "std",
    "min",
    "max"
])
```

---

# 70. Question 5 — Fare

```python
data["fare"].describe()
```

Visualize:

```python
data["fare"].hist(
    bins=40,
    figsize=(8, 5)
)

plt.xlabel("Fare")
plt.title("Fare Distribution")
plt.show()
```

Notice whether the distribution is symmetrical or skewed.

---

# 71. What is Feature Engineering?

Feature engineering means creating useful variables from existing information.

Raw data:

```text
sibsp = 1
parch = 2
```

New information:

```text
family_size = 4
```

because:

```text
passenger + spouse/siblings + parents/children
```

Feature engineering helps transform raw information into representations that may be more useful for analysis or machine learning.

---

# 72. Feature Engineering 1 — Family Size

```python
data["family_size"] = (
    data["sibsp"] +
    data["parch"] +
    1
)
```

Check:

```python
data[
    ["sibsp", "parch", "family_size"]
].head()
```

---

# 73. Family Size and Survival

```python
data.groupby("family_size")["survived"].agg([
    "count",
    "mean"
])
```

This can help us investigate whether travelling in different-sized family groups is associated with survival.

---

# 74. Feature Engineering 2 — Travelling Alone

```python
data["is_alone_engineered"] = (
    data["family_size"] == 1
).astype(int)
```

Check:

```python
data[
    ["family_size", "is_alone_engineered"]
].head()
```

---

# 75. Compare with Existing `alone`

The dataset already contains:

```python
data["alone"]
```

Compare them:

```python
pd.crosstab(
    data["alone"],
    data["is_alone_engineered"]
)
```

This demonstrates an important lesson:

> Sometimes a dataset already contains a feature that can be reconstructed from other features.

---

# 76. Feature Engineering 3 — Age Groups

A raw age like:

```text
27
```

may be useful.

But a category may also be useful:

```text
Young Adult
```

Create age groups:

```python
bins = [0, 12, 18, 35, 50, 65, np.inf]

labels = [
    "Child",
    "Teen",
    "Young Adult",
    "Adult",
    "Older Adult",
    "Senior"
]

data["age_group"] = pd.cut(
    data["age"],
    bins=bins,
    labels=labels
)
```

---

# 77. Age Group Distribution

```python
data["age_group"].value_counts()
```

Survival rate:

```python
data.groupby(
    "age_group",
    observed=False
)["survived"].mean()
```

---

# 78. Feature Engineering 4 — Child

Create a very simple binary feature:

```python
data["is_child"] = (
    data["age"] < 16
).astype(int)
```

Then investigate:

```python
data.groupby("is_child")["survived"].mean()
```

---

# 79. Feature Engineering 5 — Family Type

Instead of exact family size:

```python
def family_type(size):
    if size == 1:
        return "Alone"
    elif size <= 4:
        return "Small Family"
    else:
        return "Large Family"
```

Apply:

```python
data["family_type"] = (
    data["family_size"]
    .apply(family_type)
)
```

Inspect:

```python
data[
    ["family_size", "family_type"]
].head(10)
```

---

# 80. Family Type and Survival

```python
data.groupby(
    "family_type"
)["survived"].agg([
    "count",
    "mean"
])
```

---

# 81. Feature Engineering 6 — Fare Per Person

Imagine one fare covers people travelling together.

A simple derived feature is:

```python
data["fare_per_person"] = (
    data["fare"] /
    data["family_size"]
)
```

Inspect:

```python
data[
    ["fare", "family_size", "fare_per_person"]
].head()
```

This is only a proxy; the dataset does not directly tell us exactly how each ticket fare was shared.

---

# 82. Feature Engineering 7 — Fare Bands

Instead of exact fare:

```python
data["fare_band"] = pd.qcut(
    data["fare"],
    q=4,
    duplicates="drop"
)
```

Check:

```python
data["fare_band"].value_counts()
```

Survival:

```python
data.groupby(
    "fare_band",
    observed=False
)["survived"].mean()
```

---

# 83. `cut()` vs `qcut()`

### `cut()`

Creates ranges based on values.

Example:

```text
0–10
10–20
20–30
```

### `qcut()`

Tries to create groups with approximately equal numbers of observations.

Example:

```text
bottom 25%
25–50%
50–75%
top 25%
```

---

# 84. Feature Engineering 8 — Fare Log Transformation

Fare is highly skewed.

A common transformation is:

```python
data["log_fare"] = np.log1p(
    data["fare"]
)
```

Why `log1p()`?

Because:

```python
np.log1p(x)
```

calculates approximately:

```text
log(1 + x)
```

and works safely when:

```text
x = 0
```

---

# 85. Compare Fare Distribution

Original:

```python
data["fare"].hist(bins=40)

plt.title("Original Fare")
plt.show()
```

Transformed:

```python
data["log_fare"].hist(bins=40)

plt.title("Log Transformed Fare")
plt.show()
```

Students can visually compare skewness.

---

# 86. Categorical Data

Machine-learning algorithms usually work with numbers.

But we have:

```text
male
female
```

and:

```text
S
C
Q
```

We need a numerical representation.

---

# 87. Binary Encoding

For sex:

```python
data["sex_encoded"] = (
    data["sex"]
    .map({
        "male": 0,
        "female": 1
    })
)
```

Check:

```python
data[
    ["sex", "sex_encoded"]
].head()
```

---

# 88. One-Hot Encoding

For embarkation:

```python
pd.get_dummies(
    data["embarked"],
    dtype=int
)
```

Or directly modify a model dataset:

```python
model_data = pd.get_dummies(
    data,
    columns=["embarked"],
    dtype=int
)
```

---

# 89. Understanding Redundant Features

The Titanic dataset is excellent for teaching redundancy.

Consider:

```text
pclass
class
```

Both represent passenger class.

Similarly:

```text
embarked
embark_town
```

represent similar information.

And:

```text
alone
```

can be derived from:

```text
sibsp + parch
```

Redundant features are not necessarily incorrect.

But we should understand where information comes from.

---

# 90. The Most Important Example — Data Leakage

Look at:

```python
data[
    ["survived", "alive"]
].head(20)
```

`survived`:

```text
0 / 1
```

`alive`:

```text
no / yes
```

They represent the same outcome.

If our machine-learning target is:

```python
survived
```

and we give the model:

```python
alive
```

as an input feature, the model already knows the answer.

That is **target leakage**.

---

# 91. Leakage vs Redundancy

These concepts are different.

### Target Leakage

```text
alive → essentially gives away survived
```

This is a serious problem.

### Redundancy

```text
class ↔ pclass
```

Both describe passenger class.

### Derived Information

```text
alone ← sibsp + parch
```

or:

```text
adult_male ← age + sex logic
```

These may contain information already available elsewhere.

---

# 92. Build a Clean Working Dataset

For teaching/model preparation, start from relatively fundamental fields:

```python
clean = df[
    [
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]
].copy()
```

---

# 93. Add Missing-Value Indicator

```python
clean["age_missing"] = (
    clean["age"]
    .isna()
    .astype(int)
)
```

---

# 94. Impute Age

```python
age_medians = clean.groupby(
    ["pclass", "sex"]
)["age"].transform("median")

clean["age"] = clean["age"].fillna(
    age_medians
)
```

---

# 95. Fill Embarkation

```python
clean["embarked"] = clean["embarked"].fillna(
    clean["embarked"].mode()[0]
)
```

---

# 96. Add Family Size

```python
clean["family_size"] = (
    clean["sibsp"] +
    clean["parch"] +
    1
)
```

---

# 97. Add Alone Feature

```python
clean["is_alone"] = (
    clean["family_size"] == 1
).astype(int)
```

---

# 98. Add Child Feature

```python
clean["is_child"] = (
    clean["age"] < 16
).astype(int)
```

---

# 99. Add Fare Per Person

```python
clean["fare_per_person"] = (
    clean["fare"] /
    clean["family_size"]
)
```

---

# 100. Add Age Groups

```python
clean["age_group"] = pd.cut(
    clean["age"],
    bins=[0, 12, 18, 35, 50, 65, np.inf],
    labels=[
        "Child",
        "Teen",
        "Young Adult",
        "Adult",
        "Older Adult",
        "Senior"
    ]
)
```

---

# 101. Inspect Our Engineered Dataset

```python
clean.head()
```

And:

```python
clean.info()
```

Our dataset is now much richer than the original subset.

This demonstrates:

> **Feature engineering = transforming existing information into potentially more useful representations.**

---

# Part 2 — Hypothesis-Driven Data Analysis

---

# 102. What is a Hypothesis?

Suppose we notice:

```python
clean.groupby("sex")["survived"].mean()
```

The numbers may look different.

But we need to ask:

> Is this difference potentially just random variation?

This introduces statistical hypothesis testing.

---

# 103. Scientific Thinking

Instead of saying:

> "Females survived more."

we ask:

> "Is survival associated with sex in this dataset?"

This changes our thinking from:

```text
Observation
```

to:

```text
Question
→ Hypothesis
→ Statistical Test
→ Evidence
→ Interpretation
```

---

# 104. Null Hypothesis

Usually:

```text
H₀
```

means:

> No meaningful statistical relationship or difference exists.

Example:

```text
H₀:
Passenger sex and survival are independent.
```

---

# 105. Alternative Hypothesis

```text
H₁
```

means:

> A relationship or difference exists.

Example:

```text
H₁:
Passenger sex and survival are associated.
```

---

# 106. What is a p-value?

Very roughly, the p-value asks:

> If the null hypothesis were true, how compatible would our observed result be with random sampling variation under the test model?

A common teaching threshold is:

```text
α = 0.05
```

If:

```text
p < 0.05
```

we reject the null hypothesis at that threshold.

If:

```text
p >= 0.05
```

we do **not** reject the null hypothesis.

Important:

```text
p > 0.05
```

does **not** prove the null hypothesis is true.

---

# 107. Statistical Significance ≠ Practical Importance

A very small p-value does not automatically mean:

```text
"very important"
```

Large datasets can make small differences statistically detectable.

We should also consider:

- effect size
- actual percentages
- domain context
- assumptions
- possible confounding variables

---

# 108. Hypothesis 1 — Sex and Survival

Question:

> Is passenger sex associated with survival?

Both variables are categorical:

```text
sex
survived
```

A common test is:

> **Chi-Square Test of Independence**

---

# 109. Create the Contingency Table

```python
sex_survival_table = pd.crosstab(
    clean["sex"],
    clean["survived"]
)

sex_survival_table
```

---

# 110. Run Chi-Square Test

```python
from scipy.stats import chi2_contingency

chi2, p_value, dof, expected = chi2_contingency(
    sex_survival_table
)

print("Chi-square:", chi2)
print("p-value:", p_value)
print("Degrees of freedom:", dof)
```

---

# 111. Interpret the Test

```python
alpha = 0.05

if p_value < alpha:
    print("Reject H0")
    print("Evidence of an association between sex and survival.")
else:
    print("Do not reject H0")
    print("Insufficient evidence of an association.")
```

Do **not** interpret this as:

```text
sex caused survival
```

This is observational historical data.

The test investigates **association**.

---

# 112. Effect Size — Cramér's V

Statistical significance alone is not enough.

Calculate an effect-size measure:

```python
n = sex_survival_table.to_numpy().sum()

rows, cols = sex_survival_table.shape

cramers_v = np.sqrt(
    chi2 /
    (
        n *
        min(rows - 1, cols - 1)
    )
)

print("Cramer's V:", cramers_v)
```

This gives additional information about the strength of association.

---

# 113. Hypothesis 2 — Passenger Class and Survival

Question:

> Is passenger class associated with survival?

Hypotheses:

```text
H₀:
Passenger class and survival are independent.

H₁:
Passenger class and survival are associated.
```

Create table:

```python
class_survival = pd.crosstab(
    clean["pclass"],
    clean["survived"]
)
```

Run:

```python
chi2, p_value, dof, expected = chi2_contingency(
    class_survival
)

print("Chi-square:", chi2)
print("p-value:", p_value)
```

---

# 114. Don't Stop at the p-value

Inspect the actual survival rates:

```python
clean.groupby(
    "pclass"
)["survived"].mean()
```

And percentages:

```python
pd.crosstab(
    clean["pclass"],
    clean["survived"],
    normalize="index"
) * 100
```

Statistics should complement understanding of the actual data.

---

# 115. Hypothesis 3 — Age and Survival

Now our question changes:

> Was passenger age different between survivors and non-survivors?

`age` is numerical.

`survived` creates two groups.

First create groups:

```python
survived_age = clean.loc[
    clean["survived"] == 1,
    "age"
]

not_survived_age = clean.loc[
    clean["survived"] == 0,
    "age"
]
```

---

# 116. Compare Descriptive Statistics First

```python
print(
    "Survivor mean:",
    survived_age.mean()
)

print(
    "Non-survivor mean:",
    not_survived_age.mean()
)

print(
    "Survivor median:",
    survived_age.median()
)

print(
    "Non-survivor median:",
    not_survived_age.median()
)
```

Never jump immediately to hypothesis tests.

First understand the data.

---

# 117. Welch's t-test

Hypotheses:

```text
H₀:
Mean age is equal between survivors and non-survivors.

H₁:
Mean age differs between survivors and non-survivors.
```

Run:

```python
from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(
    survived_age,
    not_survived_age,
    equal_var=False
)

print("t-statistic:", t_stat)
print("p-value:", p_value)
```

Using:

```python
equal_var=False
```

runs Welch's t-test, which does not assume equal population variances.

---

# 118. Important Assumptions

Before blindly using a t-test, discuss:

- independence of observations
- distribution shape
- outliers
- sample size
- whether comparing means is meaningful

Statistical tests should not be selected only because we know their names.

---

# 119. Hypothesis 4 — Fare and Survival

First compare:

```python
clean.groupby(
    "survived"
)["fare"].agg([
    "mean",
    "median",
    "std"
])
```

Then visualize:

```python
clean[
    clean["survived"] == 1
]["fare"].hist(
    bins=40,
    alpha=0.5,
    label="Survived"
)

clean[
    clean["survived"] == 0
]["fare"].hist(
    bins=40,
    alpha=0.5,
    label="Did not survive"
)

plt.legend()
plt.xlabel("Fare")
plt.title("Fare Distribution by Survival")
plt.show()
```

Notice that fare is highly skewed.

---

# 120. Mann–Whitney U Test

For heavily skewed data, we may demonstrate a non-parametric comparison.

```python
from scipy.stats import mannwhitneyu

survived_fare = clean.loc[
    clean["survived"] == 1,
    "fare"
]

not_survived_fare = clean.loc[
    clean["survived"] == 0,
    "fare"
]

u_stat, p_value = mannwhitneyu(
    survived_fare,
    not_survived_fare,
    alternative="two-sided"
)

print("U statistic:", u_stat)
print("p-value:", p_value)
```

The interpretation is not simply:

> "This proves survivors paid more."

The test evaluates differences in the distributions/ranks under its assumptions.

---

# 121. Hypothesis 5 — Travelling Alone and Survival

We created:

```text
is_alone
```

Now test it.

```python
alone_survival = pd.crosstab(
    clean["is_alone"],
    clean["survived"]
)

alone_survival
```

Chi-square:

```python
chi2, p_value, dof, expected = chi2_contingency(
    alone_survival
)

print("p-value:", p_value)
```

Then inspect actual rates:

```python
clean.groupby(
    "is_alone"
)["survived"].mean()
```

---

# 122. Feature Engineering Can Create New Hypotheses

This is an important connection.

Raw variables:

```text
sibsp
parch
```

created:

```text
family_size
```

Then:

```text
family_size
```

created:

```text
family_type
```

Now we can ask:

> Is family type associated with survival?

This shows how feature engineering is more than preparing machine-learning inputs.

It can also help us **think differently about the data**.

---

# 123. Test Family Type and Survival

Create:

```python
clean["family_type"] = pd.cut(
    clean["family_size"],
    bins=[0, 1, 4, np.inf],
    labels=[
        "Alone",
        "Small Family",
        "Large Family"
    ]
)
```

Table:

```python
family_survival = pd.crosstab(
    clean["family_type"],
    clean["survived"]
)

family_survival
```

Test:

```python
chi2, p_value, dof, expected = chi2_contingency(
    family_survival
)

print("p-value:", p_value)
```

---

# 124. Association Does Not Mean Causation

This is one of the most important lessons in the entire exercise.

Suppose we discover:

```text
higher fare ↔ higher survival
```

We must **not** immediately conclude:

```text
paying more money caused survival
```

Why?

Fare is related to:

```text
Passenger class
Cabin location
Wealth
Ticket arrangements
Other passenger characteristics
```

These relationships are called potential **confounding factors**.

---

# 125. Example of Confounding

Suppose:

```text
Fare → Survival
```

appears strong.

But:

```text
Fare ↔ Passenger Class
Passenger Class ↔ Survival
```

Therefore passenger class may explain some of the relationship.

Check:

```python
clean.groupby(
    "pclass"
)["fare"].median()
```

Then:

```python
clean.groupby(
    "pclass"
)["survived"].mean()
```

Now the relationship becomes more complicated.

This is how real data science works.

---

# 126. Ask Better Questions

Beginner question:

> Did fare affect survival?

Better analytical question:

> Is fare associated with survival in this dataset?

Even better:

> Does the relationship between fare and survival remain after considering passenger class and other variables?

The last question eventually leads toward:

- regression
- statistical modelling
- machine learning

---

# 127. Correlation

Select numerical variables:

```python
numeric_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "family_size",
    "fare_per_person"
]

correlations = clean[
    numeric_columns
].corr()

correlations
```

---

# 128. Correlation with Survival

```python
correlations[
    "survived"
].sort_values(
    ascending=False
)
```

Important:

> Correlation does not prove causation.

Also remember:

> Pearson correlation primarily describes linear association between numerical representations.

For categorical variables, different analytical approaches may be more appropriate.

---

# 129. A Complete Data Analysis Workflow

At this point students should understand this process:

```text
1. Define the question

2. Get the data

3. Inspect the data

4. Understand column meanings

5. Check data types

6. Check missing values

7. Clean data

8. Explore distributions

9. Compare groups

10. Engineer useful features

11. Form hypotheses

12. Select appropriate tests

13. Calculate statistics

14. Interpret results

15. Consider confounders and limitations

16. Prepare features for modelling
```

---

# 130. Prepare a Machine-Learning Dataset

Start with selected features:

```python
ml_data = clean[
    [
        "survived",
        "pclass",
        "sex",
        "age",
        "fare",
        "embarked",
        "family_size",
        "is_alone",
        "is_child",
        "fare_per_person",
        "age_missing"
    ]
].copy()
```

---

# 131. Encode Categorical Variables

```python
ml_data = pd.get_dummies(
    ml_data,
    columns=[
        "sex",
        "embarked"
    ],
    drop_first=True,
    dtype=int
)
```

Inspect:

```python
ml_data.head()
```

---

# 132. Check Everything is Numerical

```python
ml_data.dtypes
```

At this stage, the dataset is much closer to something that could be used by a machine-learning algorithm.

---

# 133. Separate Features and Target

Target:

```python
y = ml_data["survived"]
```

Features:

```python
X = ml_data.drop(
    columns="survived"
)
```

Check:

```python
X.head()
```

and:

```python
y.head()
```

This introduces the basic machine-learning structure:

```text
X = input features
y = target/output
```

---

# 134. What Did We Actually Do?

We started with:

```text
Raw Titanic CSV
```

Then performed:

```text
Data Loading
       ↓
Inspection
       ↓
NumPy Operations
       ↓
Pandas Operations
       ↓
Filtering
       ↓
Grouping
       ↓
Missing Value Analysis
       ↓
Cleaning
       ↓
EDA
       ↓
Feature Engineering
       ↓
Hypothesis Generation
       ↓
Statistical Testing
       ↓
Model-Ready Features
```

That is a major part of a real data-science workflow.

---

# 135. Important NumPy Commands Learned

```python
np.array()
np.mean()
np.median()
np.std()
np.min()
np.max()
np.sum()
np.nan
np.nanmean()
np.nanmedian()
np.where()
np.log1p()
```

Plus:

```text
indexing
slicing
boolean masks
vectorization
```

---

# 136. Important Pandas Commands Learned

```python
pd.read_csv()

df.head()
df.tail()
df.sample()

df.shape
df.columns
df.dtypes

df.info()
df.describe()

df["column"]

df[["column1", "column2"]]

df.loc[]
df.iloc[]

df.isna()
df.fillna()
df.dropna()

df.value_counts()

df.groupby()

df.agg()

df.sort_values()

pd.crosstab()

pd.cut()

pd.qcut()

pd.get_dummies()
```

---

# 137. Important Statistics Learned

Students should now recognize:

```text
Mean
Median
Standard deviation
Distribution
Percentage
Conditional comparison
Correlation
Null hypothesis
Alternative hypothesis
p-value
Significance level
Chi-square test
Welch's t-test
Mann–Whitney U test
Effect size
Confounding
Association vs causation
```

---

# 138. The Most Important Thinking Skill

The goal is **not** to memorize:

```python
df.groupby(...)
```

The important skill is learning to convert questions into data operations.

Example:

```text
Question:
Did women have different survival rates?

↓
Variables:
sex + survived

↓
Data operation:
groupby / crosstab

↓
Statistical question:
Are sex and survival associated?

↓
Statistical test:
Chi-square

↓
Interpretation:
Look at p-value + proportions + effect size + context
```

That is data thinking.

---

# 139. Guided Student Exercise

Students should answer the following without being given the complete code.

### Exercise 1

How many passengers were:

```text
male?
female?
```

---

### Exercise 2

Calculate:

```text
overall survival percentage
```

---

### Exercise 3

Calculate survival percentage by:

```text
sex
```

---

### Exercise 4

Calculate survival percentage by:

```text
passenger class
```

---

### Exercise 5

Find the average age by:

```text
survived
```

---

### Exercise 6

Find passengers who:

```text
were female
AND
were first class
AND
survived
```

---

### Exercise 7

Create:

```text
family_size
```

using:

```text
sibsp
parch
```

---

### Exercise 8

Create:

```text
is_child
```

where:

```text
age < 16
```

---

### Exercise 9

Create three age categories of your own.

For example:

```text
Young
Adult
Senior
```

---

### Exercise 10

Create a hypothesis about:

```text
embarked
```

and:

```text
survival
```

Write:

```text
H₀
H₁
```

Then choose an appropriate statistical test.

---

# 140. Advanced Student Challenge

Investigate the following claim:

> "First-class passengers survived more frequently than third-class passengers."

Students must provide:

```text
1. Relevant descriptive statistics
2. Survival percentages
3. Visualization
4. Null hypothesis
5. Alternative hypothesis
6. Statistical test
7. p-value
8. Interpretation
9. At least one possible confounding factor
10. A statement explaining why association does not prove causation
```

---

# 141. Feature Engineering Challenge

Students must create at least **five new features**.

Possible examples:

```text
family_size
is_alone
is_child
age_group
fare_per_person
fare_band
log_fare
age_missing
deck_known
family_type
```

For every feature, explain:

```text
1. Original columns used
2. Formula / transformation
3. Why the feature might be useful
4. What information could be lost
5. Whether the feature duplicates existing information
```

---

# 142. Hypothesis Challenge

Students must create three hypotheses.

Example structure:

## Hypothesis A

**Question**

Is passenger class associated with survival?

**H₀**

Passenger class and survival are independent.

**H₁**

Passenger class and survival are associated.

**Test**

Chi-square test.

---

## Hypothesis B

**Question**

Does passenger age differ between survivors and non-survivors?

**H₀**

The groups do not differ in the population quantity targeted by the selected test.

**H₁**

A difference exists.

**Possible test**

Welch's t-test or another appropriate comparison after checking assumptions.

---

## Hypothesis C

Students must create their own.

Possible variables:

```text
family_size
is_alone
embarked
fare
age_group
is_child
```

---

# 143. Final Mini Project

## Titanic Data Investigation

Each student should submit one Jupyter Notebook or Google Colab notebook containing:

### Section 1 — Dataset

- Download Titanic CSV
- Load with pandas
- Explain dataset shape
- Explain important columns

### Section 2 — NumPy

Demonstrate:

- arrays
- indexing
- slicing
- vectorization
- boolean filtering
- mean
- median
- standard deviation
- missing-value handling

### Section 3 — Pandas

Demonstrate:

- Series
- DataFrames
- filtering
- sorting
- `loc`
- `iloc`
- `value_counts`
- `groupby`
- aggregation
- crosstab

### Section 4 — Data Cleaning

Investigate:

- missing values
- missing percentages
- age
- embarked
- deck
- duplicates

Explain every cleaning decision.

### Section 5 — EDA

Answer at least five questions.

Example:

```text
What percentage survived?
How did survival differ by sex?
How did survival differ by class?
What was the age distribution?
How was fare distributed?
```

### Section 6 — Feature Engineering

Create at least five new features.

### Section 7 — Hypothesis Testing

Test at least three hypotheses.

For each hypothesis provide:

```text
Research Question

H₀

H₁

Statistical Test

Test Statistic

p-value

Decision at chosen α

Interpretation

Limitation
```

### Section 8 — Conclusions

Explain:

```text
What patterns were observed?

Which features appear useful?

What statistical relationships were found?

What can we NOT conclude?

What variables would you use in a future ML model?
```

---

# 144. Final Lesson

Data science is not:

```text
CSV
↓
Machine Learning Model
↓
Prediction
```

A better workflow is:

```text
Problem
↓
Question
↓
Data
↓
Understanding
↓
Cleaning
↓
Exploration
↓
Feature Engineering
↓
Hypothesis
↓
Evidence
↓
Model
↓
Evaluation
↓
Decision
```

The model is only one part of the process.

---

# 145. Main Takeaway

A beginner often asks:

> "Which machine-learning algorithm should I use?"

A better first set of questions is:

```text
What does my data mean?

What is missing?

What relationships exist?

Which variables represent the same information?

Could any variable leak the answer?

What useful features can I create?

What hypotheses can I investigate?

What evidence supports those hypotheses?

What assumptions am I making?
```

Once students begin asking these questions, they are moving from simply writing Python code toward **thinking like data and AI practitioners**.