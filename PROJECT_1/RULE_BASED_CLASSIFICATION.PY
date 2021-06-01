import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# TASK-1
# I scanned the persona.csv file and showed general information about the dataset.
def check_df(dataframe, head=5):
    print(" - shape - ".upper().center(50, "*"))
    print(dataframe.shape)
    print(" - types - ".upper().center(50, "*"))
    print(dataframe.dtypes)
    print(" - head - ".upper().center(50, "*"))
    print(dataframe.head(head))
    print(" - tail - ".upper().center(50, "*"))
    print(dataframe.tail(head))
    print(" - info - ".upper().center(50, "*"))
    print(dataframe.info())
    print(" - columns - ".upper().center(50, "*"))
    print(dataframe.columns)
    print(" - values - ".upper().center(50, "*"))
    print(dataframe.values)
    print(" - index - ".upper().center(50, "*"))
    print(dataframe.index)
    print(" - nunique - ".upper().center(50, "*"))
    print(dataframe.nunique())
    print(" - na - ".upper().center(50, "*"))
    print(dataframe.isnull().sum())
    print(" - quantiles - ".upper().center(50, "*"))
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print(" - describe - ".upper().center(50, "*"))
    print(dataframe.describe().T)
    print()

def load_persona():
    return pd.read_csv('dataset/persona.csv')

df = load_persona()

check_df(df)

# ▪ How many unique 'SOURCE' in the dataset? What are their frequencies?
df["SOURCE"].unique()
# frequencies
df["SOURCE"].value_counts()

# ▪ how many unique 'PRICE' in the dataset?
df["PRICE"].unique()

# ▪ How many sales have been made from which 'PRICE'?
df["PRICE"].value_counts()

# ▪ How many sales were there from which 'COUNTRY'?
df["COUNTRY"].value_counts()

# ▪ How much was earned in total from sales by 'COUNTRY'?
df.groupby("COUNTRY").agg({"PRICE": "sum"})

# ▪ What are the sales numbers by 'SOURCE' types?
df["SOURCE"].value_counts()

# ▪ What are the 'PRICE' averages by <COUNTRY>?
df.groupby("COUNTRY").agg({"PRICE": "mean"})

# ▪ What are the 'PRICE' averages by 'SOURCE'?
df.groupby("SOURCE").agg({"PRICE": "mean"})

# ▪ What is the average PRICE in the 'COUNTRY-SOURCE' distribution?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# TASK-2
# ▪ What is the total earnings in distribution of 'COUNTRY, SOURCE, GENDER, AGE'?
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "sum"})

# TASK-3
# ▪ Sort the output by 'PRICE'. Apply in descending order according to 'PRICE'. Save the output as agg_df.
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "sum"}).sort_values(by="PRICE", ascending=False)

# TASK-4
# ▪ Convert the names in the index to variable names.
agg_df = agg_df.reset_index()
agg_df.head()

# TASK-5
# ▪ Convert AGE variable to categorical variable and add it to agg_df.
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0,18,24,30,40,70], labels=["0_18","19_23","24_30","31_40","41_70"])

# TASK-6
# ▪ Define customers (personas) based on new level.
# ▪ Define new level-based customers (personas) and add them as variables to the dataset.
# ▪ Name of new variable to add: customers_level_based
# ▪ You need to create the customer_level_based variable by combining the observations from the output of the previous question.
# ▪ My goal: I will make a sales strategy. I want to make this persona specific, which I think shows some common behavioral patterns, so I'll define high-level communication.

agg_df.values   # I checked the values

# ▪ 1st WAY
agg_df["customers_level_based"] = agg_df.apply(lambda row: row["COUNTRY"].upper() + "_" + row["SOURCE"].upper() + "_" + row["SEX"].upper() + "_" + row["AGE_CAT"].upper(), axis=1)

# ▪ 2nd WAY
col_names = ["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]
agg_df[col_names]

agg_df["customers_level_based"] = ["_".join(col).upper() for i, col in enumerate(agg_df[col_names].values)]

# ▪ 3rd WAY
agg_df["customers_level_based"] = ["_".join(row).upper() for row in agg_df[col_names].values]

agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df.value_counts().head()

# It looked like the customer_level_based index, I made a variable and converted the output of the PRICE variable to an int. I did the unification process. Because there can be many people with this personality.
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"}).astype(int).reset_index()
agg_df.head()

# TASK-7
# ▪ Segment new customers (personas).
# ▪ Divide new customers into 4 segments according to PRICE.
# ▪ Describe the segments.
# ▪ Analyze C segments
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B","A"])
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})
agg_df[agg_df["SEGMENT"]=="C"].describe().T


# TASK-8
# ▪ Categorize new customers by segment and estimate how much revenue they can generate.

# ▪ 1st WAY
# ▪ What segment does a 33-year-old Turkish woman using ANDROID belong to and how much income is expected on average?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user][["PRICE", "SEGMENT"]]

# ▪ In which segment and how much income on average would a 35-year-old French woman using iOS be expected to earn?
new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user2][["PRICE", "SEGMENT"]]

# ▪ 2nd WAY
def new_persona(dataframe, col, age, country, sources, sex):
    """
    Forecasts a new customer's revenue and segmentation based on user-entered information.

    Parameters
    ----------
    dataframe: Dataframe
        data set to make predictions
    col: string
        specified column name
    age: int
        determined age ranges
    country: string
        entered country
    sources: string
        desired sources
    sex: string
        desired sex
    Returns
    -------
    There isn't statement return

    """
    if (0 <= age <= 18):
        age_cat = "0_18"
    if (19 <= age <= 23):
        age_cat = "19_23"
    if (24 <= age <= 30):
        age_cat = "24_30"
    if (31 <= age <= 40):
        age_cat = "31_40"
    if (41 <= age <= 70):
        age_cat = "41_70"

    new_user = country.upper() + "_" + sources.upper() + "_" + sex.upper() + "_" + age_cat
    print(dataframe[dataframe[col] == new_user][["PRICE", "SEGMENT"]].to_string(index=False))

new_persona(agg_df, "customers_level_based", 33, "tur", "androıd", "female")

new_persona(agg_df, "customers_level_based", 35, "fra", "ıos", "female")

help(new_persona)

# For analysis
def summary(dataframe, col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[col].describe(quantiles).T)
    if plot:
        dataframe[col].hist()
        plt.show()
    if plot:
        plt.boxplot(dataframe[col])
        plt.show()
    if plot:
        sns.barplot(x="SEGMENT", y="PRICE", data=dataframe)
        plt.show()
    if plot:
        sns.catplot(x="SEGMENT", y="PRICE", data=agg_df)
        plt.show()

summary(agg_df, "PRICE", plot=True)

# for analysis
def segmentt(dataframe, segment:str):
    if segment == "C":
        return dataframe[dataframe["SEGMENT"] == "C"]["PRICE"].sum()
    if segment == "D":
        return dataframe[dataframe["SEGMENT"] == "D"]["PRICE"].sum()
    if segment == "B":
        return dataframe[dataframe["SEGMENT"] == "B"]["PRICE"].sum()
    if segment == "A":
        return dataframe[dataframe["SEGMENT"] == "A"]["PRICE"].sum()

segment=["D", "C", "B","A"]

for col in segment:
    print(segmentt(agg_df, col))
    print("***********")


segmentt(agg_df, "C")

segmentt(agg_df, "A")

segmentt(agg_df, "B")

segmentt(agg_df, "D")

# Analysis result: People with the most "A" segments earned more money.