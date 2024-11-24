import os
import pandas as pd
import re
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def extract_data_from_file(file_name):
    # [age]_[gender]_[race]_[date&time].jpg
    pattern = r"(\d+)_(\d)_(\d)_(\d+)\.jpg"
    match = re.match(pattern, file_name)

    if match:
        unique_id = file_name
        age = int(match.group(1))  # age
        gender = int(match.group(2))  # gender (0 = male, 1 = female)
        race = int(match.group(3))  # race (0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Others)
        date_time = match.group(4)  # date-time
        return unique_id, age, gender, race, date_time
    return None


def convert_to_meaningful_data(data):
    unique_id, age, gender, race, datetime_str = data

    # convert gender
    gender_str = "male" if gender == 0 else "female"

    # convert race
    race_dict = {
        0: "White",
        1: "Black",
        2: "Asian",
        3: "Indian",
        4: "Others"
    }
    race_str = race_dict.get(race)

    # convert date-time to a more readable format
    try:
        formatted_datetime = datetime.datetime.strptime(datetime_str[:14], "%Y%m%d%H%M%S")
        formatted_datetime_str = formatted_datetime.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        formatted_datetime_str = "Invalid date"

    return unique_id, age, gender_str, race_str, formatted_datetime_str


def get_dataframe(folders):
    all_data = []

    # Collect all data
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".jpg"):
                data = extract_data_from_file(filename)
                if data is not None:
                    meaningful_data = convert_to_meaningful_data(data)
                    full_filename = folder + '/' + filename
                    all_data.append(meaningful_data + (full_filename,))

    return pd.DataFrame(all_data, columns=["Unique-Identifier", "Age", "Gender", "Race", "DateTime", "FilePath"])


def create_csv(dataframe, file_name):
    dataframe.to_csv("data/" + file_name, index=False)


# as we have two wrong files:
# 55_0_0_20170116232725357JPG
# 44_1_4_20170116235150272.pg
def print_file_endings():
    folders = ['data/part1', 'data/part2', 'data/part3']
    format_counts = defaultdict(int)

    # checking all file-endings
    for folder in folders:
        for filename in os.listdir(folder):
            file_extension = filename.split('.')[-1].lower()
            format_counts[file_extension] += 1

    for ext, count in format_counts.items():
        print(f"Number of {ext.upper()} files: {count}")


def plot_age_histogram(dataframe, ax):
    sns.histplot(dataframe['Age'], bins=20, kde=True, ax=ax)
    ax.set_title('Age Distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.grid(True)


def plot_age_boxplot(dataframe):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=dataframe['Age'], color='lightblue')
    plt.title('Age Boxplot')
    plt.xlabel('Age')
    plt.grid(True)


def plot_gender_histogram(dataframe, ax):
    sns.countplot(x='Gender', data=dataframe, ax=ax)
    ax.set_title('Gender Distribution')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    ax.grid(True)


def plot_race_histogram(dataframe, ax):
    sns.countplot(x='Race', hue='Race', data=dataframe, palette='Set2', ax=ax, legend=False)
    ax.set_title('Race Distribution')
    ax.set_xlabel('Race')
    ax.set_ylabel('Count')
    ax.grid(True)


def plot_datetime_histogram(dataframe, ax):
    dataframe['DateTime'] = pd.to_datetime(dataframe['DateTime'], errors='coerce')
    dataframe['Month'] = dataframe['DateTime'].dt.to_period('M')

    # images per month
    month_counts = dataframe['Month'].value_counts().sort_index()

    month_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Images Collected by Month')
    ax.set_xlabel('Month (Year-Month)')
    ax.set_ylabel('Count')
    ax.grid(True)


def show_all_plots(dataframe, title):
    # 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    plot_age_histogram(dataframe, axs[0, 0])
    plot_gender_histogram(dataframe, axs[0, 1])
    plot_race_histogram(dataframe, axs[1, 0])
    plot_datetime_histogram(dataframe, axs[1, 1])
    fig.suptitle(title, fontsize=16)

    # prevent overlap
    plt.tight_layout()
    plt.show()

    # show boxplot
    plot_age_boxplot(dataframe)


def print_statistics(df):
    age_summary = df['Age'].describe()
    gender_counts = df['Gender'].value_counts()
    race_counts = df['Race'].value_counts()
    print(age_summary)
    print("\nGender Distribution:")
    for gender, count in gender_counts.items():
        print(f"\t{gender}: {count} ({(count / len(df) * 100):.2f}%)")

    print("\nRace Distribution:")
    for race, count in race_counts.items():
        print(f"\t{race}: {count} ({(count / len(df) * 100):.2f}%)")


def plot_age_bin_counts(df, title, age_bin_order):
    df['age_bin'] = pd.Categorical(df['age_bin'], categories=age_bin_order, ordered=True)
    age_counts = df['age_bin'].value_counts().reindex(age_bin_order)

    plt.figure(figsize=(10, 6))
    age_counts.plot(kind='bar', color='skyblue')
    plt.xlabel("Age Bins")
    plt.ylabel("Images")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()
