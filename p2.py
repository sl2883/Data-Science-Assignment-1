import pickle

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
import scipy
from pyspark.sql import SparkSession, SQLContext

import random

def color_negative_red(val):
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

# http://queirozf.com/entries/pandas-dataframe-plot-examples-with-matplotlib-pyplot
# https://github.com/ResidentMario/missingno
# https://stackoverflow.com/questions/45069828/how-to-plot-2-histograms-side-by-side


def assignment_part_2():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    extrasensory_individual_data = pickle.load(open("Extrasensory_individual_data.p", "rb"))
    extrasensory_sensor_data = pickle.load(open("Extrasensory_sensor_data.p", "rb"))

    df = pd.concat(extrasensory_sensor_data)

    print("Checking values as Null in extrasensory_sensor_data - ")
    print(df.isnull().any())
    print("Checking values as NaN in extrasensory_sensor_data - ")
    print(df.isna().any())
    # df.info()
    # msno.bar(df)

    # msno.matrix(df)
    # df.plot(kind='scatter',x='location:raw_latitude',y='location:raw_longitude',color='red')
    # new_df.hist(column="discrete:app_state:is_active")

    df2 = pd.DataFrame(extrasensory_individual_data)

    print("Checking values as Null in extrasensory_individual_data - ")
    print(df2.isnull().any())
    print("Checking values as NaN in extrasensory_individual_data - ")
    print(df2.isna().any())
    # df2.info()
    # msno.bar(df2)

    # column_name = "perceived_average_screen_time"
    column_name = "actual_average_screen_time"

    fig, axes = plt.subplots(1, 5)

    # df2.plot(kind='scatter',x='age',y='location:raw_longitude',color='red')

    df2.hist(column=column_name, bins=10, ax=axes[0])
    df2 = df2.dropna()

    # print("\nNegative numbers red and positive numbers black:")
    # df2 = df2.style.applymap(color_negative_red)
    #
    print(df2)

    df_mean = df2.copy()
    df_median = df2.copy()
    df_random = df2.copy()

    df_filter_negative = df_mean[column_name].isin([-1.0])
    column_name_mean = df_mean[column_name].sum() / (
                len(df_mean.index) - len(df_mean[df_filter_negative].index))
    print("Mean is = ", column_name_mean)
    df_mean[column_name] = df_mean[column_name] \
        .apply(lambda x: column_name_mean if x == -1 else x)

    df_mean.hist(column=column_name, bins=5, ax=axes[1])

    df_filter_not_negative = df_median[df_median[column_name] != -1.0]
    column_name_median = df_filter_not_negative[column_name].median()
    print("Median is = ", column_name_median)
    df_median[column_name] = df_median[column_name] \
        .apply(lambda x: column_name_median if x == -1 else x)

    df_median.hist(column=column_name, bins=5, ax=axes[2])

    column_name_random = df_random[column_name].min() \
                                           + (df_random[column_name].max()
                                              - df_random[column_name].min()) \
                                           * random.uniform(0, 1)
    print("Random is ", column_name_random)
    df_random[column_name] = df_random[column_name] \
        .apply(lambda x: column_name_random if x == -1 else x)

    df_random.hist(column=column_name, bins=5, ax=axes[3])

    df_normal = pd.DataFrame(np.random.normal(3.75, 1.25, 60), columns={"val"})
    df_normal.hist(column="val", bins=5, ax=axes[4])

    axes[0].set_title("Original")
    axes[1].set_title("Mean = %.1f" % column_name_mean)
    axes[2].set_title("Median = %.1f" % column_name_median)
    axes[3].set_title("Rand = %.1f" % column_name_random)
    axes[3].set_title("Normal")

    ttest_mean = scipy.stats.ttest_ind(df_normal["val"], df_mean[column_name])
    ttest_median = scipy.stats.ttest_ind(df_normal["val"], df_median[column_name])
    ttest_random = scipy.stats.ttest_ind(df_normal["val"], df_random[column_name])

    print("T-test results - ")
    print("Mean - ", ttest_mean)
    print("Median - ", ttest_median)
    print("Rand - ", ttest_random)

    plt.show()

def assignment_part_3():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    extrasensory_individual_data = pickle.load(open("Extrasensory_individual_data.p", "rb"))

    df2 = pd.DataFrame(extrasensory_individual_data)

    column_name = "perceived_average_screen_time"
    fig, axes = plt.subplots(1, 5)

    df2.hist(column=column_name, bins=10, ax=axes[0])
    df2 = df2.dropna()

    print(df2)

    df_mean = df2.copy()
    df_chi = df2.copy()
    df_random = df2.copy()

    df_filter_negative = df_mean[column_name].isin([-1.0])
    column_name_mean = df_mean[column_name].sum() / (
                len(df_mean.index) - len(df_mean[df_filter_negative].index))
    print("Mean is = ", column_name_mean)
    df_mean[column_name] = df_mean[column_name] \
        .apply(lambda x: column_name_mean if x == -1 else x)

    df_std = df_mean[column_name].std()
    print("Standard Deviation is  ", df_std)

    intense_users = df_mean[column_name] \
        .apply(lambda x: 1 if x > (column_name_mean + df_std) else 0)

    print("Number of intense users is ", intense_users.sum())

    df_chi["missing"] = df_mean[column_name] \
        .apply(lambda x: True if x == -1 else False)
    df_chi["intense"] = df_mean[column_name] \
        .apply(lambda x: True if x > (column_name_mean + df_std) else False)

    contingency_table = pd.crosstab(
        df_chi["missing"],
        df_chi["intense"],
        margins=True
    )
    print(contingency_table)

    chi2 = scipy.stats.chi2_contingency(contingency_table)
    print(chi2)


def assignment_part_4_A():
    extrasensory_sensor_data = pickle.load(open("Extrasensory_sensor_data.p", "rb"))
    df = pd.concat(extrasensory_sensor_data)

    df_NaN = df[(df["location:raw_latitude"].isna()) & (df["lf_measurements:battery_level"] < 0.20)]
    uuids = df_NaN.index.get_level_values(0).unique()

    df_NaN.index.names = ["uuid", "eventid"]
    print(df_NaN.groupby(level=[0]).size())

    print("Number of unique users with battery low and no location data is ", len(uuids))

def assignment_part_4_B():
    extrasensory_sensor_data = pickle.load(open("Extrasensory_sensor_data.p", "rb"))
    df_subject = extrasensory_sensor_data.get("F50235E0-DD67-4F2A-B00B-1F31ADA998B9")
    print(df_subject)

    fig, axes = plt.subplots(1, 5)

    df_subject.index.names = ["id"]

    df_subject["location:raw_latitude_back"] = df_subject["location:raw_latitude"]
    df_subject["location:raw_latitude_fwd"] = df_subject["location:raw_latitude"]
    df_subject["location:raw_latitude_lp"] = df_subject["location:raw_latitude"]

    df_subject["location:raw_latitude_back"] = df_subject["location:raw_latitude_back"]\
        .replace(to_replace=None, method='ffill')
    df_subject["location:raw_latitude_fwd"] = df_subject["location:raw_latitude_back"]\
        .replace(to_replace=None, method='bfill')

    df_subject["location:raw_latitude_lp"] = df_subject["location:raw_latitude_lp"]\
        .interpolate(method ='linear', limit_direction ='forward')

    df_subject.plot.line(y=['location:raw_latitude'], ax=axes[0])

    df_subject.plot.line(y=['location:raw_latitude_back'], ax=axes[1])
    df_subject.plot.line(y=['location:raw_latitude_fwd'], ax=axes[2])
    df_subject.plot.line(y=['location:raw_latitude_lp'], ax=axes[3])
    df_subject.plot.line(y=['location:raw_latitude', 'location:raw_latitude_back', 'location:raw_latitude_fwd', 'location:raw_latitude_lp'], ax=axes[4], alpha=0.7)

    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    axes[3].get_legend().remove()
    axes[4].get_legend().remove()

    axes[0].set_title("Raw")
    axes[1].set_title("Fwd")
    axes[2].set_title("Back")
    axes[3].set_title("LP")
    axes[4].set_title("All")

    plt.show()


def create_dataframe(filepath, format, spark):
     sqlContext = SQLContext(spark)
     spark_df = sqlContext.read.format(format).options(header="True", inferschema='true').load(filepath)
     return spark_df

def assignment_viz():
    spark = SparkSession.builder.getOrCreate()
    df = create_dataframe('weather.csv', 'csv', spark)
    print(df)

if __name__ == '__main__':
    # assignment_part_2()
    # assignment_part_3()
    # assignment_part_4()
    # assignment_part_4_B()
    assignment_viz()


