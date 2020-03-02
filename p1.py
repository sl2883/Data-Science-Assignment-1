# Imports
import sys
from pyspark.sql import SparkSession, SQLContext
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, StringType

def create_dataframe(filepath, format, spark):
    """
    Create a spark df given a filepath and format.

    :param filepath: <str>, the filepath
    :param format: <str>, the file format (e.g. "csv" or "json")
    :param spark: <str> the spark session

    :return: the spark df uploaded
    """
    sqlContext = SQLContext(spark)
    spark_df = sqlContext.read.format(format).options(header="True", inferschema='true').load(filepath)

    return spark_df


def transform_nhis_data(nhis_df):
    """
    Transform df elements

    :param nhis_df: spark df
    :return: spark df, transformed df
    """

    nhis_df_udf_1 = F.udf(map_hisp_mrac_to_imprace, IntegerType())
    nhis_df = nhis_df.withColumn("hisp_mrac_to_imprace", nhis_df_udf_1(nhis_df.HISPAN_I, nhis_df.MRACBPI2))
    nhis_df_udf_2 = F.udf(map_agep_to_ageg5yr, IntegerType())
    nhis_df = nhis_df.withColumn("agep_to_ageg5yr", nhis_df_udf_2(nhis_df.AGE_P))
    transformed_df = nhis_df.withColumnRenamed("SEX", "SEX_1")

    return transformed_df

def is_diabatic(inp):
    if inp == 1:
        return 1
    else:
        return 0

def calculate_statistics(joined_df):
    """
    Calculate prevalence statistics

    :param joined_df: the joined df

    :return: None
    """
    joined_df_1 = F.udf(is_diabatic, IntegerType())
    joined_df_temp = joined_df.withColumn("is_diabatic", joined_df_1(joined_df.DIBEV1))
    joined_df_temp.groupBy('_IMPRACE').mean("is_diabatic").show()
    joined_df_temp.groupBy('SEX').mean("is_diabatic").show()
    joined_df_temp.groupBy('_AGEG5YR').mean("is_diabatic").show()

def map_hisp_mrac_to_imprace(hisp, mrac):
    """
    # output
    # 1 White, Non - Hispanic
    # 2 Black, Non-Hispanic
    # 3 Asian, Non-Hispanic
    # 4 American Indian/Alaskan Native, Non-Hispanic
    # 5 Hispanic
    # 6 Other race, Non-Hispanic

    HISPAN_I
        00 Multiple Hispanic
        01 Puerto Rico
        02 Mexican
        03 Mexican-American
        04 Cuban/Cuban American
        05 Dominican (Republic)
        06 Central or South American
        07 Other Latin American, type not specified
        08 Other Spanish
        09 Hispanic/Latino/Spanish, non-specific type
        10 Hispanic/Latino/Spanish, type refused
        11 Hispanic/Latino/Spanish, type not ascertained
        12 Not Hispanic/Spanish origin

    MRACBPI2
        01 White
        02 Black/African American
        03 Indian (American) (includes Eskimo, Aleut)
        06 Chinese
        07 Filipino
        12 Asian Indian
        16 Other race*
        17 Multiple race, no primary race selected
    """
    is_hisp = (int(hisp) == 1
               or int(hisp) == 2
               or int(hisp) == 3
               or int(hisp) == 4
               or int(hisp) == 5
               or int(hisp) == 6
               or int(hisp) == 7
               or int(hisp) == 8
               or int(hisp) == 12)

    if int(mrac) == 1 \
            and is_hisp:
        ret = 1
    elif int(mrac) == 2 \
            and is_hisp:
        ret = 2
    elif (int(mrac) == 6
          or int(mrac) == 7
          or int(mrac) == 12) \
            and is_hisp:
        ret = 3
    elif int(mrac) == 3 \
            and is_hisp:
        ret = 4
    elif (int(mrac) == 16
          or int(mrac) == 17) \
            and is_hisp:
        ret = 6
    else:
        ret = 5

    return ret


def map_agep_to_ageg5yr(agep):
    ret = -1

    if 18 <= int(agep) <= 24:
        ret = 1
    elif 25 <= int(agep) <= 29:
        ret = 2
    elif 30 <= int(agep) <= 34:
        ret = 3
    elif 35 <= int(agep) <= 39:
        ret = 4
    elif 40 <= int(agep) <= 44:
        ret = 5
    elif 45 <= int(agep) <= 49:
        ret = 6
    elif 50 <= int(agep) <= 54:
        ret = 7
    elif 55 <= int(agep) <= 59:
        ret = 8
    elif 60 <= int(agep) <= 64:
        ret = 9
    elif 65 <= int(agep) <= 69:
        ret = 10
    elif 70 <= int(agep) <= 74:
        ret = 11
    elif 75 <= int(agep) <= 79:
        ret = 12
    elif 80 <= int(agep):
        ret = 13
    elif 7 <= int(agep) <= 9:
        ret = 14

    return ret


def join_dfs(nhis_df, brfss_df):

    joined_df = brfss_df.join(nhis_df
                               , [(nhis_df.hisp_mrac_to_imprace == brfss_df._IMPRACE)
                                  & (nhis_df.agep_to_ageg5yr == brfss_df._AGEG5YR)
                                  & (nhis_df.SEX_1 == brfss_df.SEX)], how="inner")

    joined_df = joined_df.drop("MRACBPI2")
    joined_df = joined_df.drop("HISPAN_I")
    joined_df = joined_df.drop("AGE_P")

    joined_df = joined_df.drop("hisp_mrac_to_imprace")
    joined_df = joined_df.drop("agep_to_ageg5yr")

    # joined_df = brfss_df.join(nhis_df, [(nhis_df.SEX_1 == brfss_df.SEX)], how="inner")
    joined_df = joined_df.drop("SEX_1")

    return joined_df

def save_test_file(df, path, format):
    df.write.format(format).mode('overwrite').option("header", "true").save('test_'+path)

def save_file(df, path, format):
    df.write.format(format).mode('overwrite').option("header", "true").save(path)


def save_all_test_file(nhis_df, nhis_file_arg, csv_format, brfss_df, brfss_file_arg, json_format):
    test_nhis_df = nhis_df.select("*").limit(100)
    save_test_file(test_nhis_df, nhis_file_arg, csv_format)
    test_brfss_df = brfss_df.select("*").limit(100)
    save_test_file(test_brfss_df, brfss_file_arg, json_format)

if __name__ == '__main__':

    csv_format = 'csv'
    json_format = "json"

    brfss_file_arg = sys.argv[1]
    nhis_file_arg = sys.argv[2]
    save_output_arg = sys.argv[3]
    if save_output_arg == "True":
        output_filename_arg = sys.argv[4]
    else :
        output_filename_arg = None

    # Start spark session
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    nhis_df = create_dataframe(nhis_file_arg, csv_format, spark)
    brfss_df = create_dataframe(brfss_file_arg, json_format, spark)

    # Uncoment this to create test files again
    # save_all_test_file(nhis_df, nhis_file_arg, csv_format, brfss_df, brfss_file_arg, json_format)

    transformed_nhis_df = transform_nhis_data(nhis_df)

    joined_df = join_dfs(transformed_nhis_df, brfss_df)
    joined_df.na.drop()

    if output_filename_arg:
        save_file(joined_df, output_filename_arg, csv_format)

    # joined_df = create_dataframe("joined.csv", csv_format, spark)
   
    calculate_statistics(joined_df)

    # Stop spark session 
    spark.stop()
