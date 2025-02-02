from pyspark.sql import SparkSession
from pyspark.sql.functions import count, percentile_approx, broadcast, col, concat_ws, collect_list


def main():
    spark = SparkSession.builder \
        .appName("Spark") \
        .getOrCreate()

    input_folder = spark.conf.get("spark.app.inputPath")
    output_folder = spark.conf.get("spark.app.outputPath")
    df_crime_raw = spark.read.csv(input_folder+"/crime.csv", header=True, inferSchema=True)
    df_codes_raw = spark.read.csv(input_folder+"/offense_codes.csv", header=True, inferSchema=True)

    spark = SparkSession.builder.appName("ExampleApp").getOrCreate()
    
    df_crime_no_duplicates = df_crime_raw.distinct()
    data_crime_cleaned = df_crime_no_duplicates.collect()
    columns_crime = ["INCIDENT_NUMBER", "OFFENSE_CODE", "OFFENSE_CODE_GROUP", "OFFENSE_DESCRIPTION", "DISTRICT", "REPORTING_AREA", "SHOOTING", "OCCURRED_ON_DATE", "YEAR", "MONTH", "DAY_OF_WEEK", "HOUR", "UCR_PART", "STREET", "Lat", "Long", "Location"]
    df_crime_cleaned = spark.createDataFrame(data_crime_cleaned, columns_crime)

    df_codes_no_duplicates = df_codes_raw.distinct()
    data_codes_cleaned = df_codes_no_duplicates.collect()
    columns_codes = ["CODE", "NAME"]
    df_codes_cleaned = spark.createDataFrame(data_codes_cleaned, columns_codes)


    # create base sql view
    df_crime_base = df_crime_cleaned.groupBy("DISTRICT", "YEAR", "MONTH").count() \
                        .withColumnRenamed("count", "crimes") \
                        .orderBy("DISTRICT", "crimes")
    df_crime_base.createOrReplaceTempView("crime_base_sql")
    df_crime_base2 = df_crime_cleaned.select("DISTRICT", "Lat", "Long")
    df_crime_base2.createOrReplaceTempView("crime_base2_sql")

    # crimes total
    df_crimes_total = spark.sql("SELECT DISTRICT, sum(crimes) as crimes_total from crime_base_sql group by DISTRICT order by crimes_total")

    # crimes month median
    df_crimes_monthly = spark.sql("SELECT DISTRICT, percentile_approx(crimes, 0.5) as crimes_monthly  from crime_base_sql group by DISTRICT order by crimes_monthly")

    # crime location
    df_crime_location = spark.sql("SELECT DISTRICT, avg(Lat) as lat, avg(Long) as lng from crime_base2_sql group by DISTRICT")

    # create crime_type sql view
    df_join_type = df_crime_cleaned.join(broadcast(df_codes_cleaned), col("CODE") == col("OFFENSE_CODE"))
    df_join_type.createOrReplaceTempView("crime_join_sql")

    df_crime_type = spark.sql("SELECT DISTRICT, substring_index(NAME, ' - ', 1) as crime_type, count(*) as crimes FROM crime_join_sql group by DISTRICT, crime_type order by DISTRICT, crimes")
    df_crime_type.createOrReplaceTempView("crime_type_sql")

    # frequent crime types 
    df_freq_crime = spark.sql("""
        SELECT DISTRICT, crime_type, crimes 
        FROM (
            SELECT DISTRICT, crime_type, crimes, dense_rank() OVER (PARTITION BY DISTRICT ORDER BY crimes DESC) as rank 
            FROM crime_type_sql
        ) tmp 
        WHERE rank <= 3 
        ORDER BY DISTRICT, crimes DESC
    """)

    df_freq_group = df_freq_crime.groupBy("DISTRICT").agg(
        concat_ws(", ", collect_list("crime_type")).alias("frequent_crime_types")
    ).orderBy("DISTRICT")

    # summary table
    df_summary_1 = df_crimes_total.alias("a").join(df_crimes_monthly.alias("b"), col("a.DISTRICT") == col("b.DISTRICT"), "inner") \
        .select(col("a.DISTRICT").alias("DISTRICT"), col("a.crimes_total").alias("crimes_total"), col("b.crimes_monthly").alias("crimes_monthly"))

    df_summary_2 = df_summary_1.alias("c").join(df_freq_group.alias("d"), col("c.DISTRICT") == col("d.DISTRICT"), "inner") \
        .select(col("c.DISTRICT").alias("DISTRICT"), col("c.crimes_total").alias("crimes_total"), col("c.crimes_monthly").alias("crimes_monthly"), col("d.frequent_crime_types").alias("frequent_crime_types"))

    df_summary_3 = df_summary_2.alias("e").join(df_crime_location.alias("f"), col("e.DISTRICT") == col("f.DISTRICT"), "inner") \
        .select(col("e.DISTRICT").alias("DISTRICT"), col("e.crimes_total").alias("crimes_total"), col("e.crimes_monthly").alias("crimes_monthly"), col("e.frequent_crime_types").alias("frequent_crime_types"), col("f.lat").alias("lat"), col("f.lng").alias("lng"))

    df_summary_3.orderBy("crimes_total").repartition(1).write.mode("overwrite").parquet(output_folder)
    spark.stop

if __name__ == "__main__":
    main()
