# real_time_processing.py

"""
Real-Time Processing Module for Energy Consumption Optimization in Smart Grids

This module contains functions for real-time data ingestion, processing, and integration 
to adapt to changing conditions and optimize energy consumption and distribution in smart grids.

Tools Used:
- Apache Kafka
- Apache Spark
- Real-time databases
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import kafka
from kafka import KafkaProducer, KafkaConsumer
import json

class RealTimeProcessing:
    def __init__(self, kafka_bootstrap_servers='localhost:9092', topic='energy_data'):
        """
        Initialize the RealTimeProcessing class.
        
        :param kafka_bootstrap_servers: str, Kafka bootstrap servers
        :param topic: str, Kafka topic to consume and produce data
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.topic = topic
        self.spark = SparkSession.builder.appName("EnergyConsumptionOptimization").getOrCreate()
        
        self.schema = StructType([
            StructField("timestamp", TimestampType(), True),
            StructField("energy_consumption", DoubleType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("humidity", DoubleType(), True)
        ])

    def read_from_kafka(self):
        """
        Read real-time data from Kafka topic.
        
        :return: DataFrame, real-time data
        """
        return self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers) \
            .option("subscribe", self.topic) \
            .load() \
            .selectExpr("CAST(value AS STRING)") \
            .select(
                col("value").cast("string").alias("json_data")
            ) \
            .select(
                from_json(col("json_data"), self.schema).alias("data")
            ) \
            .select("data.*")

    def write_to_kafka(self, df):
        """
        Write processed data to Kafka topic.
        
        :param df: DataFrame, processed data
        """
        df.selectExpr("to_json(struct(*)) AS value") \
            .writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers) \
            .option("topic", self.topic) \
            .option("checkpointLocation", "/tmp/kafka_checkpoint") \
            .start() \
            .awaitTermination()

    def process_data(self, df):
        """
        Process real-time data to extract relevant features and perform necessary transformations.
        
        :param df: DataFrame, input data
        :return: DataFrame, processed data
        """
        df = df.withColumn("hour", hour(col("timestamp"))) \
               .withColumn("day_of_week", dayofweek(col("timestamp"))) \
               .withColumn("month", month(col("timestamp")))
        return df

    def start_processing(self):
        """
        Start the real-time data processing pipeline.
        """
        raw_data = self.read_from_kafka()
        processed_data = self.process_data(raw_data)
        self.write_to_kafka(processed_data)

if __name__ == "__main__":
    rtp = RealTimeProcessing()
    rtp.start_processing()
