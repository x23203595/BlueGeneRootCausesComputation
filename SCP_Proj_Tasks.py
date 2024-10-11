from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import split
from mpi4py import MPI
import datetime

log_file_path = "BGL.log"

# Initialize SparkSession
spark = SparkSession.builder.appName("log_analysis").getOrCreate()

# Define schema for the log data
log_schema = StructType([
    StructField("Alert_Message_Flag", StringType(), nullable=True),
    StructField("Timestamp", StringType(), nullable=True),
    StructField("Date", StringType(), nullable=True),
    StructField("Node", StringType(), nullable=True),
    StructField("Date_and_Time", StringType(), nullable=True),
    StructField("Node_Repeated", StringType(), nullable=True),
    StructField("RAS_Message_Type", StringType(), nullable=True),
    StructField("System_Component", StringType(), nullable=True),
    StructField("Level", StringType(), nullable=True),
    StructField("Message_Content", StringType(), nullable=True)
])

# ================================
# Spark DataFrame Operations
# ================================
# Read the log file into a DataFrame with the specified schema
log_df = spark.read.csv(log_file_path, schema=log_schema, sep=" ", header=False)

# Show the transformed DataFrame schema and data
log_df.printSchema()
log_df.show(truncate=False)

# ================================
# MapReduce Operations
# ================================
# 3. How many fatal log entries in the months of December or January resulted from an ”invalid or missing program image”?

# Adapted MapReduce Job
def map_fatallogentries(lines):
    for line in lines:
        # Split the line contents into a list of columns using the space delimiter
        columns = line.split(" ")
        
        # Extract the full date from the third column
        full_date = columns[2]
        
        # Extract year and month_number from the date
        year = full_date[0:4]
        month_number = full_date[5:7]
        
        # Map month_number to month_of_year
        if month_number == "12":
            month_of_year = "December"
        elif month_number == "01":
            month_of_year = "January"
        else:
            continue
        
        # Extract the Message Content Column for the message
        # Message content starts from column 9 onward
        message_content = " ".join(columns[9:])
        
        # If month of year lies in December or January and the message content is "invalid or missing program image"
        if "invalid or missing program image" in message_content:
            # Output the year and month as key, and 1 as value
            yield year + "-" + month_of_year, 1

def reduce_fatallogentries(mapped_data):
    reduced_data = {}
    for key, value in mapped_data:
        if key not in reduced_data:
            reduced_data[key] = 0
        reduced_data[key] += value
    return reduced_data

def reduce_count_entries(reduced_data):
    total_count = sum(reduced_data.values())
    return {"Total fatal log entries in December or January due to 'invalid or missing program image'": total_count}

# Read the log file
with open(log_file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Perform the MapReduce job
mapped_data = list(map_fatallogentries(lines))
reduced_data = reduce_fatallogentries(mapped_data)
final_result = reduce_count_entries(reduced_data)

# Print the result
for key, value in final_result.items():
    print(key + ": " + str(value))

# ================================
# Spark RDD Operations
# ================================
# 8. For each hour of the day, calculate the average number of seconds during which L3 EDRAM error(s) were detected and corrected

def parse_and_filter(line):
    columns = line.split(" ")
    if len(columns) > 9:
        date_and_time = columns[4]
        message_content = " ".join(columns[9:])
        if "L3 EDRAM error" in message_content:
            try:
                dt_obj = datetime.datetime.strptime(date_and_time, '%Y-%m-%d-%H.%M.%S.%f')
                hour = dt_obj.hour
                seconds = dt_obj.second
                return (hour, (seconds, 1))
            except ValueError:
                pass
    return None

# Read the log file into an RDD
log_rdd = spark.sparkContext.textFile(log_file_path)

# Apply parsing and filtering
parsed_rdd = log_rdd.map(parse_and_filter).filter(lambda x: x is not None)
reduced_rdd = parsed_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
avg_seconds_rdd = reduced_rdd.mapValues(lambda x: x[0] / x[1])
result = avg_seconds_rdd.collect()

print("Average number of seconds during which L3 EDRAM error(s) were detected and corrected per hour:")
for hour, avg_seconds in sorted(result):
    print(f"Hour: {hour}, Average Seconds: {avg_seconds}")

# ================================
# Spark SQL Operations
# ================================
# 9. What are the top 5 most frequently occurring dates in the log?

log_df.createOrReplaceTempView("log_table")
sql_query = """
SELECT
    Date,
    COUNT(*) AS Count
FROM
    log_table
GROUP BY
    Date
ORDER BY
    Count DESC
LIMIT 5
"""
top_dates_df = spark.sql(sql_query)
top_dates = top_dates_df.collect()
print("Top 5 most frequently occurring dates:")
for row in top_dates:
    print(f"Date: {row['Date']}, Count: {row['Count']}")

# ================================
# Spark RDD Operations for APPUNAV Events
# ================================
# 16. Which node generated the largest number of APPUNAV events?

def parse_log_line(line):
    columns = line.split(" ")
    if len(columns) >= 10:
        alert_message_flag = columns[0]
        node = columns[3]
        return (alert_message_flag, node)
    return None

appunav_rdd = log_rdd.map(parse_log_line).filter(lambda x: x is not None and x[0] == 'APPUNAV')
node_counts_rdd = appunav_rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b)
max_node = node_counts_rdd.takeOrdered(1, key=lambda x: -x[1])

print("Node with the largest number of APPUNAV events:")
for node, count in max_node:
    print(f"Node: {node}, Events: {count}")

# Stop SparkSession
spark.stop()

# ================================
# MPI
# ================================
# 18. On which date and time was the earliest fatal kernel error where the message contains ”Lustre mount FAILED”?

class DateAndTimeExecutor:
    def __init__(self, date_and_time_col, log_file_path, rank, size):
        self.date_and_time_col = date_and_time_col
        self.log_file_path = log_file_path
        self.rank = rank
        self.size = size
        self.earliest_date = None
        self.message_content = None

    def call(self):
        date_format = '%Y-%m-%d-%H.%M.%S.%f'
        with open(self.log_file_path, 'r', encoding='utf-8', errors='replace') as file:
            for i, line in enumerate(file):
                if i % self.size != self.rank:
                    continue

                if "Lustre mount FAILED" in line:
                    columns = line.split(" ")
                    if self.date_and_time_col - 1 < len(columns):
                        date_str = columns[self.date_and_time_col - 1]
                        try:
                            log_date = datetime.datetime.strptime(date_str, date_format)
                            if self.earliest_date is None or log_date < self.earliest_date:
                                self.earliest_date = log_date
                                self.message_content = line
                        except ValueError:
                            continue
        return self.message_content

    def get_earliest_date(self):
        return self.earliest_date

def main_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    executor = DateAndTimeExecutor(5, log_file_path, rank, size)
    local_earliest_error = executor.call()
    local_earliest_date = executor.get_earliest_date()
    local_timestamp = local_earliest_date.timestamp() if local_earliest_date else float('inf')

    local_data = [local_timestamp, rank]
    global_data = comm.allreduce(local_data, op=MPI.MINLOC)
    global_earliest_error = local_earliest_error if rank == global_data[1] else None
    global_earliest_error = comm.bcast(global_earliest_error, root=global_data[1])

    if rank == 0 and global_earliest_error:
        print(f"Earliest fatal kernel error with 'Lustre mount FAILED' occurred at: {global_earliest_error}")

if __name__ == "__main__":
    main_mpi()

# ================================
# MapReduce for Specific Log Messages
# ================================
# Which node has the highest occurrences of 'trap instruction' over 'imprecise exception'?

def map_node_counts(lines):
    for line in lines:
        columns = line.split(" ")
        if len(columns) >= 10:
            node = columns[3]
            message_content = " ".join(columns[9:])
            if "trap instruction" in message_content or "imprecise exception" in message_content:
                yield node, 1

def reduce_node_counts(mapped_data):
    node_counts = {}
    for key, value in mapped_data:
        if key not in node_counts:
            node_counts[key] = 0
        node_counts[key] += value
    return node_counts

def find_max_node(reduced_data):
    if not reduced_data:
        return None, 0
    max_node = max(reduced_data, key=reduced_data.get)
    return max_node, reduced_data[max_node]

def main_mapreduce(log_file_path):
    with open(log_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        
    mapped_data = list(map_node_counts(lines))
    reduced_data = reduce_node_counts(mapped_data)
    max_node, max_count = find_max_node(reduced_data)
    
    if max_node:
        print(f"Node with the highest occurrences: {max_node}")
        print(f"Number of occurrences: {max_count}")
    else:
        print("No occurrences found.")

# Example usage for MapReduce
log_file_path = "BGL.log"  # Path to BGL.log
main_mapreduce(log_file_path)

# ================================
# SparkRDD for ASSERT expression=0
# ================================
# Determine if at least 10% of all messages from January 2005 to September 2005 are ASSERT expression=0.

from mpi4py import MPI
import datetime

def parse_line(line):
    columns = line.split(" ")
    if len(columns) < 10:
        return None
    return {
        "Alert_Message_Flag": columns[0],
        "Timestamp": columns[1],
        "Date": columns[2],
        "Node": columns[3],
        "Date_and_Time": columns[4],
        "Node_Repeated": columns[5],
        "RAS_Message_Type": columns[6],
        "System_Component": columns[7],
        "Level": columns[8],
        "Message_Content": " ".join(columns[9:])
    }

def filter_logs(logs, start_date, end_date):
    filtered = []
    for log in logs:
        if log is None:
            continue
        try:
            log_date = datetime.datetime.strptime(log['Date'], '%Y.%m.%d')
            if start_date <= log_date <= end_date:
                filtered.append(log)
        except ValueError:
            continue
    return filtered

def count_assert_expression_0(logs):
    count = 0
    for log in logs:
        if 'ASSERT expression=0' in log['Message_Content']:
            count += 1
    return count

def main_assert_expression():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    log_file_path = "BGL.log"

    if rank == 0:
        try:
            with open(log_file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            chunk_size = len(lines) // size
            chunks = [lines[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
            if len(lines) % size != 0:
                chunks[-1].extend(lines[size * chunk_size:])
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        chunks = None

    chunk = comm.scatter(chunks, root=0)
    parsed_logs = [parse_line(line) for line in chunk]

    start_date = datetime.datetime.strptime('2005-01-01', '%Y-%m-%d')
    end_date = datetime.datetime.strptime('2005-09-30', '%Y-%m-%d')
    filtered_logs = filter_logs(parsed_logs, start_date, end_date)

    local_total_count = len(filtered_logs)
    local_assert_expression_0_count = count_assert_expression_0(filtered_logs)
    total_count = comm.reduce(local_total_count, op=MPI.SUM, root=0)
    assert_expression_0_count = comm.reduce(local_assert_expression_0_count, op=MPI.SUM, root=0)

    if rank == 0:
        if total_count > 0:
            percentage = (assert_expression_0_count / total_count) * 100
            print(f"Total messages from January 2005 to September 2005: {total_count}")
            print(f"'ASSERT expression=0' messages: {assert_expression_0_count}")
            print(f"Percentage of 'ASSERT expression=0': {percentage:.2f}%")
            if percentage >= 10:
                print("At least 10% of the messages are 'ASSERT expression=0'.")
            else:
                print("Less than 10% of the messages are 'ASSERT expression=0'.")
        else:
            print("No relevant messages found.")

if __name__ == "__main__":
    main_assert_expression()
