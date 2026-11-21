"""
spark_setup.py
--------------
Creates a local SparkSession on Windows (JDK 17, PySpark 4).
local[*] uses all CPU cores as executors - the "big data" parallelism used in
the PySpark experiments.
"""
import os
import sys
import glob


def get_spark(app="hurricane-bigdata", cores="*"):
    if not os.environ.get("JAVA_HOME"):
        jdks = sorted(glob.glob(r"C:\Program Files\Eclipse Adoptium\jdk-17*"))
        if jdks:
            os.environ["JAVA_HOME"] = jdks[-1]
            os.environ["PATH"] = os.path.join(jdks[-1], "bin") + os.pathsep + os.environ["PATH"]
    # the Python workers must use the same interpreter as the driver
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    src = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = src + os.pathsep + os.environ.get("PYTHONPATH", "")

    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName(app)
             .master(f"local[{cores}]")
             .config("spark.driver.memory", "6g")
             .config("spark.sql.shuffle.partitions", "16")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    # arm_common.py reaches the Python workers through PYTHONPATH (local mode);
    # sc.addPyFile would need Hadoop's winutils.exe on Windows.
    return spark
