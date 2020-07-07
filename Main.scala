import myclustering.MyKmeans
import org.apache.spark.mllib.linalg.{Vectors, Vector => SparkVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


object Main {
  def main(args: Array[String]) {

    /*This creates a SparkSession, which will be used to operate on the DataFrames that we create.*/
    val spark = SparkSession.builder()
      .appName("languages-project")
      .master("local[*]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .getOrCreate()

    /* The SparkContext (usually denoted sc in code) is the entry point for low-level Spark APIs */
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR") // suppress warnings


    /** *********************************/


    // Load and parse the data
    val data: RDD[String] = sc.textFile("blobs.data")

    val parsedData: RDD[SparkVector] = data.map(row => Vectors.dense(row.split(",").map(cell => cell.toDouble)))
    //parsedData.foreach(println)

    // Cluster the data into two classes using KMeans
    val numClusters = 3
    val numIterations = 10

    // OUR KMEANS
    val kmeansModel = new MyKmeans(sc)
    println("\nMyKmeans Clustering...")
    kmeansModel.fit(parsedData, numClusters, numIterations, 0)
  }
}
