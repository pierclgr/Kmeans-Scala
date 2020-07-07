package myclustering

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vectors, Vector => SparkVector}
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import scala.util.Random

/**
 * Class representing the KMeans clustering
 *
 * @param sc Spark Context
 */
class MyKmeans(val sc: SparkContext) extends Serializable {

  /**
   * Function that starts the Kmeans calculation procedure
   *
   * @param data          dataset to calculate clusters for
   * @param nClusters     number of clusters to calculate
   * @param maxIterations maximum
   * @param epsilon       maximum shift between old and new centroids
   */
  def fit(data: RDD[SparkVector], nClusters: Int, maxIterations: Int, epsilon: Double): Unit = {

    /**
     * Function to calculate the euclidean distance between two points
     *
     * @param point1     point 1
     * @param point2     point 2
     * @param currentDim dimension on which to calculate the distance
     * @return distance between point 1 and 2 over the axis currentDim
     */
    def calculateDistance(point1: SparkVector, point2: SparkVector, currentDim: Int): Double = {
      // Calculate the squared distance over the axis "currentDim" between the points
      Math.pow(point1(currentDim) - point2(currentDim), 2)
    }

    /**
     * Tail recursive step function of the Kmeans centroid calculation
     *
     * @param oldCentroids old centroids
     * @param dims         list containing the dimensions of the dataset
     * @param iteration    current iteration of the algorithm
     * @return newCentroids calculated with Kmeans iteration
     */
    @tailrec
    def stepFit(oldCentroids: RDD[(Int, SparkVector)], dims: List[Int], iteration: Int): RDD[(Int, SparkVector)] = {
      // Print current centroids
      oldCentroids.values.foreach(print)
      println()

      // Compute the distances bewteen each point and the centroids
      val distances: RDD[(SparkVector, (Int, Double))] = (data cartesian oldCentroids)
        .map { case (point: SparkVector, (cluster: Int, centroid: SparkVector)) =>
          point -> (cluster -> Math.sqrt(dims
            .map(calculateDistance(point, centroid, _)).sum
          ))
        }

      // Assign to each cluster all the nearest points
      val assignments: RDD[(Int, Iterable[SparkVector])] =
        distances
          .groupByKey
          .map { case (point: SparkVector, distance: Iterable[(Int, Double)]) =>
            (point, distance
              .minBy { case (_: Int, distance: Double) =>
                distance
              })
          }
          .map { case (point: SparkVector, (cluster: Int, _: Double)) =>
            (cluster, point)
          }
          .groupByKey

      // Compute the new centroids for each cluster
      val newCentroids: RDD[(Int, SparkVector)] = assignments
        .map { case (cluster: Int, points: Iterable[SparkVector]) =>
          (cluster, Vectors.dense(points
            .reduce((point1, point2) => Vectors.dense(dims
              .map(dim => point1(dim) + point2(dim)).toArray)).toArray
            .map(dim => dim / points.size)))
        }
        .sortByKey()

      // Compute the shift of the centroids
      val moves: RDD[(Int, Double)] =
        newCentroids.join(oldCentroids)
          .map { case (x: Int, (oldCentroid: SparkVector, newCentroid: SparkVector)) =>
            x -> Math.sqrt(dims.map(calculateDistance(newCentroid, oldCentroid, _)).sum)
          }

      // Sum the distances to calculate the new shift
      val shift: Double = moves.values.sum
      println("Delta: " + f"$shift%.3f")

      // If the shift is less than epsilon or the iteration reached the maximum number
      if (shift < epsilon || iteration >= maxIterations) newCentroids // Then terminate
      else stepFit(newCentroids, dims, iteration + 1) // Otherwise repeat
    }

    // Create list containing the dataset dimensions
    val dims: List[Int] = List.range(0, data.first.size)

    // Create a list containing the clusters
    val clusters: List[Int] = List.range(1, nClusters + 1)

    // Get Min and Max values for each dimension
    val minMaxValues: RDD[(Double, Double)] = sc.parallelize(dims.map(dim =>
      (data.min()(Ordering[Double].on(vect => vect(dim))).apply(dim),
        data.max()(Ordering[Double].on(vect => vect(dim))).apply(dim)))
    )

    // Initialize random centroids
    val randomCentroids: RDD[(Int, SparkVector)] = sc.parallelize(clusters.map(
      cluster => (cluster, Vectors.dense(minMaxValues
        .map { case (min: Double, max: Double) =>
          min + (max - min) * Random.nextDouble()
        }
        .collect))))
      .sortBy { case (cluster: Int, _: SparkVector) =>
        cluster
      }

    println("log:")
    // Recursive call
    val centroids = stepFit(randomCentroids, dims, 1)

    // Print the centroids and clusters calculated
    println("centroids found:")
    centroids.foreach(println)
  }
}