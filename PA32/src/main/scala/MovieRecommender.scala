import java.io._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating

object MovieRecommender {
  def main(arguments: Array[String]) {
  
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Movie_Recommender")
    val sc = new SparkContext(conf)
    var Rating_Path = arguments(0)
    var Test_Path = arguments(1)


    //set initial time
    val t2 = System.nanoTime

    //load in entire rating file into an RDD (strings)
    val EntireSet: RDD[String] = sc.textFile(Rating_Path)
    val header = EntireSet.first()
    val Data: RDD[((String, String), Float)] = EntireSet.filter(row => row!= header)
      .map(line=>line.split(","))
      .map(x=>((x(0),x(1)),x(2).toFloat))

    //load in entire test file into an RDD (strings)
    val TestSet: RDD[String] = sc.textFile(Test_Path)
    val headerA = TestSet.first()
    val Test = TestSet.filter(row => row!= headerA)
      .map(line=>line.split(","))
      .map(x=>((x(0),x(1)),1))
    val testwithRating = Test.leftOuterJoin(Data).map(x=>Rating(x._1._1.toInt, x._1._2.toInt, x._2._2.get.toDouble))

    // create the training dataset
    val Train = Data.subtractByKey(Test)
    val ratings: RDD[Rating] = Train.map(x=> Rating(x._1._1.toInt, x._1._2.toInt, x._2.toDouble))

    // build recomendation model using ALS collaborative filter
    val rank = 8 
    val numIterations = 20 
    val model = ALS.train(ratings, rank, numIterations, 0.1) 


    // eval on rating data
    val user_movies = ratings.map { case Rating(user, movie, rate) =>
      (user, movie)
    }

    //make prediction based off training set
    //predictions is a list of users, the movie and the precicted rating
    val predictions =
      model.predict(user_movies).map { case Rating(user, movie, rate) =>
        ((user, movie), rate)
      }

    // (actual rating, predicted rating)
    //contains a list of the actual rating and predicted rating to be used in comparison later
    val ratings_and_predictions: RDD[((Int, Int), (Double, Double))] = ratings.map { case Rating(user, movie, rate) =>
      ((user, movie), rate)
    }.join(predictions)

    // MSE of the training set
    val MSE = ratings_and_predictions.map { case ((user, movie), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()

    // testing set
    val usersmovies_withRating = testwithRating.map { case Rating(user, movie, rate) =>
      (user, movie)
    }

    val prediction_test: RDD[((Int, Int), Double)] =
      model.predict(usersmovies_withRating).map { case Rating(user, movie, rate) =>
        ((user, movie), rate)
      }

    val ratings_and_predictions_test: RDD[((Int, Int), (Double, Option[Double]))] = testwithRating.map { case Rating(user, movie, rate) =>
      ((user, movie), rate)
    }.leftOuterJoin(prediction_test)

    val results_cleanup: RDD[((Int, Int), (Double, Double))] = ratings_and_predictions_test.map(x=> {
      if(x._2._2 isEmpty) {
        ((x._1),(x._2._1, 2.5))
      } else if (x._2._2.get < 0) {
        ((x._1), (x._2._1, 0.toDouble))
      } else if (x._2._2.get > 5){
        ((x._1), (x._2._1, 5.toDouble))
      } else ((x._1), (x._2._1, x._2._2.get))
    })


    val RMSE = results_cleanup.map { case ((user, movie), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()

    println("RSME = " + math.sqrt(RMSE))

    // print the time
    println("The total execution time taken is : %1.4f".format((System.nanoTime - t2)/1000000000.0) + " sec." )

    val final_result = results_cleanup.collect().sortBy(_._1)
      .map(x=> x._1._1.toString +  "," + x._1._2.toString + "," + x._2._2.toString)
    val output_path = "result.txt"
    val file = new FileWriter(output_path)
    file.write("UserId,MovieId,Pred_rating" + "\n")
    final_result.map(line => file.write(line + "\n"))
    file.close()
  }
}
