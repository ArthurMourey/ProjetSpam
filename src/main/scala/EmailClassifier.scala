import java.io.File

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ListBuffer

object EmailClassifier extends App {
    val conf = new SparkConf().setAppName("ProjetSpam").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val tf = new HashingTF(numFeatures = 1000)

    val spam = lireTousLesEmail("/donnees/modele/spam", estSpam = true)
    val normal = lireTousLesEmail(s"/donnees/modele/nonSpam", estSpam = false)

    val trainingData = spam.union(normal)
    trainingData.cache()

    val model = new LogisticRegressionWithLBFGS().run(trainingData)

    //Test on a positive example (spam) and a negative one (normal).
    val posTest = tf.transform("insurance plan which change your life ...".split(" "))
    val negTest = tf.transform("hi sorry yaar i forget tell you i cant come today".split(" "))

    println("Prediction for positive test example: " + model.predict(posTest))
    println("Prediction for negative test example: " + model.predict(negTest))

    def lireTousLesEmail(cheminDossier: String, estSpam: Boolean): RDD[LabeledPoint] = {
        var listeLabeledPoint: ListBuffer[LabeledPoint] = ListBuffer()
        val dossier = new File(cheminDossier)

        for(fichier <- dossier.listFiles() if fichier.getName endsWith ".txt"){
            var spam = -1
            if(estSpam){
                spam = 1;
            }
            else {
                spam = 0;
            }

            val spamFeatures = sc.textFile(fichier.getAbsolutePath).map(email => tf.transform(email.split(" ")))
            val lPoint = new LabeledPoint(spam, tf.transform(spamFeatures.collect()))
            listeLabeledPoint += lPoint
        }
        val toReturn = sc.parallelize(listeLabeledPoint);
        toReturn
    }
}

