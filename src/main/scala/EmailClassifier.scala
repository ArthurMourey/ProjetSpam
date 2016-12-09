import java.io.File

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ListBuffer

object EmailClassifier extends App {
    val conf = new SparkConf().setAppName("email-spam").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spam = sc.textFile("./donneesModele/nonSpam/*", 4)
    val normal = sc.textFile("./donneesModele/spam/*", 4)
    val tf = new HashingTF(numFeatures = 10000)
    val spamFeatures = spam.map(email => tf.transform(email.split(" ")))
    val normalFeatures = normal.map(email => tf.transform(email.split(" ")))
    val positiveExamples = spamFeatures.map(features => LabeledPoint(1, features))
    val negativeExamples = normalFeatures.map(features => LabeledPoint(0, features))
    val trainingData = positiveExamples.union(negativeExamples)
    trainingData.cache()
    val model = new LogisticRegressionWithLBFGS().run(trainingData)

    val mails = sc.wholeTextFiles("./maildir/*/*/*")

   /* //Test on a positive example (spam) and a negative one (normal).
    val posTest = tf.transform("remembd that sharks ass with the  billy club?".split(" "))
    val negTest = tf.transform("hi sorry yaar i forget tell you i cant come today".split(" "))*/

    val res = mails.map(email => {
        val t = model.predict(tf.transform(email._2.split(" ")))
        if(t ==1){
            println("SPAM :"+email._1)
        }else{
            println("NON SPAM : "+email._1)
        }
        t
    }).reduce((val1, val2) => val1 + val2)
    println("Nb de spams : "+res)
    println("Pourcentage spams : "+(res/mails.count()*100))
   /* println("Prediction for positive test example: " + model.predict(posTest))
    println("Prediction for negative test example: " + model.predict(negTest))*/

    /*val conf = new SparkConf().setAppName("ProjetSpam").setMaster("local[*]")
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

        println(dossier)
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
    }*/
}

