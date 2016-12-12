
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.rdd.RDD

object EmailClassifier {
    def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("email-spam")
        if(args(0).equals("local")) {
            // Création du context spark
            val sc = new SparkContext(conf)

            // On récupère les données pour créer le modèle
            // Crée un RDD avec un mail par ligne
            // Nb de partitions = nb de mails qu'on a mit dans les dossiers
            val valide = sc.wholeTextFiles("donneesModele/nonSpam/*")
            val spam = sc.wholeTextFiles("donneesModele/spam/*")

            val tf = new HashingTF(10000)

            // Retourne un autre RDD en transformant spam en vecteur de fréquence qui permet d'avoir la fréquence d'un mot
            val caracteristiquesSpam = spam.map(email => tf.transform(email._2.split(" ")))
            val caracteristiquesNonSpam = valide.map(email => tf.transform(email._2.split(" ")))


            // Retourne un autre RDD en utilisant les vecteurs LabeledPoints (1 pour spam, 0 pour non spam)
            // On fait l'union des deux RDD contenant les spams et les non spams
            val donnesModele = caracteristiquesSpam.map(carac => LabeledPoint(1, carac))
              .union(caracteristiquesNonSpam.map(carac => LabeledPoint(0, carac)))
            // On cache les données pour que l'exécution soit plus rapide
            donnesModele.cache()

            // On crée le modele
            val modele = new LogisticRegressionWithLBFGS().run(donnesModele)

            // On obtient les mails qui vont être appliqués au model
            val mails = getAllMails(sc) // mail._1 = nom du mail, mail._2 contenu du mail
            val nbMails = mails.count()

            // On parcourt les mails et on regarde s'ils sont spam ou non spam
            // Si le modèle renvoie 1 c'est un spam, sinon 0
            val res = mails.map(mail => {
                val result = modele.predict(tf.transform(mail._2.split(" ")))
                if (result == 1) {
                    println("Spam : " + mail._1)
                }
                if(result == 0){
                    println("valide : " + mail._1)
                }
                result
            }).reduce((val1, val2) => val1 + val2)
            Thread.sleep(5000);
            printResults(spam, valide, nbMails, res)
        }
        else{ // on est sur le cluster, pas besoin de spark context
            val sc = new SparkContext(conf)
            val valide = sc.wholeTextFiles("hdfs:/user/marinthe/nonSpam/*", 4)
            val spam = sc.wholeTextFiles("hdfs:/user/marinthe/spam/*", 4)

            val tf = new HashingTF(numFeatures = 10000)
            val caracteristiquesSpam = spam.map(email => tf.transform(email._2.split(" ")))
            val caracteristiquesNonSpam = valide.map(email => tf.transform(email._2.split(" ")))
            val positiveExamples = caracteristiquesSpam.map(features => LabeledPoint(1, features))
            val negativeExamples = caracteristiquesNonSpam.map(features => LabeledPoint(0, features))
            val trainingData = positiveExamples.union(negativeExamples)
            trainingData.cache()
            val model = new LogisticRegressionWithLBFGS().run(trainingData)

            var nbSpams = 0d
            var nbMails = 0

            for(i<- 1 to 11){ // Sur le cluster les fichiers sont dans 11 sours-repertoires
                val mails = getAllMailsCluster(sc, i)
                val res = mails.map(mail => {
                    val result = new Tuple2[Double, Int](
                        model.predict(tf.transform(mail._2.split(" "))), 1
                    )

                    if (result._1 == 1) {
                        println("Spam :" + mail._1)
                    }
                    if(result._1 == 0){
                        println("valide : " + mail._1)
                    }
                    result
                }).reduce((val1, val2) =>  (val1._1 + val2._1, val1._2 + val2._2 ))
                nbSpams += res._1
                nbMails += res._2
            }
            //val nbMails = mails.count()


            Thread.sleep(5000);
            printResults(spam, valide, nbMails, nbSpams)
        }
    }

    def getAllMails(sc: SparkContext): RDD[(String, String)] = {
        sc.wholeTextFiles("./maildir/*/*/*")
    }

    def getAllMailsCluster(sc: SparkContext, i: Int): RDD[(String, String)] ={
        sc.wholeTextFiles("hdfs:/data/maildir"+i+"/*")
    }

    def printResults(spam : RDD[(String, String)], nonSpam : RDD[(String, String)], nbMails : Double, res : Double) = {
        println("Nb de fichiers évalués : "+nbMails)
        println("Nb de spams : "+res)
        println("Pourcentage spams : "+(res/nbMails*100))
    }
}

