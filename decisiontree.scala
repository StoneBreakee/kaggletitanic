import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

val rawData = sc.textFile("file:///root/Downloads/covtype.data")

val data = rawData.map(line => {
  val values = line.split(",").map(_.toDouble)
  val featurevector = Vectors.dense(values.init)
  val label = values.last - 1
  LabeledPoint(label,featurevector)
})

val Array(trainData,cvData,testData) = data.randomSplit(Array(0.8,0.1,0.1))
trainData.cache()
cvData.cache()
testData.cache()

import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

val model = DecisionTree.trainClassifier(trainData,7,Map[Int,Int](),"gini",4,100)

def getMetrics(model:DecisionTreeModel,data:RDD[LabeledPoint]):MulticlassMetrics = {
  val predictionAndLabels = data.map(example => {
    (model.predict(example.features),example.label)
  })
  new MulticlassMetrics(predictionAndLabels)
}

val metrics = getMetrics(model,cvData)
println(metrics.precision)

(0 until 7).map(cat => (metrics.precision(cat),metrics.recall(cat))).foreach(println)

def classProbabilities(data:RDD[LabeledPoint]):Array[Double] = {
  val countByCategory = data.map(_.label).countByValue()
  val counts = countByCategory.toArray.sortBy(_._1).map(_._2)
  counts.map(_.toDouble / counts.sum)
}

val trainPriorProbabilities = classProbabilities(trainData)
val cvPriorProbabilities = classProbabilities(cvData)
val sum = trainPriorProbabilities.zip(cvPriorProbabilities).map{case(trainProb,cvProb) => trainProb * cvProb}.sum
println(sum)

val evaluations = for(impurity <- Array("gini","entropy");
                        depth <- Array(1,20);
                          bins <- Array(10,300)) 
                  yield {
                    val model = DecisionTree.trainClassifier(trainData,7,Map[Int,Int](),impurity,depth,bins)
                    val metrics = getMetrics(model,cvData)
                    ((impurity,depth,bins),metrics.precision)
                  }

evaluations.sortBy(_._2).reverse.foreach(println)

//使用 类别型特征
val data = rawData.map(line => {
  val values = line.split(",").map(_.toDouble)
  val wilderness = values.slice(10,14).indexOf(1.0).toDouble
  val soil = values.slice(14,54).indexOf(1.0).toDouble
  val featurevector = Vectors.dense(values.slice(0,10):+wilderness:+soil)
  val label = values.last - 1
  LabeledPoint(label,featurevector)
})

val Array(trainData,cvData,testData) = data.randomSplit(Array(0.8,0.1,0.1))
trainData.cache()
cvData.cache()
testData.cache()

val evaluations = for(impurity <- Array("gini","entropy");depth <- Array(10,20,30);bins <- Array(40,300)) yield{
                    val model = DecisionTree.trainClassifier(trainData,7,Map(10 -> 4,11 -> 40),impurity,depth,bins)
                    val trainAccuracy = getMetrics(model,trainData).precision
                    val cvAccuracy = getMetrics(model,cvData).precision
                    ((impurity,depth,bins),(trainAccuracy,cvAccuracy))
                  }

evaluations.sortBy(_._2._2).reverse.foreach(println)

//使用随机森林
val forest = RandomForest.trainClassifier(trainData,7,Map(10 -> 4,11 -> 40),20,"auto","entropy",30,300)

def getMetrics(model:RandomForestModel,data:RDD[LabeledPoint]):MulticlassMetrics = {
  val predictionAndLabels = data.map(example => {
    (model.predict(example.features),example.label)
  })
  new MulticlassMetrics(predictionAndLabels)
}

val metrics = getMetrics(forest,cvData)
