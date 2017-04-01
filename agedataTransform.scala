import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

val rawData = sc.textFile("file:///root/Downloads/train.csv").filter(!_.contains("PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked"))

var data = rawData.map(line => {
  // line = 1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
  // delete Name arrtibute
  val start = line.indexOf("\"")
  var i = 0
  for(index <- (start+1 until line.length)){
    if(line(index) == '"'){
        i = index
    }
  }
  val str = line.substring(0,start-1)+line.substring(i+1)
  str.split(",").tail
}).filter(_.size == 10).map(r => {
  var tmp = r
  tmp(2) = sexMap(tmp(2))
  tmp(9) = embarkedMap(tmp(9))
  tmp
})

data = data.filter(_(3) == "").map(r => r.slice(0,3)++:r.slice(4,6):+r(7):+r(9))

val Array(trainData,cvData,testData) = data.map(r => {
  val featurevector = r.tail.map(_.toDouble)
  val label = r.head.toDouble
  LabeledPoint(label,Vectors.dense(featurevector))
}).randomSplit(Array(0.8,0.1,0.1))
trainData.cache()
cvData.cache()
testData.cache()


def getMetrics(model:DecisionTreeModel,data:RDD[LabeledPoint]):MulticlassMetrics = {
  val predictAndLabel = data.map(example => {
    (model.predict(example.features),example.label)
  })
  new MulticlassMetrics(predictAndLabel)
}
val metrics = getMetrics(model,cvData)

val evaluations = for(impurity <- Array("gini","entropy");depth <- Array(4,30);bins <- Array(50,100,300)) yield {
  val model = DecisionTree.trainClassifier(trainData,2,Map(1 -> 2,5 -> 3),impurity,depth,bins)
  val metrics = getMetrics(model,cvData)
  ((impurity,depth,bins),metrics.precision)
}


val (impurity,depth,bins) = evaluations.sortBy(_._2).reverse.head._1
val model = DecisionTree.trainClassifier(trainData,2,Map(1 -> 2,5 ->3),impurity,depth,bins)

//对没有age字段的记录进行预测
var testRawData = sc.textFile("file:///root/Downloads/test.csv")

var data = testRawData.map(line => {
  // line = 1,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
  // delete Name arrtibute
  val start = line.indexOf("\"")
  var i = 0
  for(index <- (start+1 until line.length)){
    if(line(index) == '"'){
        i = index
    }
  }
  val str = line.substring(0,start-1)+line.substring(i+1)
  val id = str.split(",").head
  var arr = str.split(",").tail
  arr(1) = sexMap(arr(1))
  arr(8) = embarkedMap(arr(8))
  val tmp = arr.slice(0,5):+arr(6):+arr(8)
  (id -> tmp)
})

//data = data.filter(r => {
//  var flag = true
//  for(str <- r._2 if flag){
//    if(str.equals("")){
//      flag = false
//    }
//  }
//  flag
//})

// only remain the record which has no age
data = data.filter(r => r._2(2) == "").map(r => {
  (r._1,r._2.filter(_ != ""))
})

val withnoageresult = data.map(r => {
  (r._1 -> model.predict(Vectors.dense(r._2.map(_.toDouble))))
})
