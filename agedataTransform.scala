import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

val rawData = sc.textFile("file:///root/kaggle/kaggletitanic/test.csv").filter(!_.contains("PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked"))

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
})

data = data.filter(_.size > 9).map(arr => {
  arr.slice(0,6):+arr(7):+arr(9)
})

val sexMap = Map(
  "male" -> "0",
  "female" -> "1"
)

val embarkedMap = Map(
  "S" -> "0",
  "Q" -> "1",
  "C" -> "2"
)
// delete the record which the element is empty
data = data.filter(r => {
  var flag = true
  for(str <- r if flag){
    if(str.equals("")){
      flag = false
    }
  }
  flag
})

val Array(trainData,cvData,testData) = data.map(r => {
  var arr = r
  arr(2) = sexMap(arr(2))
  arr(7) = embarkedMap(arr(7))
  val feature = arr.tail.map(_.toDouble)
  val label = arr.head.toDouble 
  LabeledPoint(label,Vectors.dense(feature))
}).randomSplit(Array(0.8,0.1,0.1))
trainData.cache()
cvData.cache()
testData.cache()

val model = DecisionTree.trainClassifier(trainData,2,Map(1 -> 2,6 -> 3),"gini",4,30)

def getMetrics(model:DecisionTreeModel,data:RDD[LabeledPoint]):MulticlassMetrics = {
  val predictAndLabel = data.map(example => {
    (model.predict(example.features),example.label)
  })
  new MulticlassMetrics(predictAndLabel)
}
val metrics = getMetrics(model,cvData)

val evaluations = for(impurity <- Array("gini","entropy");depth <- Array(4,30);bins <- Array(50,100,300)) yield {
  val model = DecisionTree.trainClassifier(trainData,2,Map(1 -> 2,6 -> 3),impurity,depth,bins)
  val metrics = getMetrics(model,cvData)
  ((impurity,depth,bins),metrics.precision)
}

val (impurity,depth,bins) = evaluations.sortBy(_._2).reverse.head._1

val forest = RandomForest.trainClassifier(trainData,2,Map(1 -> 2,6 ->3),20,"auto",impurity,depth,bins)
def getMetrics(model:RandomForestModel,data:RDD[LabeledPoint]):MulticlassMetrics = { 
  val predictAndLabel = data.map(example => {
    (model.predict(example.features),example.label)
  })  
  new MulticlassMetrics(predictAndLabel)
}

val evaluations = for(impurity <- Array("gini","entropy");depth <- Array(10,20,30);bins <- Array(50,100,200)) yield {
  val model = RandomForest.trainClassifier(trainData,2,Map(1 -> 2,6 -> 3),20,"auto",impurity,depth,bins)
  val metrics = getMetrics(model,cvData)
  ((impurity,depth,bins),metrics.precision)
}

