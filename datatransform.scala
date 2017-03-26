import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

val rawData = sc.textFile("file:////home/hadoop/kaggle/kaggletitanic/train.csv").filter(!_.contains("PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked"))

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
