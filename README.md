使用spark mllib的DecisionTree，RandomForest尝试Titian数据集

version1.0
此版本先将Name,Ticket,Cabin舍弃
Survived 类标签
Pclass   数值型特征
Sex      类别型特征
Age      数值型特征
SibSp    数值型特征
Parch    数值型特征
Fare     数值型特征
Embarked 类别型特征

在Embarked这一列中发现有两个行记录为非法记录
Embarked为类别行特征，只有Q,S,C这三种
但是发现了B28第四种值,去除
之所以为B28,是因为这两条记录没有登录地点，B28为Canbin号
