


sqlType = ["INSERT", "UPDATE", "DELETE", "SELECT"]
sqlCharacterType = ["ORDER", "GROUP", "INDEX", "SORT"]   #补充
workloadType = ["heavy read", "heavy write", "large connection", "heavy workload"]
otherType = ["VERSION"]

sqlType_converse = ["no " + item for item in sqlType]
sqlCharacterType_converse = ["no " + item for item in sqlType]
workloadType_converse = ["not "+item for item in workloadType]
otherType = ["VERSION", "GROUP BY"]


conditionType = [sqlType, sqlCharacterType, workloadType, otherType]
conditionType_converse =[sqlCharacterType_converse, sqlCharacterType_converse, workloadType_converse]

conditionType.extend(conditionType_converse)

#print(conditionType)
