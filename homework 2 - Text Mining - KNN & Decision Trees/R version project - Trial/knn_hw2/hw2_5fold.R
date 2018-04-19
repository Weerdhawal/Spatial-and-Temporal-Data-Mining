accuracy.vec <- vector()
range.vec  <- c(1,3766,7531,11296,15061)
range.vec1 <- c(3765,7530,11295,15060,18828)

for(i in 1:5 ){
  #dataset train test
  test.data <- (range.vec[i]:range.vec1[i])
  train.data <-(1:18828)[-test.data]
  
  tdm.categ <- tdm.stack[,"mainCATEGORIES"]
  tdm.stack.nl <- tdm.stack[,!colnames(tdm.stack) %in% "mainCATEGORIES"]
  
  knn.pred<- knn(tdm.stack.nl[train.data, ],tdm.stack.nl[test.data,],tdm.categ[train.data])
  #accuracy
  conf.matrix <- table("Prediction"= knn.pred,Actual =tdm.categ[test.data])
  
  accuracy<- sum(diag(conf.matrix))/length(test.data)*100
  accuracy.vec<- c(accuracy.vec,accuracy)
  
  rm(test.data)
  rm(train.data)
  rm(tdm.categ)
  rm(knn.pred)
  rm(conf.matrix)
  rm(accuracy)
  

   }
mean(accuracy.vec)
