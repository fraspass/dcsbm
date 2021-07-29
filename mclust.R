#' Function to perform clustering using mclust 
#' @author Francesco Sanna Passino

require(mclust)
mclust_fit = function(X){
  M = Mclust(X, modelNames='VVV')
  return(M$classification-1)
}