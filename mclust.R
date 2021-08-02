#' Function to perform clustering using mclust 
#' @author Francesco Sanna Passino

require(mclust)
mclust_fit = function(X){
  if(dim(X)[2] == 1){
    X = as.vector(X)
  }
  M = Mclust(X)
  return(M$classification-1)
}