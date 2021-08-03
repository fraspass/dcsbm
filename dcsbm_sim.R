#' @title   Reproduce Figure 2a
#' @author  Francesco Sanna Passino

## Simulated DCSBM
n = 1000
set.seed(1771)
B = lower.tri(matrix(1,ncol=4,nrow=4), diag=TRUE) * matrix(rbeta(n=4^2,shape=1,shape2=1),ncol=4)
B = B + t(B) * upper.tri(B)
z = c(rep(1,250),rep(2,250),rep(3,250),rep(4,250))
rho = rbeta(n=n,shape1=2,shape2=1)
A = matrix(0,nrow=n,ncol=n)
for(i in 1:(n-1)){
  for(j in (i+1):n){
    A[i,j] = rbinom(size=1,n=1,prob=rho[i] * rho[j] * B[z[i],z[j]])
    A[j,i] = A[i,j] 
  }
}

## ASE
s = eigen(A)
X = s$vectors[,1:5] %*% diag(sqrt(s$values[1:5]))
X = X[rowSums(A) > 0,]
z = z[rowSums(A) > 0]

## Plot 
par(mar=c(4,4,1,1))
plot(X[,1],X[,2], pch=(15:18)[z], col=z+1, xlab=expression(X[1]), ylab=expression(X[2]))
