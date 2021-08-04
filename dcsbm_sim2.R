#' @title   Reproduces data for Figure 2a
#' @author  Francesco Sanna Passino

## Simulation DCSBM
set.seed(111)
n = 1000
A = matrix(0,nrow=n,ncol=n)
B = matrix(c(0.1,0.05,0.05,0.15),byrow=TRUE,nrow=2,ncol=2)
set.seed(1771)
z = c(rep(1,500),rep(2,500))
rho = rbeta(n=n,shape1=2,shape2=1)
for(i in 1:(n-1)){
  for(j in (i+1):n){
    A[i,j] = rbinom(size=1,n=1,prob=rho[i] * rho[j] * B[z[i],z[j]])
    A[j,i] = A[i,j] 
  }
}

## 2-dimensional ASE
s = eigen(A)
X = s$vectors[,1:2] %*% diag(sqrt(s$values[1:2]))
X = X[rowSums(A) > 0,]
z = z[rowSums(A) > 0]

## Plot of the 2-dimensional ASE
par(mar=c(4.5,4.5,1,1))
plot(X[,1],X[,2], pch=(15:18)[z], col=z+1, xlab=expression(X[1]), ylab=expression(X[2]))

## Calculate spherical coordinates
cart_to_pol = function(x){
  pol_coor = matrix(nrow=nrow(x),ncol=ncol(x)-1)
  for(i in 1:nrow(x)){
    q = acos(x[i,2] / norm(x[i,1:2],type='2'))
    if(x[i,1] >= 0){
      pol_coor[i,1] = q
    } else {
      pol_coor[i,1] = 2*pi - q
    }
    if(ncol(x) > 2){
      for(j in 2:(ncol(x)-1)){
        pol_coor[i,j] = 2 * acos(x[i,j+1] / norm(x[i,1:(j+1)],type='2'))
      }
    }
  }
  return(pol_coor)
}

## Spherical coordinates
theta = cart_to_pol(X)
hist(theta[,1], breaks=50, c='SpringGreen2', main='', xlab=expression(Theta[1]))
hist(theta[z==1,1], breaks=50, freq=FALSE, c='Coral2', main='', xlab=expression(Theta[1]))
hist(theta[z==2,1], breaks=30, freq=FALSE, add=TRUE, c='SkyBlue2')

## Normalise
X_tilde = t(scale(t(X),center=FALSE,scale=apply(X,MARGIN=1,FUN=norm,type='2')))
plot(X_tilde[,1], X_tilde[,2], col=z+1, xlab=expression(tilde(X)[1]), ylab=expression(tilde(X)[2]), pch=c(16,17)[z])
