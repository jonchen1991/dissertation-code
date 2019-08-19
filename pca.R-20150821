#!/usr/bin/env Rscript

# SVM portion
desmatrix0 <- as.matrix(read.table("trainingset/HCdesmatrix-int0"))
desmatrix1 <- as.matrix(read.table("trainingset/HCdesmatrix-int1"))
desmatrix2 <- as.matrix(read.table("trainingset/HCdesmatrix-int2"))
desmatrix <- cbind(desmatrix0,desmatrix1,desmatrix2)
colnames(desmatrix) <- seq(1,ncol(desmatrix))
#response <- as.matrix(read.table("compound_class"))

# Find eigenvalues and vectors for PCA with covariance matrix
eigen_results <- eigen(cov(desmatrix))
eigenvalues <- eigen_results$values
eigenvalues.identified <-which(eigenvalues>1)
eigenvalues.filtered <- eigenvalues[eigenvalues.identified]
loadings.filtered <- abs(eigen_results$vectors[,eigenvalues.identified]) %*% eigenvalues.filtered

# Save most relevant SVM features
svm.pca.features <-which(loadings.filtered>1)
save(svm.pca.features, file="svm.pca.features")

# Clear workspace to reset everything
rm(list=ls())

# SVR portion
desmatrix0 <- as.matrix(read.table("activity/HCdesmatrix-int0"))
desmatrix1 <- as.matrix(read.table("activity/HCdesmatrix-int1"))
desmatrix2 <- as.matrix(read.table("activity/HCdesmatrix-int2"))
desmatrix <- cbind(desmatrix0,desmatrix1,desmatrix2)
colnames(desmatrix) <- seq(1,ncol(desmatrix))
#response <- log(as.matrix(read.table("activity.value")),base=10)

# Find eigenvalues and vectors for PCA with covariance matrix
eigen_results <- eigen(cov(desmatrix))
eigenvalues <- eigen_results$values
eigenvalues.identified <-which(eigenvalues>1)
eigenvalues.filtered <- eigenvalues[eigenvalues.identified]
loadings.filtered <- abs(eigen_results$vectors[,eigenvalues.identified]) %*% eigenvalues.filtered
# Save most relevant SVM features
svr.pca.features <-which(loadings.filtered>1)
save(svr.pca.features, file="svr.pca.features")