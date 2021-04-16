install.packages("R.utils")
library(R.utils)
library("Biostrings")
library("magrittr")
library("tidyr")
library("ggplot2")
library(umap)
library("dplyr")
library("umap")
install.packages("devtools")
devtools::install_github("ropenscilabs/umapr")
library(tidyverse)

# fasta = readDNAStringSet("allFiles.sorted.minimap2.haploid.flt.merged.noINDELs.allCoveredSamples.fasta")
samples_metadata <- read.table(file = 'samples_metadata.tsv', sep = '\t', header = TRUE)

#alt_nuc <- read.table(file = 'alt_nuc.tsv', sep = '\t', header = TRUE)

# gunzipped <- gunzip("dataset1_sm_uc3.txt.gz")

# db_1 <- read.table(file = 'dataset1_sm_uc3.txt', sep = '\t', header = TRUE)

samples_matrix <- read.table(file = 'samples_matrix.tsv', sep = '\t', header = TRUE)

unique(samples_metadata[c("seq_center")])
# 30 seq_centers
unique(samples_metadata[c("seq_instrument")])
# 6 instrumentations

unique(samples_metadata[c("seq_assay")])
# 4 instrumentations

unique(samples_metadata[c("country_exposure")])
# 14 instrumentations

samples_metadata <- samples_metadata[-c(11776:12423),]
samples_matrix$seq_center <- samples_metadata$seq_center


res = load("/Users/alimehdi/Downloads/spleen_expression.rda")
head(series)
unique(series)
unique(expression)

matrix_transposed <- t(as.data.frame(samples_matrix))
dim(matrix_transposed)

matrix_numeric <- data.matrix(matrix_transposed, rownames.force = 0)
mode(matrix_numeric) = "numeric"

# convert df to numerical:
matrix_transposed_numeric <- lapply(matrix_transposed,as.numeric)

# matrix_transposed_m <- matrix_transposed %>% mutate_all(~replace(., is.na(.), 0))
matrix_transposed_m1 <- data.matrix(matrix_transposed, rownames.force = NA) 
#matrix_transpose_m2 <- sapply(matrix_transposed_m1, as.numeric)

samples_metadata$seq_center_numeric <- as.numeric(factor(samples_metadata$seq_center, levels = unique(samples_metadata$seq_center)))
seq_center_vector <- as.vector(samples_metadata$seq_center_numeric)

mode(matrix_transposed_m1)
class(matrix_transposed_m1) <- "numeric"
correctedExpression <- ComBat(dat=samples_matrix, batch=seq_center_vector, par.prior=TRUE, prior.plots=FALSE)


limma <- removeBatchEffect(matrix_numeric,batch=seq_center_vector)
dim(limma)
limma_df <- as.data.frame(limma)
limma_df_ <- t(limma_df)
head(limma)
head(limma_df)
write.csv(limma,'/Users/alimehdi/Downloads/limma_not_t')

cc = cor(matrix_numeric)
dend <- as.dendrogram(hclust(as.dist(1-cc)))

