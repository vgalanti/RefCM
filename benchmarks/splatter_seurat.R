library(Seurat)
# for data without batch effect 
# reference.data <- read.csv("~/research/RefCM/data/splatter_ref_counts.csv", header=FALSE)
# query.data <- read.csv("~/research/RefCM/data/splatter_q_counts.csv", header=FALSE)
# reference <- CreateSeuratObject(counts = t(reference.data))
# query <- CreateSeuratObject(counts = t(query.data)) 
# train_labels <- unlist(read.csv("~/research/RefCM/data/splatter_ref_labels_b0.25.csv", header=FALSE))
# test_labels <- unlist(read.csv("~/research/RefCM/data/splatter_q_labels_b0.25.csv", header=FALSE))

reference.data <- read.csv("~/research/RefCM/data/splatter_ref_counts_b1.csv", row.names=1)
query.data <- read.csv("~/research/RefCM/data/splatter_q_counts_b1.csv", row.names=1) 

train_labels <- unlist(read.csv("~/research/RefCM/data/splatter_ref_labels_b1.csv")[,2])
test_labels <- unlist(read.csv("~/research/RefCM/data/splatter_q_labels_b1.csv")[,2])

reference <- CreateSeuratObject(counts = reference.data)
query <- CreateSeuratObject(counts = query.data)

# Data normalization
reference <- NormalizeData(reference, verbose = FALSE)
query <- NormalizeData(query, verbose = FALSE)

# Feature Selection
reference <- FindVariableFeatures(reference, selection.method = "vst", nfeatures = 200)
query <- FindVariableFeatures(query, selection.method = "vst", nfeatures = 200)

# Scale Data
reference <- ScaleData(reference)
query <- ScaleData(query)

# Run FindTransferAnchors on filtered datasets
anchors <- FindTransferAnchors(reference = reference, query = query, dims = 1:30)
predictions <- TransferData(anchorset = anchors, refdata = train_labels,
                            dims = 1:30)
query <- AddMetaData(object = query, metadata = predictions)

print(sum(query$predicted.id == test_labels))
