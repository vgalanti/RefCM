library(Seurat)
MouseV1_MouseALM_HumanMTG <- read.csv("~/research/data/MouseV1_MouseALM_HumanMTG/MouseV1_MouseALM_HumanMTG.csv", 
                                      row.names=1,
                                      header=TRUE)
labels <- scan("~/research/data/MouseV1_MouseALM_HumanMTG/MouseV1_MouseALM_HumanMTG_Labels34.csv", what = character(), sep = ",")
train_labels <- labels[1:12552]
test_labels <- labels[20681:34735]

# MTG reference; ALM query
reference.data <- MouseV1_MouseALM_HumanMTG[1:12552,]
query.data <- MouseV1_MouseALM_HumanMTG[20681:34735,]
reference <- CreateSeuratObject(counts = t(reference.data))
query <- CreateSeuratObject(counts = t(query.data))

# Clean variables
rm(query.data) 
rm(reference.data) 
rm(MouseV1_MouseALM_HumanMTG)
rm(labels)

# Data normalization
reference <- NormalizeData(reference, verbose = FALSE)
query <- NormalizeData(query, verbose = FALSE)

# Feature Selection
reference <- FindVariableFeatures(reference, selection.method = "vst", nfeatures = 2000)
query <- FindVariableFeatures(query, selection.method = "vst", nfeatures = 2000)

# Run FindTransferAnchors on filtered datasets
allen_brain.anchors <- FindTransferAnchors(reference = reference, query = query, dims = 1:30)
predictions <- TransferData(anchorset = allen_brain.anchors, refdata = train_labels,
                            dims = 1:30)
query <- AddMetaData(object = query, metadata = predictions)
query$celltype <- test_labels
query$prediction.match <- query$predicted.id == query$celltype
table(query$prediction.match)

