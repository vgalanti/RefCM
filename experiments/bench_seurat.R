library(Seurat)

q <- read.csv("q.csv", row.names = 1, header = TRUE)
q_labels <- scan("q_labels.csv", what = character(), sep = ",")

ref <- read.csv("ref.csv", row.names = 1, header = TRUE)
ref_labels <- scan("ref_labels.csv", what = character(), sep = ",")

# create Seurat objects
q <- CreateSeuratObject(counts = t(q))
ref <- CreateSeuratObject(counts = t(ref))

# Data normalization
q <- NormalizeData(q, verbose = FALSE)
ref <- NormalizeData(ref, verbose = FALSE)

# Feature Selection
q <- FindVariableFeatures(q, selection.method = "vst", nfeatures = 2000)
ref <- FindVariableFeatures(ref, selection.method = "vst", nfeatures = 2000)

# Run FindTransferAnchors on filtered datasets
anchors <- FindTransferAnchors(reference = ref, query = q, dims = 1:30)
preds <- TransferData(
    anchorset = anchors, refdata = ref_labels,
    dims = 1:30
)

q <- AddMetaData(object = q, metadata = preds)

# Extract predicted cell types
preds <- q$predicted.id
preds <- data.frame(Predicted_Cell_Type = preds)
write.csv(preds, "preds.csv", row.names = FALSE)
