library(splatter)
# Set seed for reproducibility
set.seed(123) 

# Define parameters 
num_cells <- 8000
num_genes <- 1000 
de_prob <- 0.1
pop_prob <- c(0.2, 0.3, 0.1, 0.15, 0.25)
batch_cells = c(num_cells/2, num_cells/2) 
batch_facScale = 0.75

### with batch effect 
sim <- splatSimulate(
  batchCells = batch_cells, 
  nGenes = num_genes, 
  group.prob = pop_prob,
  de.prob = de_prob,
  batch.facScale = batch_facScale, 
  method = "groups"
)

# Get the counts and group info for storing
ref_cells <- which(sim$Batch == 'Batch1')
q_cells <- which(sim$Batch == 'Batch2') 

write.csv(assay(sim, "counts")[,ref_cells], "splatter_ref_counts_b0.75.csv")
write.csv(assay(sim, "counts")[,q_cells], "splatter_q_counts_b0.75.csv") 
write.csv(colData(sim)$Group[ref_cells], "splatter_ref_labels_b0.75.csv")
write.csv(colData(sim)$Group[q_cells], "splatter_q_labels_b0.75.csv")

### no batch effect 
# Create the simulation object
sim <- splatSimulate(
  batchCells = num_cells, 
  nGenes = num_genes, 
  group.prob = pop_prob,
  de.prob = de_prob,
  method = "groups"
)

# Get the counts and group info for storing
counts <- assay(sim, "counts")
group_labels <- colData(sim)$Group

write.csv(counts, "splatter_counts.csv") 
write.csv(group_labels, "splatter_labels.csv") 
