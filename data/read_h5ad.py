import anndata as ad

def read_h5ad(file_path):
    adata = ad.read_h5ad(file_path)
    print(adata)
    #print(adata.obs)  # Observation data
    #print(adata.var)  # Variable data
    print(adata.X)    # The main data matrix

if __name__ == "__main__":
    file_path = input("Enter the path of your h5ad file: ")
    read_h5ad(file_path)
