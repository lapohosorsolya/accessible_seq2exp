# Preprocessing

**Contents**
- [Dataset source links](#dataset-source-links)
- [Downloads](#downloads)
- [Preprocessing scripts](#preprocessing-scripts)
- [Cell type code reference table](#cell-type-code-reference-table)

## Dataset source links

| multiome dataset | URL |
| --- | --- |
| PBMC | [link](https://www.10xgenomics.com/resources/datasets/pbmc-from-a-healthy-donor-no-cell-sorting-10-k-1-standard-2-0-0) |
| brain | [link](https://www.10xgenomics.com/resources/datasets/frozen-human-healthy-brain-tissue-3-k-1-standard-2-0-0) |
| jejunum | [link](https://www.10xgenomics.com/resources/datasets/human-jejunum-nuclei-isolated-with-chromium-nuclei-isolation-kit-saltyez-protocol-and-10x-complex-tissue-dp-ct-sorted-and-ct-unsorted-1-standard) |

## Downloads

To make training and testing sets from the 10x Genomics datasets in the table above, obtain the following files.

1. Go to a dataset link in the table and download the files named:
   - `filtered_feature_bc_matrix.h5`
   - `atac_fragments.tsv`

2. Download the corresponding cell type annotations CSV file from [Figshare](https://figshare.com/articles/dataset/Cell_type_annotations/26426392).

3. Download the promoter data:
   - `GRCh38.p14_promoter_locs_1kb-up_1kb-down.pkl` [link](https://figshare.com/account/projects/215935/articles/29333408?file=55428860)
   - `GRCh38.p14_promoter_seqs_1kb-up_1kb-down.npz` [link](https://figshare.com/account/projects/215935/articles/29333426?file=55428866)

## Preprocessing scripts

The following scripts should be run using the environment specified in this repository.

1. Run the ATAC fragment pooling script, specifying the numeric cell type code according to the [reference table](#cell-type-code-reference-table):

        python pool_atac_fragments_by_gene.py
            -a <path to cell_type_annotations.csv>
            -f <path to atac_fragments.tsv>
            -o <path to output directory>
            -c <numeric cell type code>
            -l <path to GRCh38.p14_promoter_locs_1kb-up_1kb-down.pkl>

2. Run the dataset prep script, making sure that the input directory specified here contains the 10x Genomics files and the cell type annotations CSV file.

        python make_train_test_data.py 
            -i <path to input directory>
            -c <numeric cell type code>
            -o <path to output directory>
            -a <path to atac_signals_smoothed_normalized pickle file generated in previous step>
            -l <path to GRCh38.p14_promoter_locs_1kb-up_1kb-down.pkl>
            -p <path to GRCh38.p14_promoter_seqs_1kb-up_1kb-down.npz>

The output of this workflow can directly be used for training and testing accessible_seq2exp models.

## Cell type code reference table

| cell type | source dataset | numeric code |
| --- | --- | --- |
| B cell | PBMC | 0 |
| CD14+ monocyte | PBMC | 1 |
| CD4 T cell | PBMC | 2 |
| CD8 T cell | PBMC | 3 |
| other | PBMC | 4 |
| astrocyte/microglia | brain | 5 |
| oligodendrocyte | brain | 6 |
| neuron | brain | 7 |
| OPC | brain | 8 |
| epithelial | jejunum | 9 |
| stromal (immune) | jejunum | 10 |
| stromal (other) | jejunum | 11 |
| ISC | jejunum | 12 |