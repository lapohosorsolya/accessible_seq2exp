import os, sys, getopt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import pickle

'''
Required Inputs
---------------
    -a (full path)
        path to cell type annotation file
    -f (full path)
        path to atac fragments file
    -o (output directory)
        the directory where the output should be written (must exist)
    -c (number)
        cell type code; { 0: 'B cell', 1: 'CD14+ monocyte', 2: 'CD4 T cell', 3: 'CD8 T cell', 4: 'other', 5: 'astrocyte/microglia', 6: 'oligodendrocyte', 7: 'neuron', 8: 'OPC', 9: 'epithelial', 10: 'stromal (immune)', 11: 'stromal (other)', 12: 'ISC' }
    -l (full path)
        path to promoter locations pickle file (GRCh38.p14_promoter_locs_1kb-up_1kb-down.pkl)
'''


def main(argv):
    try:
        opts, _ = getopt.getopt(argv, 'a:f:o:c:l:')
    except getopt.GetoptError:
        print('\n::: Error: cannot parse command line inputs')
        sys.exit(2) 
    for opt, arg in opts:
        if opt == '-a':
            global cell_type_annotation_file
            cell_type_annotation_file = arg
        elif opt == '-f':
            global atac_fragments_file
            atac_fragments_file = arg
        elif opt == '-o':
            global output_dir
            output_dir = arg
        elif opt == '-c':
            global cell_type_code
            cell_type_code = int(arg)
        elif opt == '-l':
            global locs_file
            locs_file = arg


if __name__ == "__main__":

    main(sys.argv[1:])

    print('Reading input files...')

    # generate list of indices for cells
    cell_type_dict = { 0: 'B cell', 1: 'CD14+ monocyte', 2: 'CD4 T cell', 3: 'CD8 T cell', 4: 'other', 5: 'astrocyte/microglia', 6: 'oligodendrocyte', 7: 'neuron', 8: 'OPC', 9: 'epithelial', 10: 'stromal (immune)', 11: 'stromal (other)', 12: 'ISC' }

    cell_type_annotations = pd.read_csv(cell_type_annotation_file, index_col = 0)
    try:
        selected_cells = cell_type_annotations[cell_type_annotations.cell_type == cell_type_dict[cell_type_code]].index.to_list()
        n_cells = len(selected_cells)
        if n_cells == 0:
            print('Cell type not found in data.')
            sys.exit(2)
    except:
        print('Invalid cell type code!')
        sys.exit(2)
    
    print('Found {} cell barcodes annotated with cell type code {} ({})'.format(n_cells, cell_type_code, cell_type_dict[cell_type_code]))

    # subset fragments to cell type of interest
    
    with open(atac_fragments_file, 'r') as f:
        lines = f.readlines()

    with open(locs_file, 'rb') as f:
        promoter_dict = pickle.load(f)

    # make a subset of the data for only selected cells
    fragments = {}
    n_lines = len(lines)
    for line in tqdm(lines, total = n_lines):
        tokens = line.split('\t')
        if len(tokens) == 5:
            chrom, start, end, barcode, count = tokens
            if barcode in selected_cells:
                # if the chromosome is not in the dict yet, add it
                if chrom not in fragments.keys():
                    fragments[chrom] = []
                # add the fragment info
                c = count.rstrip()
                fragments[chrom].append([int(start), int(end), int(c)])

    cell_type_fragments_file = os.path.join(output_dir, 'atac_fragments_{}.pkl'.format(cell_type_code))
    with open(cell_type_fragments_file, 'wb') as f:
        pickle.dump(fragments, f)
    
    del lines

    print('Subset of ATAC fragments saved.')

    # map fragments to promoters
    genes_by_chr = {}
    for gene in promoter_dict.keys():
        chrom, start, end, strand = promoter_dict[gene]
        if chrom not in genes_by_chr.keys():
            genes_by_chr[chrom] = []
        genes_by_chr[chrom].append(gene)

    print('Mapping selected fragments to gene promoters...')

    atac_by_gene = { gene: np.zeros(2000) for gene in promoter_dict.keys() }
    for chrom in fragments.keys():
        if chrom in genes_by_chr.keys():
            print(chrom)
            flist = fragments[chrom]
            for frag in tqdm(flist, total = len(flist)):
                fstart, fend, count = frag
                # see if the fragment maps to a promoter
                mapped_count = 0
                for gene in genes_by_chr[chrom]:
                    _, pstart, pend, pstrand = promoter_dict[gene]
                    if fstart <= pstart and fend >= pend:
                        mapped_count += 1
                        atac_by_gene[gene][:] += count
                    elif fstart <= pstart and fend >= pstart:
                        mapped_count += 1
                        if pstrand == '+':
                            atac_by_gene[gene][:fend - pstart] += count
                        else:
                            atac_by_gene[gene][pend - fend:] += count
                    elif fstart <= pend and fend >= pend:
                        mapped_count += 1
                        if pstrand == '+':
                            atac_by_gene[gene][fstart - pstart:] += count
                        else:
                            atac_by_gene[gene][:pend - fstart] += count
                    elif fstart >= pstart and fend <= pend:
                        mapped_count += 1
                        if pstrand == '+':
                            atac_by_gene[gene][fstart - pstart : fend - pstart] += count
                        else:
                            atac_by_gene[gene][pend - fend : pend - fstart] += count
                    # don't expect to map one fragment to more than 2 promoters
                    if mapped_count == 2:
                        break

    atac_signals_file = os.path.join(output_dir, 'atac_signals_{}.pkl'.format(cell_type_code))
    with open(atac_signals_file, 'wb') as f:
        pickle.dump(atac_by_gene, f)

    print('Pooled ATAC signals saved.')

    # smooth the signal with a gaussian filter
    genes = list(atac_by_gene.keys())
    atac_signal_matrix = np.zeros((len(genes), 2000))
    for i, gene in enumerate(genes):
        atac_signal_matrix[i] = gaussian_filter1d(atac_by_gene[gene], 20)

    # min-max normalize signal amplitude
    max_fragments = np.max(atac_signal_matrix)
    atac_signal_matrix_norm = atac_signal_matrix / max_fragments

    norm_atac_by_gene = {}
    for i, gene in enumerate(genes):
        norm_atac_by_gene[gene] = atac_signal_matrix_norm[i]

    norm_atac_file = os.path.join(output_dir, 'atac_signals_smoothed_normalized_{}.pkl'.format(cell_type_code))
    with open(norm_atac_file, 'wb') as f:
        pickle.dump(norm_atac_by_gene, f)

    print('Smoothed and normalized ATAC signals saved.')