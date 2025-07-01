import csv

def is_valid_dna(seq):
    """Check if the sequence only contains valid DNA nucleotides."""
    valid_nucleotides = {'A', 'T', 'G', 'C'}
    return all(base in valid_nucleotides for base in seq.upper())

def gc_content(seq):
    """Calculate the GC content of a DNA sequence as a percentage."""
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    return round((gc_count / len(seq)) * 100, 2) if len(seq) > 0 else 0

def read_fasta(file_path):
    """Read sequences from a FASTA file and return a dictionary of ID: sequence."""
    sequences = {}
    with open(file_path, 'r') as file:
        seq_id = None
        seq_lines = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if seq_id:
                    sequences[seq_id] = ''.join(seq_lines)
                seq_id = line[1:]  
                seq_lines = []
            else:
                seq_lines.append(line)
        if seq_id:
            sequences[seq_id] = ''.join(seq_lines)  
    return sequences

def unique_nucleotides(sequences):
    """Return a set of unique nucleotides found in all sequences."""
    nucleotides = set()
    for seq in sequences.values():
        nucleotides.update(seq.upper())
    return nucleotides

def save_results_to_csv(sequences, output_file):
    """Save ID, length, GC content, and validity of sequences to a CSV file."""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Length', 'GC_Content(%)', 'Is_Valid_DNA'])
        for seq_id, seq in sequences.items():
            length = len(seq)
            gc = gc_content(seq)
            valid = is_valid_dna(seq)
            writer.writerow([seq_id, length, gc, valid])

if __name__ == "__main__":
    fasta_file = "seq.fasta"         
    output_csv = "dna_analysis_results.csv"

    sequences = read_fasta(fasta_file)

    print("Unique nucleotides across all sequences:", unique_nucleotides(sequences))

    save_results_to_csv(sequences, output_csv)
    print(f"Results saved to '{output_csv}'")
