import pandas as pd
import requests
import os

CLINVAR_TSV_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
OUTPUT_FILE = "clinvar_snv_sample.csv"

def download_clinvar_tsv(url=CLINVAR_TSV_URL, out_gz="variant_summary.txt.gz"):
    print("Downloading ClinVar variant summary...")
    r = requests.get(url, stream=True)
    with open(out_gz, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")
    return out_gz
def extract_sample_snvs(tsv_gz, output_csv, n=10000):
    print("Extracting SNV sample...")
    df = pd.read_csv(tsv_gz, sep='\t', compression='gzip', low_memory=False)
    # Filter for human SNVs with clinical significance
    snv_df = df[(df['Type'] == 'single nucleotide variant') & (df['Assembly'] == 'GRCh38') & (df['ClinicalSignificance'].notnull())]
    snv_df = snv_df[['GeneSymbol', 'Chromosome', 'Start', 'ReferenceAllele', 'AlternateAllele', 'ClinicalSignificance', 'PhenotypeList', 'Assembly', 'Type']]
    snv_df = snv_df.head(n)
    snv_df.to_csv(output_csv, index=False)
    print(f"Saved sample SNVs to {output_csv}")

if __name__ == "__main__":
    gz_file = download_clinvar_tsv()
    extract_sample_snvs(gz_file, OUTPUT_FILE) 