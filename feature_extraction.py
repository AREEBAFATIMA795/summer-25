import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def label_snv(row):
    # Mark as deleterious if "Pathogenic" or "Likely_pathogenic" in ClinicalSignificance
    if 'Pathogenic' in row['ClinicalSignificance']:
        return 1
    else:
        return 0

def extract_features(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    # Sequence context placeholder: use N's (real pipeline would fetch sequence from genome)
    df['context'] = ['N'*21]*len(df)  # 10bp up/down + variant
    df['ref_allele'] = df['ReferenceAllele']
    df['alt_allele'] = df['AlternateAllele']
    df['label'] = df.apply(label_snv, axis=1)
    # Encode alleles
    le = LabelEncoder()
    df['ref_allele_enc'] = le.fit_transform(df['ref_allele'])
    df['alt_allele_enc'] = le.fit_transform(df['alt_allele'])
    # Only keep features and label
    features = df[['ref_allele_enc', 'alt_allele_enc', 'label']]
    features.to_csv(output_csv, index=False)
    print(f"Saved features to {output_csv}")

if __name__ == "__main__":
    extract_features('clinvar_snv_sample.csv', 'features.csv')