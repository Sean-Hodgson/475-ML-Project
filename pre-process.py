import pandas as pd
import os
from sklearn.impute import KNNImputer
import numpy as np
import statistics as stat

def combine_csvs():
    frames = []
    filenames = os.listdir(os.getcwd())
    filenames.remove("main.py")
    for file in filenames:
        if '02' in file and ('14' in file or '21' in file or '28' in file):
            df = pd.read_csv(file)
            frames += [df[df["Dst Port"] != "Dst Port"]]
    df = pd.concat(frames)
    df.to_csv("02_14_21_28.csv")

if __name__ == '__main__':
    files = os.listdir(os.getcwd())
    if "02_14_21_28.csv" not in files:
        combine_csvs()
    if "02_14_21_28_cleaned.csv" not in files:
        df = pd.read_csv("02_14_21_28.csv")
        # Drop Non-Encrypted Features
        df = df.drop(axis=1, columns=["Dst Port", "Protocol", "Timestamp", "Fwd PSH Flags", "Bwd PSH Flags", 
                                    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Len", "Bwd Header Len", "FIN Flag Cnt", 
                                    "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", 
                                    "CWE Flag Count", "ECE Flag Cnt", "Fwd Seg Size Avg", "Bwd Seg Size Avg", 
                                    "Fwd Seg Size Min", "Fwd Act Data Pkts", "Init Fwd Win Byts", "Init Bwd Win Byts", 
                                    "Flow Byts/s", "Fwd Byts/b Avg", "Bwd Byts/b Avg", "Subflow Fwd Byts", "Subflow Bwd Byts"])
        # Encode Labels
        df["Label"] = df["Label"].replace({"Benign": 0, "FTP-BruteForce": 1, "SSH-Bruteforce": 2, 
                                           "DDOS attack-LOIC-UDP": 3, "DDOS attack-HOIC": 4, "Infilteration": 5})
        # Replace Infinities with NA
        df = df.replace({np.inf: np.nan, -np.inf: np.nan})
        # Fill missing values
        imp = KNNImputer()
        df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
        # Drop Outliers
        for col in df.columns:
            df = df[df[col] <= ((3 * stat.stdev(df[col])) + stat.mean(df[col]))]
        df.to_csv("02_14_21_28_cleaned.csv")

        
