import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pandas as pd

def get_n_top(features, comp, n):
    top_val = -1
    top_feature = ''
    top_features = []
    while len(top_features) < n:
        for i in range(len(comp)):
            if abs(comp[i]) > top_val and features[i] not in top_features:
                top_val = abs(comp[i])
                top_feature = features[i]
        top_features += [top_feature]
        top_val = -1
        top_feature = ''
    return top_features

if __name__ == "__main__":
     files = os.listdir(os.getcwd())
     if "02_14_21_28_cleaned.csv" in files:
        df = pd.read_csv("02_14_21_28_cleaned.csv")
        features = df[["Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd Pkts/s","Bwd Pkts/s","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","Down/Up Ratio","Pkt Size Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg","Subflow Fwd Pkts","Subflow Bwd Pkts","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"]]
        # Standardize the data
        sclr = StandardScaler()
        features = sclr.fit_transform(features)
        # Get first 12 components
        pca = PCA(n_components=12)
        pca.fit(features)
        # Sort them by load
        outfile = open("top_features_by_component.txt", "w")
        for i in range(len(pca.components_)):
            out = "PC: " + str(i+1) + ", Features in Descending Load: " + str(get_n_top(["Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd Pkts/s","Bwd Pkts/s","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","Down/Up Ratio","Pkt Size Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg","Subflow Fwd Pkts","Subflow Bwd Pkts","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"], pca.components_[i], len(pca.components_[i]))) + ", PEV: " + str(pca.explained_variance_ratio_[i]) + "\n"
            outfile.write(out)
        outfile.close()
        # Plot Top 5 Features of Biplot
        top_features = ['Pkt Len Max', 'Pkt Len Std', 'Pkt Len Var', 'Pkt Len Mean', 'Pkt Size Avg']
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        for i in range(len(top_features)):
            k = ["Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd Pkts/s","Bwd Pkts/s","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","Down/Up Ratio","Pkt Size Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg","Subflow Fwd Pkts","Subflow Bwd Pkts","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"].index(top_features[i])
            ax.scatter(pca.components_[0,k], pca.components_[1,k], label=top_features[i])
        plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PC 1 Top 5 Best Feature Biplot")
        plt.savefig("top_5_pc1.png")
        plt.clf()