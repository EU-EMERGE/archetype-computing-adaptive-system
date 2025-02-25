import torch
from torch.utils.data import Dataset
from aeon.datasets import load_from_ts_file

class PenDigitsDataset(Dataset):
    def __init__(self, ts_file):
        """
        ts_file: percorso al file .ts
        """
        # Carica dal file .ts in due dataframe:
        # X_df con le serie, y_df con le etichette
        # In alcuni dataset la label è integrata in X_df
        X_df, y_df = load_from_ts_file(ts_file)

        # Se y_df non è un DataFrame, potrebbe essere una serie di etichette
        # e X_df potrebbe contenere, per ciascun campione, una (o più) serie
        # di dimensioni diverse. Ogni colonna di X_df può essere una dimensione
        # temporale.
        #
        # Nel caso delle PenDigits potresti avere una sola riga con la forma:
        # X_df.iloc[i,0] -> la serie di coordinate X
        # X_df.iloc[i,1] -> la serie di coordinate Y
        # y_df[i]        -> la label
        # Verifica la struttura effettiva del file .ts.
        #
        # Esempio: se i = 0, X_df.iloc[0, 0] -> pd.Series (es. 8 punti),
        #          X_df.iloc[0, 1] -> pd.Series (stessa lunghezza).
        
        self.data = []
        self.labels = []

        for i in range(len(X_df)):
            # Supponendo che ci siano esattamente due colonne: X e Y
            coords_x = X_df[i, 0]  # array di lunghezza 8
            coords_y = X_df[i, 1]  # array di lunghezza 8

            # Creiamo un tensore [8, 2]
            coords = torch.tensor(list(zip(coords_x, coords_y)), dtype=torch.float)

            # Label associata
            label = y_df[i]
            
            self.data.append(coords)
            self.labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_seq = self.data[idx]   # shape [8, 2]
        y = self.labels[idx]
        return x_seq, y