import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


class GPRegression():
    def __init__(self, X, y, Np, Ntr, Nval):
        # Inizializzazione dell'oggetto GPRegression con i dati di training, testing e validation.
        self.X_tr = X[0:Ntr].values
        self.X_te = X[Ntr:Np - Nval].values
        self.X_val = X[Np - Nval:].values
        self.y_tr = y[0:Ntr].values
        self.y_te = y[Ntr:Np - Nval].values
        self.y_val = y[Np - Nval:].values
        self.Nval = Nval
        self.Ntr = Ntr
        self.Nte = self.X_te.shape[0] #n° di righe di Nte = a quelle di Xte
        self.y_hat_GP = np.zeros(Nval)
        return

    def run(self, r2, s2, t, N):
        # Metodo per eseguire la regressione GP sui dati di validation.
        for k in range(self.Nval):
            # Prendo il k-esimo vettore di features dalla matrice di validation
            x = self.X_val[k, :]
            # Prendo il k-esimo valore target dalla matrice di validation
            y = self.y_val[k]
            dist_tr = []

            # Calcolo le distanze euclidee tra il vettore di validation e tutti i vettori di training
            for i in range(self.Ntr):
                dist_tr.append(euclidean_distance(x, self.X_tr[i, :]))

            # Ordino gli indici in base alle distanze in ordine crescente
            neighbors_index_tr = np.argsort(dist_tr)

            # Inizializzo le strutture dati per memorizzare i vicini più vicini
            neighbors_Xtr_tr = np.zeros([N, self.X_tr.shape[1]])  # 10 righe, 20 colonne, vettori delle features piu vicine
            neighbors_y_tr = np.zeros([N - 1, 1])  # 9 righe e 1 colonna, valori delle distanze di queste N-1 features piu vicine l'enesima si vuole calcolare

            # Prendo i primi N - 1 vicini più vicini
            for i in range(N - 1):
                neighbors_Xtr_tr[i] = self.X_tr[neighbors_index_tr[i]]
                neighbors_y_tr[i] = self.y_tr[neighbors_index_tr[i]]

            # Aggiungo il vettore di validation come l'ultimo vicino più vicino
            neighbors_Xtr_tr[N - 1] = x #??????

            # Creo la matrice di covarianza R utilizzando una funzione kernel
            R = np.zeros([N, N])
            for i in range(N):
                for j in range(N):
                    R[i, j] = t * np.exp(
                        -np.linalg.norm(neighbors_Xtr_tr[i] - neighbors_Xtr_tr[j]) / (2 * r2)) + s2

            # Calcolo le quantità necessarie per la predizione del GP
            k_GP = R[:-1, -1]
            R_N_1 = R[:-1, :-1]
            d = R[-1, -1]
            mu = k_GP.T @ np.linalg.inv(R_N_1) @ neighbors_y_tr
            var = d - k_GP.T @ np.linalg.inv(R_N_1) @ neighbors_y_tr
            std = np.sqrt(var)

            # Assegno la media predetta al k-esimo elemento di y_hat_GP
            self.y_hat_GP[k] = mu

        return {'y_hat': self.y_hat_GP}
