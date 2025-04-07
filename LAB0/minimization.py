# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:33:35 2023

@author: d001834
"""
import numpy as np  # Importa la libreria NumPy
import matplotlib.pyplot as plt  # Importa la libreria Matplotlib


def condition_absolute_improvement(fw0, fw1, eps):  # Definisce una funzione per la condizione di miglioramento assoluto
    return np.abs(fw0 - fw1) < eps


def condition_relative_improvement(fw0, fw1, eps):  # Definisce una funzione per la condizione di miglioramento relativo
    return np.abs(fw0 - fw1) / np.maximum(1, np.abs(fw0)) < eps


def condition_absolute_value(w0, w1, eps):  # Definisce una funzione per la condizione di valore assoluto
    return np.linalg.norm(w0 - w1) < eps


def condition_relative_value(w0, w1, eps):  # Definisce una funzione per la condizione di valore relativo
    return np.linalg.norm(w0 - w1) / np.maximum(1, np.linalg.norm(w0)) < eps


class SolverMinProblem:  # Definisce la classe base per la risoluzione di problemi di minimizzazione
    def __init__(self, A=np.eye(3), y=np.ones(shape=(3,))):  # Costruttore della classe, np.eye crea una matrice identità 3x3,  np.one un vettore con tutti 1, shape=(3,) da la dimensione al vettore
        self.A = A  # Assegna la matrice A
        self.y = y  # Assegna il vettore y
        self.Np = A.shape[0]  # Calcola il numero di righe di A
        self.Nf = A.shape[1]  # Calcola il numero di colonne di A
        self.result = np.zeros(shape=(self.Nf,))  # Inizializza il risultato come un vettore di zeri

    def plot_result(self):  # Metodo per visualizzare il risultato
        plt.figure()  # Crea una nuova figura
        plt.title('LLS solution')  # Assegna un titolo al grafico
        plt.plot(self.result, label='w_hat')  # Plotta il risultato con etichetta 'w_hat'
        plt.xlabel('n')  # Etichetta l'asse x
        plt.ylabel('w_hat(n)')  # Etichetta l'asse y
        plt.legend()  # Aggiungi la legenda
        plt.grid()  # Abilita la griglia nel grafico
        plt.show()  # Mostra il grafico


class SolverLLS(SolverMinProblem):  # Estende la classe SolverMinProblem per la risoluzione dei minimi quadrati
    def solve(self):  # Metodo per risolvere il problema
        A = self.A  # Assegna la matrice A
        y = self.y  # Assegna il vettore y
        self.result = np.linalg.pinv(A) @ y  # Calcola la soluzione usando l'inversa generalizzata di A


class SolverGradient(SolverMinProblem):  # Estende la classe SolverMinProblem per la risoluzione tramite metodo del gradiente
    def solve(self, gamma=1e-3, Nit=100):  # Metodo di risoluzione con parametri gamma e Nit
        A = self.A  # Assegna la matrice A
        y = self.y  # Assegna il vettore y
        w_hat0 = np.random.rand(A.shape[1])  # Inizializza w_hat0 con valori casuali
        i = 0  # Inizializza un contatore
        while True:  # Ciclo di iterazioni
            gradient = 2 * A.T @ (A @ w_hat0 - y)  # Calcola il gradiente
            w_hat1 = w_hat0 - gamma * gradient  # Aggiorna la soluzione
            i += 1  # Incrementa il contatore

            err0 = np.linalg.norm(y - A @ w_hat0) ** 2  # Calcola l'errore attuale
            err1 = np.linalg.norm(y - A @ w_hat1) ** 2  # Calcola il nuovo errore
            if i >= Nit > 0 or condition_relative_improvement(err0, err1, eps=1e-30):  # Condizione di convergenza
                break  # Esci dal ciclo
            w_hat0 = w_hat1  # Aggiorna la soluzione


class SolverSteepestDescent(SolverLLS):  # Estende la classe SolverLLS per la risoluzione tramite steepest descent
    def solve(self, Nit=100):  # Metodo di risoluzione con parametro Nit
        A = self.A  # Assegna la matrice A
        y = self.y  # Assegna il vettore y
        w_hat0 = np.random.rand(A.shape[1])  # Inizializza w_hat0 con valori casuali
        i = 0  # Inizializza un contatore
        print("culooooooo")  # Stampa un messaggio (potrebbe essere un segnaposto)
        hessian = 2 * A.T @ A  # Calcola la matrice Hessiana
        while True:  # Ciclo di iterazioni
            gradient = 2 * A.T @ (A @ w_hat0 - y)  # Calcola il gradiente
            N = np.linalg.norm(gradient) ** 2  # Calcola la norma del gradiente
            D = gradient.T @ hessian @ gradient  # Calcola un termine D
            gamma = N / D  # Calcola il passo gamma
            w_hat1 = w_hat0 - gamma * gradient  # Aggiorna la soluzione
            i += 1  # Incrementa il contatore

            err0 = np.linalg.norm(y - A @ w_hat0) ** 2  # Calcola l'errore attuale
            err1 = np.linalg.norm(y - A @ w_hat1) ** 2  # Calcola il nuovo errore
            if i >= Nit > 0 or condition_relative_improvement(err0, err1, eps=1e-30):  # Condizione di convergenza
                break  # Esci dal ciclo
            w_hat0 = w_hat1  # Aggiorna la soluzione
        self.result = w_hat1  # Assegna il risultato
    
        
        