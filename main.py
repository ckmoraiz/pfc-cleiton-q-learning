# Importando bibliotecas
import src.functions as fct


#####################
#Lógica de interface#
#####################
print("\nEscolha:")
print("[1]Simulação única.")
print("[2]Simulação de treino com N episódios.")
choice = int(input(':'))

if choice == 1:
    fct.treino(1)
elif choice == 2:
    print("Insira quantidade N de episódios para iterações:")
    eps = int(input('Episódios:'))
    fct.treino(eps)


