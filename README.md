# Projeto de Conclusão de Curso  
## Engenharia Mecatrônica - UFU 

## Objetivos
O objetivo deste projeto é aplicar o método Q-Learning no controle de um aeropêndulo.
Foi utilizado um modelo computacional do aeropêndulo e desenvolvido um agente controlador capaz de aprender (estimando Q-Values) através da interação com o modelo.

## Arquitetura
### Diagrama simplicado

<p align="center">
<img src="img/diagrama-codigo-final.png" width="500" heigth="550">
</p>

### Funções
<p align="center">
<img src="img/functions.png" width="550" heigth="600">
</p>


## Utilização

### Pré requisitos

- Python
https://python.org.br/instalacao-windows/
https://python.org.br/instalacao-linux/ 
<br/>

- pip
```
python3 -m pip install --user --upgrade pip
```
<br/>

- Bibliotecas
```
pip install -r requirements.txt
```

### Simulação

```
python3 -m main.py
```



## Exemplo

<p align="center">
<img src="img/exemplo-codigo.png" width="450" heigth="600">
</p>

<p align="center">
<img src="img/theta-p-15-graus.png" width="450" heigth="600">
</p>

<p align="center">
<img src="img/omega2-15-graus.png" width="400" heigth="600">
</p>

<p align="center">
<img src="img/q-values-1-eps.png" width="550" heigth="600">
</p>
