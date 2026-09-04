# Etapa 3 — Treinamento do MLP e registro no MLflow

## Objetivo

Esta etapa treina cinco redes neurais MLP com configuracoes diferentes. Cada
treinamento gera um run separado no experimento `telco-churn-mlp`.

O pre-processador da Etapa 2 e o MLP ficam no mesmo `Pipeline`. Assim, o modelo
salvo pelo MLflow recebe os dados originais do cliente e aplica internamente a
imputacao, padronizacao, one-hot encoding e predicao.

## Configuracoes

| Run | Camadas ocultas | Ativacao | Learning rate | Batch | Max iter | Alpha |
|---|---|---|---:|---:|---:|---:|
| MLP-01 | `(32,)` | ReLU | 0,001 | 32 | 300 | 0,0001 |
| MLP-02 | `(64,)` | ReLU | 0,001 | 32 | 300 | 0,0001 |
| MLP-03 | `(64, 32)` | ReLU | 0,001 | 32 | 400 | 0,0001 |
| MLP-04 | `(64, 32)` | Tanh | 0,001 | 32 | 400 | 0,0001 |
| MLP-05 | `(128, 64)` | ReLU | 0,0001 | 64 | 500 | 0,001 |

Todos usam `solver=adam`, `early_stopping=True`, `random_state=42` e uma
validacao interna de 15% do conjunto de treino.

## O que cada run registra

Parametros:

- hidden layers;
- activation;
- learning rate;
- batch size;
- max iter;
- alpha;
- solver;
- early stopping;
- test size e random state.

Metricas:

- accuracy;
- precision da classe churn;
- recall da classe churn;
- F1-score da classe churn;
- ROC AUC;
- tempo de treinamento, iteracoes, loss final e melhor score de validacao.

Artifacts:

- pipeline completo em `model`;
- matriz de confusao;
- curva de perda;
- relatorio de classificacao;
- arquivo JSON com as metricas da avaliacao.

O script ainda recarrega cada modelo usando sua URI `runs:/.../model` e confere
se ele continua produzindo a mesma predicao. Isso valida que o artifact podera
ser recuperado pela futura API.

## Resultado da validacao do codigo

| Run | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| MLP-01 | 0,7885 | 0,6159 | 0,5401 | 0,5755 | 0,8397 |
| MLP-02 | 0,7991 | 0,6757 | 0,4679 | 0,5529 | 0,8404 |
| MLP-03 | 0,7899 | 0,6175 | 0,5481 | 0,5807 | 0,8398 |
| MLP-04 | 0,7949 | 0,6778 | 0,4332 | 0,5285 | 0,8418 |
| MLP-05 | 0,7956 | 0,6287 | 0,5615 | 0,5932 | 0,8391 |

Esses valores servem como referencia. Com `random_state=42`, uma nova execucao
deve produzir resultados iguais ou muito proximos. A escolha oficial do melhor
run deve ser demonstrada na interface do MLflow, e nao somente nesta tabela.

## Executar os cinco treinamentos

Na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m src.train_experiments
```

O comando cria localmente o banco `mlflow.db` e a pasta `mlartifacts`. Eles nao
devem ser apagados antes da gravacao do video, pois contem os runs e os modelos.

## Abrir o MLflow

Depois que os cinco treinamentos terminarem:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5001
```

Com o comando ainda executando, abrir no navegador:

```text
http://127.0.0.1:5001
```

Na interface, selecionar `telco-churn-mlp`. Os cinco runs devem aparecer. A
comparacao e a escolha do melhor modelo serao feitas na Etapa 4.

Avisos amarelos relacionados a serializacao `cloudpickle` podem aparecer no
terminal. Eles alertam para carregar somente artifacts confiaveis e nao
significam que o treinamento falhou. O sucesso e indicado pela mensagem
`Cinco runs registrados com sucesso`.
