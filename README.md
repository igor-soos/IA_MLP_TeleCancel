# MLP + MLflow + API de predição de churn

Projeto acadêmico de classificação binária que utiliza uma rede neural MLP
para prever o cancelamento de clientes de telecomunicações. Cinco
configurações são treinadas e comparadas no MLflow; o melhor pipeline é
recuperado como artifact e disponibilizado por uma API FastAPI local.

## Problema e dataset

- Dataset: IBM Telco Customer Churn.
- Fonte: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data).
- Registros: 7.043 clientes.
- Target: `Churn` (`Yes` ou `No`).
- Objetivo: identificar clientes com maior risco de cancelamento.
- Tipo: classificação binária.

O arquivo utilizado está em `data/telco_churn.csv`, permitindo reproduzir os
experimentos sem baixar dados adicionais.

## Fluxo da solução

```text
CSV -> preparação dos dados -> cinco MLPs -> MLflow -> seleção do MLP-05
    -> carregamento do artifact -> FastAPI -> previsão
```

O pré-processamento e o MLP ficam no mesmo `Pipeline` do Scikit-learn. Por
isso, a API recebe as 19 características originais do cliente e o artifact
aplica automaticamente imputação, padronização, one-hot encoding e predição.

## Resultados principais

| Run | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| MLP-01 | 0,7885 | 0,6159 | 0,5401 | 0,5755 | 0,8397 |
| MLP-02 | **0,7991** | 0,6757 | 0,4679 | 0,5529 | 0,8404 |
| MLP-03 | 0,7899 | 0,6175 | 0,5481 | 0,5807 | 0,8398 |
| MLP-04 | 0,7949 | **0,6778** | 0,4332 | 0,5285 | **0,8418** |
| MLP-05 | 0,7956 | 0,6287 | **0,5615** | **0,5932** | 0,8391 |

O MLP-05 foi selecionado porque apresentou o maior F1-score e o maior recall.
Embora o MLP-02 tenha obtido accuracy ligeiramente maior, ele identificou uma
parcela menor dos clientes que realmente cancelaram. Para o problema de churn,
o equilíbrio entre precision e recall é mais útil do que considerar somente a
accuracy.

Configuração vencedora:

- camadas ocultas: `(128, 64)`;
- ativação: `relu`;
- learning rate: `0.0001`;
- batch size: `64`;
- máximo de iterações: `500`;
- regularização alpha: `0.001`;
- solver: `adam`;
- early stopping: ativado.

## Estrutura do projeto

```text
mlp-mlflow-api/
|-- config/
|   `-- README.md
|-- data/
|   `-- telco_churn.csv
|-- docs/
|   |-- etapa1_analise.md
|   |-- etapa2_preprocessamento.md
|   |-- etapa3_mlflow.md
|   |-- etapa4_selecao.md
|   |-- etapa5_api.md
|   |-- etapa6_entrega.md
|   |-- roteiro_video.md
|   |-- data_profile.json
|   `-- preprocessing_summary.json
|-- examples/
|   `-- predict_churn.json
|-- src/
|   |-- analyze_data.py
|   |-- api.py
|   |-- model_selection.py
|   |-- prepare_data.py
|   |-- preprocessing.py
|   |-- select_best_model.py
|   |-- train_experiments.py
|   `-- training.py
|-- tests/
|   |-- test_api.py
|   |-- test_model_selection.py
|   |-- test_preprocessing.py
|   `-- test_training.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

`mlflow.db`, `mlartifacts`, `config/best_model.json` e
`docs/experiment_validation.json` são gerados no computador que executa o
projeto. Eles não estão no código-fonte compactado.

## Requisitos

- Python 3.12;
- Windows PowerShell, Linux ou macOS;
- navegador para acessar MLflow e Swagger.

As versões das bibliotecas estão fixadas em `requirements.txt`.

## Instalação no Windows PowerShell

Na raiz do projeto:

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Não é necessário ativar o ambiente virtual, porque os comandos deste README
chamam diretamente o Python localizado em `.venv`.

## Execução completa

### 1. Analisar o dataset

```powershell
.\.venv\Scripts\python.exe src\analyze_data.py
```

Resultado esperado: 7.043 registros, 21 colunas, 11 valores vazios em
`TotalCharges` e 26,54% de churn positivo.

### 2. Validar o pré-processamento

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data
```

Resultado esperado: 5.634 clientes no treino, 1.409 no teste, 19 features
originais transformadas em 46 features numéricas e nenhum valor ausente após o
tratamento.

### 3. Executar os testes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Resultado esperado:

```text
Ran 16 tests
OK
```

### 4. Treinar os cinco MLPs

```powershell
.\.venv\Scripts\python.exe -m src.train_experiments
```

Esse comando cria:

- `mlflow.db`, com runs, parâmetros e métricas;
- `mlartifacts`, com modelos e artifacts de avaliação;
- `docs/experiment_validation.json`, com o resumo da execução.

### 5. Abrir a interface do MLflow

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5001
```

Com o terminal aberto, acessar:

```text
http://127.0.0.1:5001
```

Abrir o experimento `telco-churn-mlp`. Para a comparação usada no vídeo,
selecionar os cinco runs, abrir `Compare` e usar o gráfico de coordenadas
paralelas com `hidden_layers`, `f1_score`, `recall`, `precision` e `accuracy`.

### 6. Selecionar o melhor modelo

Em outro terminal, na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m src.select_best_model
```

O script seleciona por F1-score, usando recall e accuracy como desempates,
valida o artifact e cria `config/best_model.json`. A URI armazenada segue o
formato `runs:/RUN_ID/model`.

### 7. Iniciar a API

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Manter o terminal aberto e acessar:

```text
http://127.0.0.1:8000/docs
```

## Endpoints da API

| Método | Endpoint | Finalidade |
|---|---|---|
| GET | `/` | Informações básicas e links da API |
| GET | `/health` | Confirma o carregamento do MLP-05 |
| POST | `/predict` | Recebe um cliente e retorna a previsão |

`GET /health` também pode ser aberto diretamente no navegador:

```text
http://127.0.0.1:8000/health
```

Resultado esperado:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_run": "MLP-05"
}
```

Para testar `POST /predict`, use o Swagger em `/docs`: abrir o endpoint, clicar
em `Try it out` e depois em `Execute`. Digitar `/predict` na barra do navegador
não funciona porque o navegador envia uma requisição GET, enquanto esse
endpoint exige POST com um JSON.

Exemplo de resposta:

```json
{
  "prediction": 1,
  "label": "Churn",
  "probability": 0.8299,
  "model_run": "MLP-05"
}
```

O valor da probabilidade depende dos dados enviados. `prediction = 1` significa
previsão de cancelamento; `prediction = 0` significa previsão de permanência.

## Documentação da entrega

- `docs/etapa6_entrega.md`: revisão, checklist e preparação do ambiente.
- `docs/roteiro_video.md`: sequência de telas e falas para um vídeo de até
  cinco minutos.
- `docs/etapa1_analise.md` a `docs/etapa5_api.md`: explicação de cada etapa.

## Status

- [x] Etapa 1: estrutura, dataset e análise inicial.
- [x] Etapa 2: pré-processamento e divisão treino/teste.
- [x] Etapa 3: cinco experimentos e artifacts no MLflow.
- [x] Etapa 4: comparação e seleção do MLP-05.
- [x] Etapa 5: API FastAPI local.
- [x] Etapa 6: testes finais, README, checklist e roteiro da demonstração.

## Encerrar os servidores

Nos terminais do MLflow e do Uvicorn, pressione `Ctrl+C`.

## Solução de problemas

- `mlflow.db nao encontrado`: execute `python -m src.train_experiments`.
- `config/best_model.json nao encontrado`: execute
  `python -m src.select_best_model`.
- Porta ocupada: encerre o processo anterior com `Ctrl+C` ou use outra porta.
- Página não abre: confirme que o terminal do respectivo servidor continua
  aberto.
- Avisos sobre `cloudpickle`: são avisos de segurança de serialização e não
  indicam falha no treinamento realizado com artifacts confiáveis do projeto.
