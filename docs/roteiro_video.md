# Roteiro da demonstração — duração prevista: 4min40s

Somente uma pessoa deve realizar a gravação. O roteiro deixa cerca de 20
segundos de margem em relação ao limite de cinco minutos.

## 0:00–0:30 — Problema e dataset

**Tela:** início do README, mostrando problema e dataset.

**Fala sugerida:**

> Nosso projeto utiliza o dataset IBM Telco Customer Churn para prever se um
> cliente de telecomunicações cancelará o serviço. É uma classificação binária,
> em que Churn igual a Yes representa cancelamento. O conjunto possui 7.043
> clientes e foi escolhido por conter dados numéricos e categóricos, permitindo
> demonstrar preparação de dados, MLP, MLflow e uma API de predição.

## 0:30–1:20 — Pré-processamento e MLP

**Tela:** tabela de configurações em `docs/etapa3_mlflow.md` ou código de
`src/training.py`.

**Fala sugerida:**

> Removemos customerID por ser apenas um identificador e convertemos o target
> para zero e um. Os 11 valores ausentes de TotalCharges são preenchidos pela
> mediana. As variáveis numéricas são padronizadas e as categóricas passam por
> one-hot encoding. Tudo fica no mesmo pipeline do MLP, evitando diferenças
> entre treinamento e API. Usamos 80% dos dados para treino e 20% para teste e
> experimentamos cinco configurações, alterando camadas, ativação, learning
> rate, batch size, número de iterações e regularização.

## 1:20–2:50 — Experimentos e comparação no MLflow

### 1:20–1:45 — Lista dos runs

**Tela:** experimento `telco-churn-mlp` com MLP-01 a MLP-05.

**Fala sugerida:**

> Cada configuração gerou um run separado no MLflow. Registramos os
> hiperparâmetros, accuracy, precision, recall, F1-score, ROC AUC, tempo de
> treinamento e informações da convergência.

### 1:45–2:20 — Gráfico de comparação

**Tela:** gráfico de coordenadas paralelas usando `hidden_layers`, `f1_score`,
`recall`, `precision` e `accuracy`.

**Fala sugerida:**

> Neste gráfico comparamos os cinco experimentos e suas principais métricas.
> Como somente 26,54% dos clientes pertencem à classe de churn, não escolhemos
> o modelo apenas pela accuracy. Priorizamos o F1-score, observando também o
> recall da classe positiva.

### 2:20–2:50 — MLP-05 e artifacts

**Tela:** detalhes do MLP-05 e depois `Artifacts`, exibindo `evaluation` e
`model`.

**Fala sugerida:**

> O MLP-05 apresentou o maior F1-score, aproximadamente 0,593, e o maior
> recall, aproximadamente 0,562. Ele possui duas camadas ocultas de 128 e 64
> neurônios. O pipeline completo foi salvo como artifact do MLflow junto com a
> matriz de confusão, a curva de perda e o relatório de classificação.

## 2:50–4:10 — API local

### 2:50–3:10 — Inicialização

**Tela:** terminal do Uvicorn mostrando `Uvicorn running on
http://127.0.0.1:8000`.

**Fala sugerida:**

> A API foi criada com FastAPI e executa localmente pelo Uvicorn. Na
> inicialização, ela lê o arquivo de seleção e recupera diretamente do MLflow o
> artifact correspondente ao MLP-05.

### 3:10–3:30 — Health check

**Tela:** `GET /health` no Swagger e resposta 200.

**Fala sugerida:**

> O endpoint health confirma que a API está disponível e que o MLP-05 foi
> carregado corretamente.

### 3:30–4:10 — Predição

**Tela:** `POST /predict`, JSON preenchido e resposta 200.

**Fala sugerida:**

> O endpoint predict recebe as 19 características originais de um cliente. O
> próprio pipeline aplica o mesmo pré-processamento do treinamento. A resposta
> informa a classe prevista, o rótulo, a probabilidade de churn e o run usado.
> Neste exemplo, o resultado exibido é [leia o label e a probabilidade mostrados
> na tela].

## 4:10–4:40 — Conclusão

**Tela:** tabela de resultados no README ou gráfico de comparação.

**Fala sugerida:**

> Concluímos que o MLP-05 foi a melhor configuração para este objetivo porque
> obteve o maior F1-score e o maior recall, apresentando o melhor equilíbrio
> para identificar clientes com risco de cancelamento. O projeto registra todos
> os experimentos no MLflow, recupera o modelo escolhido como artifact e o
> disponibiliza por uma API local funcional.

## Lembretes durante a gravação

- Não execute novamente os treinamentos durante o vídeo.
- Não explique todas as colunas do JSON; mostre a entrada e destaque a resposta.
- No resultado da API, leia a probabilidade que estiver na tela, pois ela depende
  do cliente enviado.
- Se errar uma frase, retome com calma; ainda há margem de aproximadamente 20
  segundos.
- Encerre antes de 5:00.
