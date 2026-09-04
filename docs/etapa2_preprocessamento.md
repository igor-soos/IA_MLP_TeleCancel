# Etapa 2 — Pre-processamento dos dados

## O que esta etapa resolve

O MLP nao consegue trabalhar diretamente com textos como `Female`, `Yes` ou
`Month-to-month`. Ele tambem nao pode receber campos ausentes. Esta etapa cria
um pre-processador que converte o dataset em uma matriz numerica pronta para o
treinamento.

O fluxo implementado e:

```text
CSV -> limpeza inicial -> separacao treino/teste -> pre-processador numerico
```

## Preparacao inicial

- `customerID` e removido por ser somente um identificador.
- `Churn` e separado como target.
- `Churn = No` vira `0` e `Churn = Yes` vira `1`.
- Espacos em branco em `TotalCharges` viram valores ausentes.
- As 19 colunas restantes continuam como features.

## Divisao dos dados

| Conjunto | Registros | Uso |
|---|---:|---|
| Treino | 5.634 | Aprender transformacoes e treinar o MLP |
| Teste | 1.409 | Avaliar o modelo com dados nao usados no treino |

A divisao usa `random_state=42`, `test_size=0.20` e `stratify=y`. O `stratify`
mantem praticamente a mesma proporcao de churn nos dois conjuntos.

## Transformacoes numericas

Aplicadas a `tenure`, `MonthlyCharges` e `TotalCharges`:

1. `SimpleImputer(strategy="median")` preenche valores ausentes com a mediana
   calculada apenas no conjunto de treino.
2. `StandardScaler` coloca as variaveis em escalas comparaveis, o que e
   especialmente importante para uma rede neural.

## Transformacoes categoricas

Aplicadas as outras 16 features:

1. `SimpleImputer(strategy="most_frequent")` trata eventuais valores ausentes.
2. `OneHotEncoder(handle_unknown="ignore")` converte cada categoria em colunas
   de 0 e 1 e permite que a futura API receba uma categoria desconhecida sem
   interromper a execucao.

Depois do one-hot encoding, as 19 features originais tornam-se 46 features
numericas. Nao houve valores ausentes depois das transformacoes.

## Prevencao de vazamento de dados

O pre-processador aprende mediana, moda, media, desvio padrao e categorias
somente com os 5.634 registros de treino. O conjunto de teste e apenas
transformado. Isso evita que informacoes do teste influenciem o treinamento.

Na proxima etapa, esse mesmo pre-processador sera colocado junto do
`MLPClassifier` em um unico `Pipeline`. O pipeline completo sera registrado no
MLflow e depois carregado pela API.

## Como executar

Na raiz do projeto, com o ambiente virtual configurado:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data
```

Para executar os testes automatizados:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

