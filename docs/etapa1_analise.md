# Etapa 1 — Definicao e analise inicial

## Problema

O projeto utilizara classificacao binaria para prever se um cliente de uma
empresa de telecomunicacoes cancelara o servico. A classe positiva sera
`Churn = Yes`.

Esse problema foi escolhido porque possui uma aplicacao de negocio clara:
identificar antecipadamente clientes em risco permite priorizar acoes de
retencao.

## Dataset

- Nome: IBM Telco Customer Churn.
- Fonte: repositorio publico da IBM.
- Arquivo local: `data/telco_churn.csv`.
- Registros: 7.043.
- Colunas: 21.
- Linhas completamente duplicadas: 0.
- Identificadores de cliente duplicados: 0.
- SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.

O checksum registra exatamente a versao do CSV utilizada pelo grupo e ajuda a
garantir que todos executem os experimentos sobre os mesmos dados.

## Variavel alvo

| Valor | Registros | Percentual |
|---|---:|---:|
| `No` | 5.174 | 73,46% |
| `Yes` | 1.869 | 26,54% |

O target apresenta desequilibrio moderado. Por isso, a acuracia nao sera usada
isoladamente para escolher o melhor modelo. A comparacao dara prioridade ao
F1-score da classe positiva e tambem observara precision, recall e ROC AUC.

## Features selecionadas

A coluna `customerID` sera removida porque funciona apenas como identificador e
nao representa uma caracteristica generalizavel do cliente. A coluna `Churn`
sera separada como target. As 19 colunas restantes serao inicialmente mantidas.

Features numericas:

- `tenure`;
- `MonthlyCharges`;
- `TotalCharges`.

Features categoricas ou indicadoras:

- `gender`;
- `SeniorCitizen`;
- `Partner`;
- `Dependents`;
- `PhoneService`;
- `MultipleLines`;
- `InternetService`;
- `OnlineSecurity`;
- `OnlineBackup`;
- `DeviceProtection`;
- `TechSupport`;
- `StreamingTV`;
- `StreamingMovies`;
- `Contract`;
- `PaperlessBilling`;
- `PaymentMethod`.

Embora `SeniorCitizen` seja armazenada como 0 ou 1, ela representa uma categoria
binaria e sera tratada junto das variaveis categoricas.

## Qualidade dos dados

O CSV nao apresenta valores nulos convencionais. Entretanto, existem 11 campos
em branco em `TotalCharges`. Depois da conversao dessa coluna para tipo numerico,
esses campos passam a ser valores ausentes e precisarao de imputacao.

Decisoes para a proxima etapa:

1. Converter `TotalCharges` para numerico.
2. Preencher valores numericos ausentes com a mediana calculada no treino.
3. Preencher categoricos ausentes com a moda, caso aparecam novas ausencias.
4. Aplicar `StandardScaler` somente nas features numericas.
5. Aplicar `OneHotEncoder(handle_unknown="ignore")` nas categoricas.
6. Converter o target `No/Yes` para `0/1`.
7. Separar 80% para treino e 20% para teste com `stratify=y` e
   `random_state=42`.
8. Encapsular todo o pre-processamento e o MLP em um unico `Pipeline`.

O `Pipeline` e essencial para que a API aplique exatamente as mesmas
transformacoes utilizadas durante o treinamento.

## Ambiente validado

- Python 3.12.13.
- pandas 2.3.3.
- scikit-learn 1.9.0.
- MLflow 3.15.2.
- FastAPI 0.141.1.
- Uvicorn 0.52.4.
- Pydantic 2.13.4.

Para reproduzir o diagnostico:

```bash
python src/analyze_data.py
```

O resultado estruturado e salvo em `docs/data_profile.json`.

