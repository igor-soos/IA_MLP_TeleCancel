# Etapa 5 — API local de predicao

## Objetivo

A API recebe os dados originais de um cliente e retorna se o modelo preve
cancelamento. Ela carrega o MLP-05 pela URI gravada em
`config/best_model.json`.

Como o modelo salvo e um `Pipeline`, a API nao precisa criar manualmente as 46
features. O proprio pipeline aplica imputacao, padronizacao, one-hot encoding e
predicao.

## Endpoints

| Metodo | Endpoint | Funcao |
|---|---|---|
| GET | `/` | Informacoes basicas da API |
| GET | `/health` | Confirma que o modelo foi carregado |
| POST | `/predict` | Recebe um cliente e devolve a previsao |

## Pre-requisitos

Antes de iniciar a API, devem existir na raiz do projeto:

- `mlflow.db`;
- `mlartifacts`;
- `config/best_model.json`.

Se `best_model.json` ainda nao existir, executar:

```powershell
.\.venv\Scripts\python.exe -m src.select_best_model
```

## Iniciar a API

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Manter esse terminal aberto e acessar:

```text
http://127.0.0.1:8000/docs
```

## Testar no Swagger

1. Abrir `POST /predict`.
2. Clicar em `Try it out`.
3. Manter ou editar o JSON de exemplo.
4. Clicar em `Execute`.
5. Conferir se o codigo HTTP e `200`.

Exemplo de resposta:

```json
{
  "prediction": 1,
  "label": "Churn",
  "probability": 0.8299,
  "model_run": "MLP-05"
}
```

`prediction = 1` significa que o modelo previu cancelamento. `prediction = 0`
significa que o modelo previu permanencia. A probabilidade e o risco estimado de
churn, entre 0 e 1.

## Testar pelo PowerShell

Com a API executando em outro terminal:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -InFile ".\examples\predict_churn.json"
```

Para encerrar a API, voltar ao terminal do Uvicorn e pressionar `Ctrl+C`.
