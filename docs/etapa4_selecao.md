# Etapa 4 — Comparacao e selecao do melhor modelo

## Comparacao no MLflow

Os cinco runs foram comparados na propria interface do MLflow. A metrica
principal definida foi o F1-score da classe positiva `Churn = Yes`. O recall foi
observado em seguida porque, neste problema, e importante reduzir clientes que
cancelariam, mas nao seriam identificados pelo modelo.

| Run | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| MLP-01 | 0,7885 | 0,6159 | 0,5401 | 0,5755 | 0,8397 |
| MLP-02 | 0,7991 | 0,6757 | 0,4679 | 0,5529 | 0,8404 |
| MLP-03 | 0,7899 | 0,6175 | 0,5481 | 0,5807 | 0,8398 |
| MLP-04 | 0,7949 | 0,6778 | 0,4332 | 0,5285 | 0,8418 |
| MLP-05 | 0,7956 | 0,6287 | 0,5615 | 0,5932 | 0,8391 |

## Modelo escolhido

O MLP-05 foi selecionado porque apresentou simultaneamente o maior F1-score
(`0,5932`) e o maior recall (`0,5615`). O MLP-02 apresentou accuracy um pouco
maior, mas identificou uma parcela menor dos clientes que realmente cancelaram.

Configuracao do MLP-05:

- hidden layers: `(128, 64)`;
- activation: `relu`;
- learning rate: `0.0001`;
- batch size: `64`;
- max iter: `500`;
- alpha: `0.001`;
- solver: `adam`;
- early stopping: ativado.

## Executar a selecao

Depois de executar os cinco treinamentos:

```powershell
.\.venv\Scripts\python.exe -m src.select_best_model
```

O script consulta o MLflow, seleciona o maior F1-score, valida o artifact do
modelo e cria `config/best_model.json`. Esse arquivo contem uma URI no formato:

```text
runs:/RUN_ID/model
```

A API da proxima etapa lera essa URI para recuperar exatamente o pipeline do
MLP-05, incluindo o pre-processamento.

## Justificativa para a apresentacao

> Selecionamos o MLP-05 porque ele apresentou o maior F1-score e o maior recall.
> Como o objetivo e identificar clientes com risco de cancelamento, priorizamos
> um modelo com melhor equilibrio entre precision e recall e com maior capacidade
> de encontrar os clientes que realmente cancelaram.

