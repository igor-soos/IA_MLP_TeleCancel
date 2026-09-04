# Etapa 6 — Revisão e preparação da entrega

## Objetivo

Confirmar que todos os critérios do trabalho foram atendidos e deixar o
ambiente pronto para a gravação. Esta etapa não cria um novo modelo: ela valida
o projeto construído nas etapas anteriores.

## O que foi concluído

| Critério | Evidência no projeto |
|---|---|
| Dataset e preparação | `data/telco_churn.csv`, `src/preprocessing.py` e documentos das Etapas 1 e 2 |
| Implementação do MLP | `src/training.py` |
| Cinco configurações | `EXPERIMENT_CONFIGS` em `src/training.py` |
| Uso do MLflow | `src/train_experiments.py`, `mlflow.db` e `mlartifacts` gerados localmente |
| Comparação e seleção | `src/select_best_model.py` e `config/best_model.json` gerado localmente |
| Modelo como artifact | pasta `model` dentro do run MLP-05 no MLflow |
| API de predição | `src/api.py` e `examples/predict_churn.json` |
| Testes | diretório `tests` com 16 testes |
| Documentação e vídeo | `README.md` e `docs/roteiro_video.md` |

## Validação final no computador

Execute a partir da raiz do projeto.

### 1. Conferir os arquivos gerados

```powershell
Test-Path .\mlflow.db
Test-Path .\mlartifacts
Test-Path .\config\best_model.json
```

Os três resultados devem ser `True`.

### 2. Executar os testes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

O final da saída deve ser:

```text
Ran 16 tests
OK
```

### 3. Abrir o MLflow

No primeiro terminal:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5001
```

Abrir `http://127.0.0.1:5001` e confirmar:

- experimento `telco-churn-mlp`;
- cinco runs com status `Finished`;
- parâmetros e métricas dos runs;
- gráfico de comparação;
- run MLP-05 com as pastas `evaluation` e `model` em Artifacts;
- tag `selected_as_best: true` no MLP-05.

### 4. Abrir a API

No segundo terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000/docs` e confirmar:

- `GET /health` responde com código 200, `model_loaded: true` e `MLP-05`;
- `POST /predict` responde com código 200;
- a resposta contém `prediction`, `label`, `probability` e `model_run`.

## Preparação antes de iniciar a gravação

Não grave instalação de bibliotecas nem treinamento: essas operações consomem
tempo e os resultados já estão salvos.

1. Feche janelas e notificações que possam exibir dados pessoais.
2. Deixe o MLflow e a API iniciados em dois terminais separados.
3. Abra previamente estas páginas no navegador:
   - lista dos cinco runs no MLflow;
   - comparação por coordenadas paralelas;
   - detalhes e Artifacts do MLP-05;
   - Swagger da API em `/docs`.
4. Deixe `POST /predict` aberto com o JSON de exemplo.
5. Faça uma gravação curta de teste para conferir áudio e compartilhamento da
   tela.
6. Use o roteiro em `docs/roteiro_video.md` e mantenha a duração próxima de
   4min40s, deixando margem até o limite de cinco minutos.

## Checklist dos entregáveis

- [ ] Repositório com `src`, `tests`, `data`, `docs` e `examples`.
- [ ] `requirements.txt` presente.
- [ ] README com comandos completos.
- [ ] Cinco runs visíveis no MLflow.
- [ ] Parâmetros, métricas e artifacts visíveis.
- [ ] MLP-05 selecionado e recuperável pela URI do MLflow.
- [ ] API local funcionando.
- [ ] Print ou demonstração do gráfico de comparação.
- [ ] Print ou demonstração do `POST /predict` com código 200.
- [ ] Vídeo gravado por apenas uma pessoa.
- [ ] Vídeo com duração inferior a cinco minutos.
- [ ] Link do repositório e vídeo conferidos antes do envio.

## Arquivos locais que não devem ser apagados antes do vídeo

- `mlflow.db`;
- `mlartifacts`;
- `config/best_model.json`;
- `.venv`.

Esses arquivos são ignorados pelo Git porque são específicos da execução
local. O código do repositório permite gerá-los novamente seguindo o README.
