# AI612-Project-1

# Set up

- Install `Ruff` extension on vscode for Python linter and code formatter

- Create `.env` file storing API_KEY. For example:
```
OPENAI_API_KEY=<your_api_key>
```
- Install dependencies 
```
pip install -r requirements.txt
```

## Baseline
```
python baseline.py
```

- Finetune basemodel gpt-4o mini on EHRSQL train dataset with `llm_finetune.ipynb`

## Evaluation
- Run evaluation on valid dataset: 
```
python evaluate.py
```
