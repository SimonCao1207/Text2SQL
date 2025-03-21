### Task
Determine if the following user question can be answered using only the provided database schema: [QUESTION]{user_question}[/QUESTION]

### Instructions
- If you can generate a valid SQL query for the user question from the schema, return "YES".
- If the schema does not support answering the question, return "NO".

### Database Schema
{table_metadata_string}

### User Question
[QUESTION]{user_question}[/QUESTION]
 