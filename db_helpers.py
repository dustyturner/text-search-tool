import sqlite3
from tqdm import tqdm

INSERT = "INSERT INTO {table} VALUES ({question_marks})"
SELECT = "SELECT {columns} from {table}"

def insert_into_db_table(values, table, db_filepath):
  conn = sqlite3.connect(db_filepath)
  c = conn.cursor()
  assert type(values) == list
  print("=" * 100)
  print(f"uploading {len(values)} entries to {table} table in {db_filepath}")
  for value in tqdm(values, total=len(values)):
    question_marks = ", ".join(["?" for i in range(len(value))])
    c.execute(
        INSERT.format(table=table, 
                      question_marks=question_marks),
        value)
  conn.commit()
  conn.close()
  print("upload complete!")
  return 0
      
def select_from_db_table(columns, table, db_filepath):
  conn = sqlite3.connect(db_filepath)
  c = conn.cursor()
  assert type(columns) == list
  columns = ", ".join(columns)
  output = c.execute(
      SELECT.format(columns=columns,
                    table=table)).fetchall()
  conn.close()
  return output

def create_sentences_table(db_filepath):
  conn = sqlite3.connect(db_filepath)
  c = conn.cursor()
  embedding_columns = ""
  for i in range(768):
    embedding_columns += "embedding_" + str(i) + ", "

  sql_create = ("""
               CREATE TABLE sentences(
               sentence_no INTEGER NOT NULL, 
               text_id TEXT NOT NULL,
               sentence text, 
               """
               + embedding_columns
               + "PRIMARY KEY (sentence_no, text_id));")
  c.execute(sql_create)
  conn.commit()
  conn.close()
  return 0