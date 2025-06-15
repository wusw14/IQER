from utils import sample_table, parse_response
from appl import ppl, gen, SystemMessage, convo, records, SystemRole
from appl.compositor import Tagged, NumberedList, DashList


@ppl
def gen_cond2col(query: str, table: str, col_vals: dict):
    with SystemRole():
        "You are a data scientist expert in SQL and database analysis. Your task is to analyze natural language queries against database schemas to identify which table columns are relevant for each query condition."

    "## Task Definition"
    "Given:"
    with DashList(indent=2):
        "A natural language query containing filtering conditions"
        "A table schema with column names and sample values"
    "Output: A mapping of each query condition to its most relevant table column"

    "## Output Format"
    "Return a dictionary where:"
    with DashList(indent=2):
        "Keys are the extracted conditions (in SQL-like syntax when possible)"
        "Values are the corresponding column names."

    "## Instructions"
    with NumberedList():
        "Parse the natural language query to identify all filtering conditions"
        "For each condition:"
        with DashList(indent=2):
            "Determine if it can be expressed in SQL syntax (e.g., comparisons, ranges). If not, keep the natural language expression."
            "Match the condition to the most semantically relevant column. Consider column names and sample values when matching."
            "If no exact column matches: Make the most reasonable inference based on available information. Note any assumptions in your <thought> explanation"
            "The condition must contain a column name."

    "### Examples"
    "Query: Find all employees in the CSE with a salary greater than 50000."
    "Table: employee"
    "id, sample_values: [1, 2, 3, 4, 5]"
    "name, sample_values: ['Alice', 'Bob', 'Charlie', 'David', 'Eve']"
    "department, sample_values: ['CSE', 'ECE', 'ME', 'CE', 'IDEA']"
    "salary, sample_values: [60000, 70000, 80000, 90000, 100000]"
    "Output:"
    "<thought>The query has two clear conditions: department='CSE' and salary>50000, which directly match columns.</thought>"
    "<answer>{'department=\"CSE\"': 'department', 'salary>50000': 'salary'}</answer>"

    "Query: Find all employees in the School of Engineering with a salary greater than 50000."
    "Table: employee"
    "id, sample_values: [1, 2, 3, 4, 5]"
    "name, sample_values: ['Alice', 'Bob', 'Charlie', 'David', 'Eve']"
    "department, sample_values: ['CSE', 'ECE', 'ME', 'CE', 'IDEA']"
    "salary, sample_values: [60000, 70000, 80000, 90000, 100000]"
    "Output:"
    "<thought>'School of Engineering' likely refers to multiple departments (CSE, ECE, etc.), so we map to department column.</thought>"
    "<answer>{'department in School of Engineering': 'department', 'salary>50000': 'salary'}</answer>"

    "## Your Task"
    "Analyze the following query and table:"
    "Query: " + query
    f"Table: {table}"
    for col, vals in col_vals.items():
        f"{col}, sample_values: {vals[:5]}"
    "Output:"
    "<thought>Analyze the query conditions and explain your column matching rationale.</thought>"
    "<answer>Provide your condition-to-column mapping dictionary here.</answer>"

    return gen()


def get_related_columns(query: str, table: str, col_vals: dict) -> dict:
    """
    Get the related columns from the query and column values.
    Args:
        query (str): The query string.
        col_vals (dict): A dictionary mapping column names to sample values.
    Returns:
        dict: A dictionary mapping conditions to columns and values.
    """
    ans = gen_cond2col(query, table, col_vals)
    ans = parse_response(ans, "answer")
    cond2col = eval(ans)
    return cond2col


@ppl
def gen_sql(query: str, db: str, table: str, col_vals: dict):
    SystemMessage(
        "You are an expert in parsing natural language queries to SQL queries."
    )

    """
    You are tasked with converting a natural language query into an SQL query. 
    
    The input will include the following:
    Query: The user's natural language query.
    Table: The name of the table where the data resides.
    Columns: The list of column names in the table along with sample values for each column.

    SQL Construction Rules:
    1. Standard SQL Operators (Preferred): Use standard SQL operators (=, >, <, BETWEEN, IN, etc.) for numerical attributes.
    2. User-Defined Function sem_filter(statement) for semantic inference:
    - Use sem_filter(statement) for textual attributes.
    - The statement should be a declarative sentence that describes the filtering condition, and must include at least one column name.
    - The column name should be enclosed in curly braces, and the column name must be in the table.
    - Example: If the query asks to find all records in "Southern California," but the table only has a county column, you can use sem_filter({county} is in Southern California) to infer whether a row belongs to Southern California.
    
    Output Format:
    <thought>Analyze the query conditions and explain your SQL construction rationale.</thought>
    <answer>Provide your SQL query here.</answer>
    <where>filter conditions in the where clause separated by ' | '.</where>
    """

    "Example Inputs and Outputs"
    # "Example 1:"
    # "Query: Find all employees in the CSE with a salary greater than 50000."
    # "Table: employee"
    # "id, sample_values: [1, 2, 3, 4, 5]"
    # "name, sample_values: ['Alice', 'Bob', 'Charlie', 'David', 'Eve']"
    # "department, sample_values: ['CSE', 'ECE', 'ME', 'CE', 'IDEA']"
    # "salary, sample_values: [60000, 70000, 80000, 90000, 100000]"
    # "Output:"
    # "<thought>The query has two clear conditions: department='CSE' and salary>50000, which directly match columns.</thought>"
    # "<answer>SELECT * FROM employee WHERE department='CSE' AND salary>50000;</answer>"
    # "<where>department='CSE' | salary>50000</where>"

    # "Example 2:"
    "Query: Find all employees in the School of Engineering with a salary greater than 50000."
    "Table: employee"
    "id, sample_values: [1, 2, 3, 4, 5]"
    "name, sample_values: ['Alice', 'Bob', 'Charlie', 'David', 'Eve']"
    "department, sample_values: ['CSE', 'ECE', 'ME', 'CE', 'IDEA']"
    "salary, sample_values: [60000, 70000, 80000, 90000, 100000]"
    "Output:"
    "<thought>'School of Engineering' likely refers to multiple departments (CSE, ECE, etc.), so we map to department column.</thought>"
    "<answer>SELECT * FROM employee WHERE sem_filter({department} is in School of Engineering) AND salary>50000;</answer>"
    "<where>sem_filter({department} is in School of Engineering) | salary>50000</where>"

    "You need to convert the following query into SQL. Your output should follow the format above."
    "Query: " + query
    "Table: " + table
    for col, vals in col_vals.items():
        f"{col}, sample_values: {vals[:5]}"
    "Output:"
    "<thought>Analyze the query conditions and explain your SQL construction rationale.</thought>"
    "<answer>Provide your SQL query here.</answer>"
    "<where>Provide your where clause here.</where>"
    return gen()


def parse_query(
    query: str, db: str, table: str, col_vals: dict
) -> tuple[str, list[str]]:
    """
    Parse the query and table to extract conditions and columns.
    Args:
        query (str): The query string.
        table (str): The table string.
        col_vals (dict): A dictionary mapping column names to sample values.
    Returns:
        dict: A dictionary mapping conditions to columns.
    """
    # # parse the query to extract conditions and map them to columns
    # cond2col = get_related_columns(query, table, col_vals)
    # return cond2col
    # parse the query to the corrersponding SQL
    ans = gen_sql(query, db, table, col_vals)
    ans = str(ans)
    sql = parse_response(ans, "answer")
    where_clause = parse_response(ans, "where")
    conditions = where_clause.split(" | ")
    conditions = [cond.strip() for cond in conditions]
    return sql, conditions


def parse_conds(cols: list[str], conds: str) -> list[str]:
    conds = conds.split(" AND ")
    sql_conds = []
    sem_filter_conds = []
    for cond in conds:
        if "sem_filter" in cond:
            sem_filter_conds.append(cond)
        else:
            for col in cols:
                if col in cond:
                    sql_conds.append(cond)
                    break
    return sql_conds, sem_filter_conds
