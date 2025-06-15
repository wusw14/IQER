from utils import sample_table, parse_response
from appl import ppl, gen, SystemMessage, convo, records, SystemRole
from appl.compositor import Tagged, NumberedList, DashList


@ppl
def gen_condition(cond: str, col: str, value: list):
    SystemMessage("You are a data scientist proficient in databases and SQL.")

    "Task: Given a filter condition expressed in natural language and sample values from the column it applies to, determine whether the condition can be translated into a SQL WHERE clause using relational algebra (e.g., comparisons, pattern matching, logical operators) without requiring semantic reasoning (e.g., external knowledge or complex interpretation of values). If possible, generate the corresponding SQL WHERE condition."

    "### Examples"
    "Condition: Age is greater than 30 but less than 50"
    "Column: age"
    "Sample Values: [25, 30, 35, 40, 45, 50, 55]"
    "Output:"
    "<thought>The condition 'Age is greater than 30 but less than 50' can be expressed as a SQL WHERE clause using relational algebra. The corresponding SQL WHERE condition would be: age > 30 AND age < 50.</thought>"
    "<feasiblity>Yes</feasiblity>"
    "<sql_where_condition>age > 30 AND age < 50</sql_where_condition>"

    "\nCondition: Products suitable for children"
    "Column: product_title"
    "Sample Values: ['women face wash', 'baby face cream', '3-6 year face lotion', 'women face mask']"
    "Output:"
    "<thought>The condition 'Products suitable for children' cannot be expressed as a SQL WHERE clause using relational algebra. The condition requires semantic reasoning to determine suitability for children.</thought>"
    "<feasiblity>No</feasiblity>"
    "<sql_where_condition>None</sql_where_condition>"

    "\nPlease analyze the following condition and provide the output in the same format as the examples."
    "Condition: " + cond
    "Column: " + col
    f"Sample Values: {value}"
    "Output:"
    "<thought>Analyze the condition and determine if it can be expressed as a SQL WHERE clause using relational algebra.</thought>"
    "<feasiblity>Yes/No</feasiblity>"
    "<sql_where_condition>Provide the SQL WHERE condition if feasible; otherwise, None.</sql_where_condition>"

    return gen()


def transform_to_sql(cond: str, col: str, value: list) -> str:
    """
    Check if the value in the condition is aligned with the sample values in the column.
    Alignment means that the filter condition could be expressed with relational algebra.

    Args:
        cond (str): The condition to check.
        col (str): The column name.
        value (list): The list of values.

    Returns:
        bool: True if the condition is aligned with the column, False otherwise.
    """
    ans = gen_condition(cond, col, value)
    ans = str(ans)
    feasiblity = parse_response(ans, "feasiblity")
    if feasiblity.lower() == "yes":
        sql_where_condition = parse_response(ans, "sql_where_condition")
        if sql_where_condition.lower() != "none":
            return sql_where_condition
    return None
