import copy
import operator

class Table:
    def __init__(self, schema, rows=None):
        self.schema = schema
        self.rows = rows or []

    def validate_row(self, row):
        if len(row) != len(self.schema):
            raise ValueError(f"Row length {len(row)} does not match schema length {len(self.schema)}.")

    def add_row(self, row):
        self.validate_row(row)
        self.rows.append(row)

    def add_rows(self, rows):
        for row in rows:
            self.add_row(row)

    def get_schema(self):
        return self.schema

    def get_rows(self):
        return self.rows

    def copy(self):

        return Table(copy.deepcopy(self.schema), copy.deepcopy(self.rows))

    def __str__(self):
        schema_str = ", ".join(self.schema)
        rows_str = "\n".join([", ".join(map(str, row)) for row in self.rows])
        return f"Schema: [{schema_str}]\nRows:\n{rows_str}"

OPERATORS = {
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge
}

def inner_join(left_table, right_table, conditions):
    """
    Perform an inner join between two tables with multiple join conditions.
    Logs a message if a column is missing.
    :param left_table: The left Table object.
    :param right_table: The right Table object.
    :param conditions: List of conditions as tuples (left_col, operator, right_col).
    :return: A new Table containing the result of the inner join.
    """
    # Prepare new schema
    new_schema = left_table.schema + [col for col in right_table.schema if col not in left_table.schema]

    # Validate columns in conditions
    for left_col, op, right_col in conditions:
        if left_col not in left_table.schema:
            logging.error(f"Column '{left_col}' not found in left table schema.")
            return None  # Return None or an empty Table
        if right_col not in right_table.schema:
            logging.error(f"Column '{right_col}' not found in right table schema.")
            return None  # Return None or an empty Table
        if op not in OPERATORS:
            logging.error(f"Unsupported operator: {op}")
            return None  # Return None or an empty Table

    # Perform the join
    new_rows = []
    for left_row in left_table.rows:
        for right_row in right_table.rows:
            # Evaluate all conditions
            matches = all(
                OPERATORS[op](
                    left_row[left_table.schema.index(left_col)],
                    right_row[right_table.schema.index(right_col)]
                )
                for left_col, op, right_col in conditions
            )
            if matches:
                # Combine rows, avoiding duplicate columns from the right table
                combined_row = left_row + [val for col, val in zip(right_table.schema, right_row) if col not in left_table.schema]
                new_rows.append(combined_row)

    return Table(new_schema, new_rows)


def left_outer_join(left_table, right_table, conditions):
    """
    Perform a left outer join between two tables with multiple join conditions.
    Logs a message if a column is missing.
    :param left_table: The left Table object.
    :param right_table: The right Table object.
    :param conditions: List of conditions as tuples (left_col, operator, right_col).
    :return: A new Table containing the result of the left outer join.
    """
    # Prepare new schema
    new_schema = left_table.schema + [col for col in right_table.schema if col not in left_table.schema]

    # Validate columns in conditions
    for left_col, op, right_col in conditions:
        if left_col not in left_table.schema:
            logging.error(f"Column '{left_col}' not found in left table schema.")
            return None  # Return None or an empty Table
        if right_col not in right_table.schema:
            logging.error(f"Column '{right_col}' not found in right table schema.")
            return None  # Return None or an empty Table
        if op not in OPERATORS:
            logging.error(f"Unsupported operator: {op}")
            return None  # Return None or an empty Table

    # Perform the join
    new_rows = []
    for left_row in left_table.rows:
        matched = False
        for right_row in right_table.rows:
            # Evaluate all conditions
            matches = all(
                OPERATORS[op](
                    left_row[left_table.schema.index(left_col)],
                    right_row[right_table.schema.index(right_col)]
                )
                for left_col, op, right_col in conditions
            )
            if matches:
                matched = True
                # Combine rows, avoiding duplicate columns from the right table
                combined_row = left_row + [val for col, val in zip(right_table.schema, right_row) if col not in left_table.schema]
                new_rows.append(combined_row)
        if not matched:
            # Append left_row with None for unmatched rows
            combined_row = left_row + [None] * (len(new_schema) - len(left_table.schema))
            new_rows.append(combined_row)

    return Table(new_schema, new_rows)

def full_outer_join(left_table, right_table, conditions):
    """
    Perform a full outer join between two tables with multiple join conditions.
    Logs a message if a column is missing.
    :param left_table: The left Table object.
    :param right_table: The right Table object.
    :param conditions: List of conditions as tuples (left_col, operator, right_col).
    :return: A new Table containing the result of the full outer join.
    """
    # Prepare new schema
    new_schema = left_table.schema + [col for col in right_table.schema if col not in left_table.schema]

    # Validate columns in conditions
    for left_col, op, right_col in conditions:
        if left_col not in left_table.schema:
            logging.error(f"Column '{left_col}' not found in left table schema.")
            return None
        if right_col not in right_table.schema:
            logging.error(f"Column '{right_col}' not found in right table schema.")
            return None
        if op not in OPERATORS:
            logging.error(f"Unsupported operator: {op}")
            return None

    # Track matched rows
    matched_left_rows = set()
    matched_right_rows = set()

    # Perform the join
    new_rows = []
    for left_idx, left_row in enumerate(left_table.rows):
        for right_idx, right_row in enumerate(right_table.rows):
            # Evaluate all conditions
            matches = all(
                OPERATORS[op](
                    left_row[left_table.schema.index(left_col)],
                    right_row[right_table.schema.index(right_col)]
                )
                for left_col, op, right_col in conditions
            )
            if matches:
                matched_left_rows.add(left_idx)
                matched_right_rows.add(right_idx)
                # Combine rows, avoiding duplicate columns from the right table
                combined_row = left_row + [val for col, val in zip(right_table.schema, right_row) if col not in left_table.schema]
                new_rows.append(combined_row)

    # Add unmatched rows from the left table
    for left_idx, left_row in enumerate(left_table.rows):
        if left_idx not in matched_left_rows:
            combined_row = left_row + [None] * (len(new_schema) - len(left_table.schema))
            new_rows.append(combined_row)

    # Add unmatched rows from the right table
    for right_idx, right_row in enumerate(right_table.rows):
        if right_idx not in matched_right_rows:
            combined_row = [None] * len(left_table.schema) + [val for val in right_row]
            new_rows.append(combined_row)

    return Table(new_schema, new_rows)

def unpivot(
    table,
    ungroup_by,
    key_column_name,
    unpivot_columns,
    drop_rows_with_null=False,
    unpivot_type="Normal",
    pick_column_names_as_values=True,
    manual_values=None
):
    """
    Perform an unpivot operation with options for normal or lateral unpivoting, dropping rows with null, 
    and specifying unpivoted column names dynamically or manually.
    :param table: Input Table object.
    :param ungroup_by: List of column names to remain unchanged.
    :param key_column_name: Name of the new column to store unpivoted column names or manual values.
    :param unpivot_columns: List of column names to unpivot.
    :param drop_rows_with_null: Whether to drop rows where unpivoted values are None.
    :param unpivot_type: Either "Normal" or "Lateral".
    :param pick_column_names_as_values: If True, uses column names as values in the key column.
    :param manual_values: List of tuples (value, allow_null) to manually specify values and their null handling.
    :return: A new Table containing the result of the unpivot operation.
    """
    # Validate input
    for col in ungroup_by + unpivot_columns:
        if col not in table.schema:
            logging.error(f"Column '{col}' not found in table schema.")
            return None

    if unpivot_type not in ["Normal", "Lateral"]:
        logging.error(f"Unsupported unpivot type: {unpivot_type}. Must be 'Normal' or 'Lateral'.")
        return None

    if not pick_column_names_as_values and not manual_values:
        logging.error("When 'Pick Column Names as Values' is False, 'manual_values' must be provided.")
        return None

    # Prepare new schema
    new_schema = ungroup_by + [key_column_name, "Value"]

    # Determine key values and null-handling rules
    key_values = []
    if pick_column_names_as_values:
        key_values = [(col, True) for col in unpivot_columns]
    else:
        key_values = manual_values

    # Perform the unpivot operation
    new_rows = []
    for row in table.rows:
        ungrouped_values = [row[table.schema.index(col)] for col in ungroup_by]
        for idx, (key_value, allow_null) in enumerate(key_values):
            col = unpivot_columns[idx]
            value = row[table.schema.index(col)]
            if not allow_null and value is None:
                continue  # Skip this row for the current column if null is not allowed
            if drop_rows_with_null and value is None:
                continue  # Skip rows entirely if drop_rows_with_null is True
            if unpivot_type == "Lateral":
                # In Lateral, create multiple rows for the same unpivoted column
                for other_value in [v for v, _ in key_values]:
                    new_row = ungrouped_values + [other_value, value]
                    new_rows.append(new_row)
            else:
                # In Normal, create a single row per unpivoted column
                new_row = ungrouped_values + [key_value, value]
                new_rows.append(new_row)

    return Table(new_schema, new_rows)


def aggregate(table, group_by_columns, aggregates):
    """
    Perform an aggregation on a table.
    :param table: Input Table object.
    :param group_by_columns: List of column names to group by.
    :param aggregates: Dictionary where keys are output column names and values are functions or expressions 
                       (e.g., {"Total_Sales": ("Sales", sum)}).
    :return: A new Table containing the aggregated results.
    """
    # Validate group_by_columns
    for col in group_by_columns:
        if col not in table.schema:
            logging.error(f"Group by column '{col}' not found in the table schema.")
            return None

    # Validate aggregate columns
    for agg_name, (agg_col, _) in aggregates.items():
        if agg_col not in table.schema:
            logging.error(f"Aggregate column '{agg_col}' not found in the table schema.")
            return None

    # Create a dictionary to hold grouped data
    grouped_data = {}
    group_indices = [table.schema.index(col) for col in group_by_columns]

    for row in table.rows:
        # Create a tuple key for the group
        group_key = tuple(row[idx] for idx in group_indices)

        # Initialize the group if not already present
        if group_key not in grouped_data:
            grouped_data[group_key] = {agg_name: [] for agg_name in aggregates.keys()}

        # Append values to aggregate lists
        for agg_name, (agg_col, _) in aggregates.items():
            grouped_data[group_key][agg_name].append(row[table.schema.index(agg_col)])

    # Perform aggregation computations
    result_rows = []
    for group_key, agg_data in grouped_data.items():
        # Compute aggregated values
        agg_results = []
        for agg_name, (agg_col, agg_func) in aggregates.items():
            agg_results.append(agg_func(agg_data[agg_name]))

        # Combine group key and aggregated values
        result_rows.append(list(group_key) + agg_results)

    # Create the result schema
    result_schema = group_by_columns + list(aggregates.keys())

    return Table(result_schema, result_rows)

def full_outer_join(table1, table2, conditions):
    """
    Perform a full outer join between two tables with dictionary-based rows.
    :param table1: The first Table object.
    :param table2: The second Table object.
    :param conditions: List of conditions in the format ("column1", "operator", "column2").
    :return: A new Table containing the result of the full outer join.
    """
    # Extract column names from schemas
    table1_columns = [col["name"] for col in table1.schema]
    table2_columns = [col["name"] for col in table2.schema]

    # Merge schemas
    merged_schema = table1.schema + [col for col in table2.schema if col["name"] not in table1_columns]

    # Track matched rows in table2
    matched_table2_rows = set()

    # Perform the join
    joined_rows = []
    for row1 in table1.rows:
        matched = False
        for j, row2 in enumerate(table2.rows):
            # Check if the rows satisfy the join conditions
            if all(compare(row1.get(cond[0]), cond[1], row2.get(cond[2])) for cond in conditions):
                # Merge matching rows
                merged_row = {**row1, **{col: row2.get(col, None) for col in table2_columns}}
                joined_rows.append(merged_row)
                matched = True
                matched_table2_rows.add(j)  # Mark row2 as matched
        if not matched:
            # Add unmatched row1 with NULLs for table2 columns
            unmatched_row = {col: None for col in table2_columns}
            joined_rows.append({**row1, **unmatched_row})

    # Add unmatched rows from table2
    for j, row2 in enumerate(table2.rows):
        if j not in matched_table2_rows:
            unmatched_row = {col: None for col in table1_columns}
            joined_rows.append({**unmatched_row, **row2})

    return Table(merged_schema, joined_rows)

def cross_join(table1, table2, condition=None):
    """
    Perform a cross join between two tables with an optional condition.
    :param table1: The first Table object.
    :param table2: The second Table object.
    :param condition: A lambda function that takes two rows and returns True if the row pair satisfies the condition.
                      Example: lambda row1, row2: row1[0] == row2[0]  # Join on ID
    :return: A new Table containing the result of the cross join.
    """
    # Validate input
    if not table1.rows or not table2.rows:
        logging.error("One or both tables have no rows.")
        return None

    # Combine schemas
    new_schema = table1.schema + table2.schema

    # Perform cross join
    new_rows = []
    for row1 in table1.rows:
        for row2 in table2.rows:
            if condition is None or condition(row1, row2):
                new_rows.append(row1 + row2)

    return Table(new_schema, new_rows)

def aggregate(table, group_by_columns, custom_expressions):
    """
    Perform aggregation with custom expressions on a table.
    :param table: The Table object.
    :param group_by_columns: List of columns to group by.
    :param custom_expressions: List of tuples in the format ("result_column", "expression").
                               Expressions can use Python syntax and reference columns.
    :return: A new Table containing the aggregated result.
    """
    from collections import defaultdict

    # Helper functions for custom expressions
    def countRows(column):
        """Count non-null rows in a column."""
        return len(column)

    def uniqueValues(column):
        """Count unique values in a column."""
        return len(set(column))

    # Validate group_by_columns
    schema_column_names = [col["name"] for col in table.schema]
    for col in group_by_columns:
        if col not in schema_column_names:
            raise ValueError(f"Group by column '{col}' not found in schema.")

    # Group rows by the specified columns
    grouped_data = defaultdict(list)
    for row in table.rows:
        key = tuple(row[col] for col in group_by_columns)
        grouped_data[key].append(row)

    # Perform aggregation
    result_rows = []
    for key, rows in grouped_data.items():
        # Initialize the aggregated row with group-by columns
        aggregated_row = {col: val for col, val in zip(group_by_columns, key)}

        # Add custom expression results
        for result_column, expression in custom_expressions:
            try:
                # Prepare evaluation context
                eval_context = {
                    col: [row[col] for row in rows if col in row and row[col] is not None]
                    for col in schema_column_names
                }
                eval_context.update({"countRows": countRows, "uniqueValues": uniqueValues})
                
                # Evaluate the custom expression
                aggregated_row[result_column] = eval(expression, {}, eval_context)
            except Exception as e:
                aggregated_row[result_column] = None  # Handle errors gracefully
                print(f"Error evaluating expression '{expression}' for '{result_column}': {e}")

        result_rows.append(aggregated_row)

    # Create result schema
    result_schema = [
        {"name": col, "desc": f"Grouped by {col}", "type": "grouping"} for col in group_by_columns
    ] + [
        {"name": col, "desc": f"Custom expression: {expr}", "type": "aggregate"}
        for col, expr in custom_expressions
    ]

    return Table(result_schema, result_rows)

custom_expressions = [
    ("Total_Sales", "sum(Sales)"),
    ("Average_Sales", "sum(Sales) / len(Sales)"),
    ("Color_Count", "len(set(Color))")  # Count distinct colors
]

table = Table(
    schema=[
        {"name": "ID", "desc": "Identifier", "type": "int"},
        {"name": "Name", "desc": "Item name", "type": "string"},
        {"name": "Sales", "desc": "Sales count", "type": "int"},
        {"name": "Color", "desc": "Item color", "type": "string"}
    ],
    rows=[
        {"ID": 1, "Name": "Apple", "Sales": 50, "Color": "Red"},
        {"ID": 2, "Name": "Banana", "Sales": 40, "Color": "Yellow"},
        {"ID": 1, "Name": "Apple", "Sales": 30, "Color": "Red"},
        {"ID": 3, "Name": "Cherry", "Sales": 20, "Color": "Red"},
        {"ID": 2, "Name": "Banana", "Sales": 60, "Color": "Yellow"}
    ]
)

result = aggregate(
    table,
    group_by_columns=["ID", "Name"],
    custom_expressions=custom_expressions
)

# Print Results
print("Aggregated Result Schema:")
print(result.schema)

print("\nAggregated Result Rows:")
for row in result.rows:
    print(row)


    [
    {"name": "ID", "desc": "Grouped by ID", "type": "grouping"},
    {"name": "Name", "desc": "Grouped by Name", "type": "grouping"},
    {"name": "Total_Sales", "desc": "Custom expression: sum(Sales)", "type": "aggregate"},
    {"name": "Average_Sales", "desc": "Custom expression: sum(Sales) / len(Sales)", "type": "aggregate"},
    {"name": "Color_Count", "desc": "Custom expression: len(set(Color))", "type": "aggregate"}
]
    

    [
    {"ID": 1, "Name": "Apple", "Total_Sales": 80, "Average_Sales": 40.0, "Color_Count": 1},
    {"ID": 2, "Name": "Banana", "Total_Sales": 100, "Average_Sales": 50.0, "Color_Count": 1},
    {"ID": 3, "Name": "Cherry", "Total_Sales": 20, "Average_Sales": 20.0, "Color_Count": 1}
]