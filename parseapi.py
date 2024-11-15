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