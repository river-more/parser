import parseapi


def tests():

    ## join INNER
    left_table = parseapi.Table(["ID", "Name", "Price"], [
        [1, "Apple", 5.0],
        [2, "Banana", 3.0],
        [3, "Cherry", 7.0]
    ])

    right_table = parseapi.Table(["ID", "Color", "Count"], [
        [1, "Red", 10],
        [2, "Yellow", 15],
        [3, "Red", 8],
        [4, "Green", 5]  
    ])

    conditions = [
        ('ID', '==', 'ID'),
        #('Color', '==', 'Color') 
    ]

    joined_table = parseapi.inner_join(left_table, right_table, conditions)

    print("Joined Table:")
    print(joined_table)



#def test_unpivot_normal_with_column_names():
    print("\nTest Case: Normal Unpivot with Column Names as Values")
    input_table = parseapi.Table(["ID", "Name", "Jan_Sales", "Feb_Sales", "Mar_Sales"], [
        [1, "Apple", 50, 60, 55],
        [2, "Banana", 40, None, 45],
        [3, "Cherry", None, 35, None],
        [4, "Grape", None, None, None]
    ])

    # Perform unpivot
    unpivoted_table = parseapi.unpivot(
        table=input_table,
        ungroup_by=["ID", "Name"],
        key_column_name="Month",
        unpivot_columns=["Jan_Sales", "Feb_Sales", "Mar_Sales"],
        drop_rows_with_null=True,  # Drop rows with null values
        unpivot_type="Normal",
        pick_column_names_as_values=True
    )

    print(unpivoted_table)






#def test_unpivot_lateral_with_manual_values():
    print("\nTest Case: Lateral Unpivot with Manual Values")
    input_table = parseapi.Table(["ID", "Name", "Jan_Sales", "Feb_Sales", "Mar_Sales"], [
        [1, "Apple", 50, 60, 55],
        [2, "Banana", 40, None, 45],
        [3, "Cherry", None, 35, None],
        [4, "Grape", None, None, None]
    ])

    # Manual values with null-handling rules
    manual_values = [("January", True), ("February", False), ("March", True)]

    # Perform unpivot
    unpivoted_table = parseapi.unpivot(
        table=input_table,
        ungroup_by=["ID", "Name"],
        key_column_name="Month",
        unpivot_columns=["Jan_Sales", "Feb_Sales", "Mar_Sales"],
        drop_rows_with_null=False,  # Do not drop rows with null values globally
        unpivot_type="Lateral",
        pick_column_names_as_values=False,  # Use manual values instead of column names
        manual_values=manual_values
    )

    print(unpivoted_table)


#def test_unpivot_drop_nulls_only():
    print("\nTest Case: Normal Unpivot, Drop Null Rows Only")
    input_table = parseapi.Table(["ID", "Name", "Jan_Sales", "Feb_Sales", "Mar_Sales"], [
        [1, "Apple", 50, 60, 55],
        [2, "Banana", 40, None, 45],
        [3, "Cherry", None, 35, None],
        [4, "Grape", None, None, None]
    ])

    # Perform unpivot
    unpivoted_table = parseapi.unpivot(
        table=input_table,
        ungroup_by=["ID", "Name"],
        key_column_name="Month",
        unpivot_columns=["Jan_Sales", "Feb_Sales", "Mar_Sales"],
        drop_rows_with_null=True,  # Drop null rows
        unpivot_type="Normal",
        pick_column_names_as_values=True
    )

    print(unpivoted_table)


#def test_unpivot_without_null_removal():
    print("\nTest Case: Normal Unpivot, Keep Null Rows")
    input_table = parseapi.Table(["ID", "Name", "Jan_Sales", "Feb_Sales", "Mar_Sales"], [
        [1, "Apple", 50, 60, 55],
        [2, "Banana", 40, None, 45],
        [3, "Cherry", None, 35, None],
        [4, "Grape", None, None, None]
    ])

    # Perform unpivot
    unpivoted_table = parseapi.unpivot(
        table=input_table,
        ungroup_by=["ID", "Name"],
        key_column_name="Month",
        unpivot_columns=["Jan_Sales", "Feb_Sales", "Mar_Sales"],
        drop_rows_with_null=False,  # Keep null rows
        unpivot_type="Normal",
        pick_column_names_as_values=True
    )

    print(unpivoted_table)

    #JOINNN

    table_a = parseapi.Table(["ID", "Name", "Sales"], [
        [1, "Apple", 50],
        [2, "Banana", 40]
    ])


    table_b = parseapi.Table(["ID", "Color", "Count"], [
        [1, "Red", 10],
        [3, "Green", 20]
    ])


    #def test_inner_join(): ##WORKS
    print("\nTest Case: Inner Join")
    joined_table = parseapi.inner_join(
        table_a,
        table_b,
        conditions=[("ID", "==", "ID")]
    )
    print(joined_table)


    #def test_left_outer_join():
    print("\nTest Case: Left Outer Join")
    joined_table = parseapi.left_outer_join(
        table_a,
        table_b,
        conditions=[("ID", "==", "ID")]
    )
    print(joined_table)


    #def test_full_outer_join():
    print("\nTest Case: Full Outer Join")
    joined_table = parseapi.full_outer_join(
        table_a,
        table_b,
        conditions=[("ID", "==", "ID")]
    )
    print(joined_table)

    input_table = parseapi.Table(["ID", "Name", "Sales", "Category"], [
        [1, "Apple", 50, "Fruit"],
        [2, "Banana", 40, "Fruit"],
        [3, "Carrot", 20, "Vegetable"],
        [4, "Potato", 30, "Vegetable"],
        [1, "Apple", 60, "Fruit"]
    ])





    #def test_sum_aggregation():
    print("\nTest Case: Sum Aggregation")
    group_by_columns = ["Category"]
    aggregates = {"Total_Sales": ("Sales", sum)}
    result_table = parseapi.aggregate(input_table, group_by_columns, aggregates)
    print(result_table)



    print("\nTest Case: Cross Join Without Condition")
    table_a = parseapi.Table(["ID", "Name", "Sales"], [
        [1, "Apple", 50],
        [2, "Banana", 40]
    ])

    table_b = parseapi.Table(["ID", "Color", "Count"], [
        [1, "Red", 10],
        [3, "Green", 20]
    ])

    # Cross Join Without Condition
    result_table = parseapi.cross_join(table_a, table_b)
    print(result_table)

    print("\nTest Case: Cross Join With Condition")
    # Cross Join With Condition (Join on ID)
    result_table_with_condition = parseapi.cross_join(
        table_a,
        table_b,
        condition=lambda row1, row2: row1[0] == row2[0]
    )
    print(result_table_with_condition)


tests()