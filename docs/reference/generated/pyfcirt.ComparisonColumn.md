# ComparisonColumn

```{eval-rst}
.. autoclass:: pyfcirt.ComparisonColumn
```

Identify the recorded statement pair represented by one wide answer column.

## Parameters


`column` : str or int
: Exact answer-column name in the wide DataFrame.

`block_id` : str or int
: Block to which this pair belongs.

`statement_1`, `statement_2` : str or int
: The two statement IDs in the order used when the answers were recorded.


## Returns

ComparisonColumn
: One descriptor in `ResponseTable(..., format="comparison_wide", comparison_columns=...)`.
  Supply one descriptor for each answer column.

## Examples

```python
import pyfcirt as fc

column = fc.ComparisonColumn("choice_01", "B1", "A", "B")
print(column)
```

```text
ComparisonColumn(column='choice_01', block_id='B1', statement_1='A', statement_2='B')
```

## See also

[Wide comparison example](pyfcirt.ResponseTable.md#a-wide-pair-table)
