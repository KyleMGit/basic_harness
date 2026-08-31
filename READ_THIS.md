Treat this block as repository-wide operator guidance.

Work carefully within the user's requested scope. Preserve unrelated work, verify changes proportionately, and report results and limitations clearly.

## Mandatory Entity-Type Clarification

When the user supplies a name or named identifier without explicitly stating what type of entity it represents, you MUST stop and ask the user to identify the entity type before writing SQL, searching metadata, or calling any database tool.

Do not infer or guess the entity type from the name, spelling, surrounding context, database metadata, or likely matches.

Ask:

> What type of entity does '<name>' refer to—for example, a customer, employee, account, organization, product, database table, report, or something else?

Clarification is not required when the user has already stated the entity type unambiguously, such as "customer John Smith" or "table customer_orders".

## Mandatory Fuzzy Name Matching

After the entity type is clear, queries that search a name value MUST use case-insensitive fuzzy containment matching rather than exact equality. Use:

```sql
name_column ILIKE '%name%'
```

Do not use `name_column = 'name'` unless the user explicitly requests an exact match. Replace `name_column` only with a metadata-verified column and replace `name` with the user's requested value, safely escaped as a SQL string literal. If the active SQL dialect does not support `ILIKE`, use its case-insensitive equivalent and state that substitution. Do not treat a fuzzy match as unique when multiple rows are returned.
