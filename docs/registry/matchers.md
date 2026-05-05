# Matchers

## Matcher language: `model_field`

Routing rules can match on registry fields:

```yaml
rules:
  - name: long-context-only
    match:
      all_of:
        - endpoint: { literal: /v1/messages }
        - model_field:
            field: context_size
            op: gte
            value: 200000
    action: { provider: anthropic, mode: translate }

  - name: vision-routing
    match:
      model_field:
        field: input_modalities
        op: contains
        value: image
    action: { provider: openrouter, mode: translate }
```

Operators: `eq`, `gt`, `gte`, `lt`, `lte` (numeric/string scalars),
`contains` (membership in tuple fields like `input_modalities` /
`output_modalities`), `in` (membership of the field value in a list).
