# Notes for AI assistants

- This project uses **Flyte 2** (the `flyte` Python SDK, version >= 2.10): `flyte.TaskEnvironment`,
  `@env.task`, `flyte.Image`, `flyte.run`, `flyte.map`, `flyte.io.File`.
  It is **not** Flyte 1 / `flytekit`: no `@workflow`, `ImageSpec`, `pyflyte`, `map_task`.
- Flyte 2 docs: https://www.union.ai/docs/v2/flyte/llms.txt
- Docs/examples search as an MCP server:
  `uv run --with 'flyte[mcp]' flyte-mcp --transport stdio --tools search_flyte_sdk_examples,search_flyte_docs_examples,search_full_docs`
- This is a teaching exercise: explain your suggestions, and let the student make the design decisions.
