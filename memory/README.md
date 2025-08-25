# ETL to SQL Converter for Data Stewards

A powerful tool for converting various ETL tool logic into standardized ANSI SQL or Spark SQL, with a conversational data lineage assistant.

## Overview

This project provides a comprehensive solution for data stewards to convert ETL processes from different tools (like Informatica, Python Pandas, Talend, etc.) into standardized SQL code. The converter leverages advanced language models (LLMs) to intelligently analyze the ETL logic, extract data lineage, and generate optimized SQL with proper data quality checks.

## Features

- **Multiple ETL Tool Support**: Convert ETL code from Informatica, Python Pandas, PySpark, Talend, Pentaho, SSIS, and more
- **Target Database Flexibility**: Generate SQL optimized for Spark SQL, Snowflake, BigQuery, PostgreSQL, and other databases
- **Data Lineage Extraction**: Automatically map source-to-target data flows
- **Data Quality Checks**: Generate SQL for validating data quality
- **Comprehensive Documentation**: Auto-generate markdown documentation for the converted SQL
- **Multiple Interfaces**: Command-line, Python API, and web interface options
- **Conversational Lineage Assistant**: Interactive chat interface to explore data lineage
- **Human-in-the-Loop Feedback**: The assistant can detect uncertainty and request clarification from users to improve responses

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables for LLM access:
```bash
# For OpenAI
export OPENAI_API_KEY=your-api-key-here

# For Anthropic
export ANTHROPIC_API_KEY=your-api-key-here

# For Google Generative AI
export GOOGLE_API_KEY=your-api-key-here
```

## Usage

### Command-Line Interface

```bash
python etl_sql_cli.py path/to/etl_code.txt --etl-tool informatica --target-db spark_sql
```

### Conversational Lineage Assistant

The project now includes a conversational agent that can help you analyze and understand data lineage across your data ecosystem. There are several ways to use this feature:

#### Option 1: Full UI with API (Requires Node.js)

If you have Node.js installed, you can run the full application with Angular frontend:

```bash
./start.sh
```

If you don't have Node.js installed, you can install it using:

```bash
./install_node.sh
```

#### Option 2: API Only (Doesn't require Node.js)

To run only the API server without the Angular frontend:

```bash
./start_api_only.sh
```

You can then interact with the API using curl or any API client:

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me the available data contracts"}'
```

#### Option 3: Direct Python Console Interface

You can also interact with the lineage agent directly in Python:

```bash
python -c "import conversational_lineage_agent; agent = conversational_lineage_agent.ConversationalLineageAgent(); print(agent.process_message('Show me all available data contracts'))"
```

Additional options:
```bash
python etl_sql_cli.py --help
```

### Human-in-the-Loop Feedback System

The conversational lineage agent includes a sophisticated feedback mechanism that detects when the AI might be uncertain about a response and proactively engages the user for clarification.

#### How It Works

1. **Uncertainty Detection**: The system analyzes responses for phrases and patterns that indicate uncertainty.
2. **Feedback Request**: When uncertainty is detected, the system asks the user for clarification.
3. **Response Refinement**: The system incorporates user feedback to generate a more accurate and complete response.
4. **Continuous Learning**: Feedback is stored in memory to improve future interactions.

#### Command-Line Arguments

The human-in-the-loop system can be configured when starting the conversational lineage agent:

```bash
python conversational_lineage_agent.py [options]
```

Options include:
- `--no-human-feedback`: Disable the human-in-the-loop feedback system
- `--interface {cli,web}`: Select interface mode (command-line or web API)

#### Example Usage

In interactive mode, when the agent is uncertain, it will prompt for feedback:

```
> Show lineage for customer data

I need your help to better answer your question:
========================================================
Your question: Show lineage for customer data

My initial understanding:
I'm not sure which specific customer data element you're referring to. There are several customer-related data elements in the system.

Could you please provide additional information or clarification?
Your feedback: I'm looking for the customer_id field lineage

Processing your refinement...

RESPONSE:
========================================================
Here's the lineage for the customer_id field:

[Detailed response with lineage graph]
```

This feature helps ensure more accurate and useful responses when the initial query lacks complete context.

### Web Interface

```bash
python etl_sql_web.py
```

Then open your browser to http://localhost:7860

### Python API

```python
import asyncio
from advanced_etl_sql_converter import AdvancedETLConverter, ETLToolType, DatabaseType

async def convert_my_etl():
    converter = AdvancedETLConverter()
    
    with open('my_etl_code.xml', 'r') as f:
        etl_code = f.read()
    
    result = await converter.convert(
        etl_code=etl_code,
        etl_tool=ETLToolType.INFORMATICA,
        target_database=DatabaseType.SPARK_SQL,
        schema_info={
            "customers": {
                "customer_id": "int primary key",
                "customer_name": "varchar(100)",
                "region": "varchar(50)"
            },
            "sales": {
                "sale_id": "int primary key",
                "customer_id": "int foreign key",
                "amount": "decimal(10,2)",
                "sale_date": "date"
            }
        },
        business_rules=[
            "Only include active customers",
            "Sales amounts must be positive"
        ]
    )
    
    if result.success:
        print(result.sql_code)
        print(result.documentation)
    else:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(convert_my_etl())
```

## Supported ETL Tools

- Informatica PowerCenter
- Python Pandas
- PySpark
- Talend
- Pentaho Data Integration
- Microsoft SSIS
- IBM DataStage
- Database Stored Procedures (SQL Server, Oracle PL/SQL, PostgreSQL PL/pgSQL, MySQL)
- SQL Scripts (various dialects)
- Custom ETL code

## Supported Target Databases

- Spark SQL (primary focus)
- PostgreSQL
- Snowflake
- BigQuery
- Redshift
- MySQL
- Oracle
- SQL Server
- Hive
- Generic ANSI SQL

## Directory Structure

```
.
├── advanced_etl_sql_converter.py  # Main converter implementation
├── etl_sql_cli.py                # Command-line interface
├── etl_sql_web.py               # Web interface using Gradio
├── requirements.txt             # Package dependencies
└── README.md                    # This file
```

## Examples

### Converting an Informatica workflow to Spark SQL:

```bash
python etl_sql_cli.py examples/informatica_workflow.xml --etl-tool informatica --target-db spark_sql --schema-file examples/schema.json --rules-file examples/business_rules.txt
```

### Converting a SQL Server stored procedure to Spark SQL:

```bash
python etl_sql_cli.py examples/customer_sales_procedure.sql --etl-tool stored_procedure --target-db spark_sql
```

### Converting an Oracle PL/SQL procedure to Spark SQL:

```bash
python etl_sql_cli.py examples/oracle_customer_sales.sql --etl-tool stored_procedure --target-db spark_sql
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
