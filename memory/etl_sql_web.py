#!/usr/bin/env python3
"""
Web interface for ETL to SQL Converter
A user-friendly GUI for data stewards to convert ETL logic to SQL
"""

import os
import json
import asyncio
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional

import gradio as gr

# Import our advanced ETL to SQL converter
from advanced_etl_sql_converter import (
    AdvancedETLConverter,
    ETLToolType,
    DatabaseType,
    LLMConfig,
    LLMProvider,
    ConversionResult
)

# Initialize global converter
converter = None

async def convert_etl_code(
    etl_code: str,
    etl_tool: str,
    target_database: str,
    schema_info: str,
    business_rules: str,
    api_key: str,
    llm_provider: str,
    include_lineage: bool,
    include_data_quality: bool
) -> Dict[str, Any]:
    """Convert ETL code to SQL using the converter"""
    global converter
    
    try:
        # Parse schema info
        schema_dict = {}
        if schema_info.strip():
            try:
                schema_dict = json.loads(schema_info)
            except json.JSONDecodeError:
                return {
                    "sql": "",
                    "documentation": "Error: Schema info must be valid JSON.",
                    "error": "Invalid schema JSON format",
                    "lineage": ""
                }
        
        # Parse business rules
        rules_list = []
        if business_rules.strip():
            rules_list = [rule.strip() for rule in business_rules.split('\n') if rule.strip()]
        
        # Set up LLM config
        llm_config = LLMConfig(
            provider=LLMProvider(llm_provider),
            api_key=api_key,
            model_name=None,  # Use default model
        )
        
        # Initialize converter if needed
        if converter is None:
            converter = AdvancedETLConverter(llm_config)
        else:
            converter.llm_orchestrator.config = llm_config
        
        # Perform conversion
        result = await converter.convert(
            etl_code=etl_code,
            etl_tool=etl_tool,
            target_database=target_database,
            schema_info=schema_dict,
            business_rules=rules_list,
            include_lineage=include_lineage,
            include_data_quality=include_data_quality
        )
        
        # Return results
        if result.success:
            return {
                "sql": result.sql_code or "",
                "documentation": result.documentation or "No documentation generated",
                "error": None,
                "lineage": result.lineage_diagram or "No lineage generated",
                "execution_time": f"{result.execution_time:.2f} seconds"
            }
        else:
            return {
                "sql": "",
                "documentation": "",
                "error": result.error or "Unknown error occurred",
                "lineage": "",
                "execution_time": f"{result.execution_time:.2f} seconds"
            }
    
    except Exception as e:
        return {
            "sql": "",
            "documentation": "",
            "error": f"Error: {str(e)}",
            "lineage": "",
            "execution_time": "N/A"
        }

def save_results(sql_code, documentation, lineage):
    """Save results to files"""
    output_dir = os.path.join(os.getcwd(), "etl_conversions")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result = {}
    
    if sql_code:
        sql_path = os.path.join(output_dir, f"conversion_{timestamp}.sql")
        with open(sql_path, 'w') as f:
            f.write(sql_code)
        result["sql_path"] = sql_path
    
    if documentation:
        doc_path = os.path.join(output_dir, f"documentation_{timestamp}.md")
        with open(doc_path, 'w') as f:
            f.write(documentation)
        result["doc_path"] = doc_path
    
    if lineage:
        lineage_path = os.path.join(output_dir, f"lineage_{timestamp}.txt")
        with open(lineage_path, 'w') as f:
            f.write(lineage)
        result["lineage_path"] = lineage_path
    
    return f"Results saved to {output_dir}"

async def process_file_upload(
    file,
    etl_tool,
    target_database,
    schema_info,
    business_rules,
    api_key,
    llm_provider,
    include_lineage,
    include_data_quality
):
    """Process uploaded ETL file"""
    try:
        # Read file content
        if file is None:
            return {
                "sql": "",
                "documentation": "",
                "error": "No file uploaded",
                "lineage": "",
                "execution_time": "N/A"
            }
        
        file_content = file.decode('utf-8')
        
        # Convert the ETL code
        return await convert_etl_code(
            file_content,
            etl_tool,
            target_database,
            schema_info,
            business_rules,
            api_key,
            llm_provider,
            include_lineage,
            include_data_quality
        )
    
    except Exception as e:
        return {
            "sql": "",
            "documentation": "",
            "error": f"Error processing file: {str(e)}",
            "lineage": "",
            "execution_time": "N/A"
        }

def create_web_interface():
    """Create the Gradio web interface"""
    with gr.Blocks(title="ETL to SQL Converter") as app:
        gr.Markdown("# ETL to SQL Converter for Data Stewards")
        gr.Markdown("Convert ETL tool logic to standardized ANSI SQL or Spark SQL")
        
        with gr.Tabs():
            with gr.TabItem("Text Input"):
                with gr.Row():
                    with gr.Column():
                        etl_code_input = gr.Textbox(
                            label="ETL Code",
                            placeholder="Paste your ETL code here...",
                            lines=10
                        )
                        
                        with gr.Row():
                            etl_tool = gr.Dropdown(
                                label="ETL Tool",
                                choices=[e.value for e in ETLToolType],
                                value="informatica"
                            )
                            target_db = gr.Dropdown(
                                label="Target Database",
                                choices=[d.value for d in DatabaseType],
                                value="spark_sql"
                            )
                        
                        schema_info = gr.Textbox(
                            label="Schema Information (JSON format)",
                            placeholder='{"table_name": {"column1": "type", "column2": "type"}}',
                            lines=5
                        )
                        
                        business_rules = gr.Textbox(
                            label="Business Rules (one per line)",
                            placeholder="Rule 1\nRule 2\nRule 3",
                            lines=3
                        )
                        
                        with gr.Row():
                            include_lineage = gr.Checkbox(
                                label="Include Data Lineage",
                                value=True
                            )
                            include_quality = gr.Checkbox(
                                label="Include Data Quality Checks",
                                value=True
                            )
                        
                        with gr.Accordion("API Settings", open=False):
                            api_key = gr.Textbox(
                                label="API Key",
                                placeholder="Enter your API key",
                                type="password"
                            )
                            llm_provider = gr.Dropdown(
                                label="LLM Provider",
                                choices=[p.value for p in LLMProvider],
                                value="openai_gpt4"
                            )
                        
                        convert_btn = gr.Button("Convert to SQL", variant="primary")
            
            with gr.TabItem("File Upload"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="Upload ETL Code File")
                        
                        with gr.Row():
                            file_etl_tool = gr.Dropdown(
                                label="ETL Tool",
                                choices=[e.value for e in ETLToolType],
                                value="informatica"
                            )
                            file_target_db = gr.Dropdown(
                                label="Target Database",
                                choices=[d.value for d in DatabaseType],
                                value="spark_sql"
                            )
                        
                        file_schema_info = gr.Textbox(
                            label="Schema Information (JSON format)",
                            placeholder='{"table_name": {"column1": "type", "column2": "type"}}',
                            lines=5
                        )
                        
                        file_business_rules = gr.Textbox(
                            label="Business Rules (one per line)",
                            placeholder="Rule 1\nRule 2\nRule 3",
                            lines=3
                        )
                        
                        with gr.Row():
                            file_include_lineage = gr.Checkbox(
                                label="Include Data Lineage",
                                value=True
                            )
                            file_include_quality = gr.Checkbox(
                                label="Include Data Quality Checks",
                                value=True
                            )
                        
                        with gr.Accordion("API Settings", open=False):
                            file_api_key = gr.Textbox(
                                label="API Key",
                                placeholder="Enter your API key",
                                type="password"
                            )
                            file_llm_provider = gr.Dropdown(
                                label="LLM Provider",
                                choices=[p.value for p in LLMProvider],
                                value="openai_gpt4"
                            )
                        
                        file_convert_btn = gr.Button("Convert to SQL", variant="primary")
        
        # Output tabs
        with gr.Tabs():
            with gr.TabItem("Generated SQL"):
                sql_output = gr.Code(language="sql", label="SQL Code")
                exec_time = gr.Text(label="Execution Time")
                save_sql_btn = gr.Button("Save Results")
                save_result_text = gr.Text(label="Save Status")
            
            with gr.TabItem("Documentation"):
                doc_output = gr.Markdown()
            
            with gr.TabItem("Data Lineage"):
                lineage_output = gr.Text(label="Data Lineage")
            
            with gr.TabItem("Errors"):
                error_output = gr.Text(label="Error Messages")
        
        # Function to unpack dictionary results into separate outputs
        def unpack_result(result_dict):
            return (
                result_dict.get("sql", ""),
                result_dict.get("documentation", ""),
                result_dict.get("error", ""),
                result_dict.get("lineage", ""),
                result_dict.get("execution_time", "")
            )
        
        # Event handlers for text input
        convert_btn.click(
            fn=lambda *args: unpack_result(asyncio.run(convert_etl_code(*args))),
            inputs=[
                etl_code_input, etl_tool, target_db,
                schema_info, business_rules,
                api_key, llm_provider,
                include_lineage, include_quality
            ],
            outputs=[
                sql_output, doc_output, error_output, lineage_output, exec_time
            ]
        )
        
        # Event handlers for file upload
        file_convert_btn.click(
            fn=lambda *args: unpack_result(asyncio.run(process_file_upload(*args))),
            inputs=[
                file_input, file_etl_tool, file_target_db,
                file_schema_info, file_business_rules,
                file_api_key, file_llm_provider,
                file_include_lineage, file_include_quality
            ],
            outputs=[
                sql_output, doc_output, error_output, lineage_output, exec_time
            ]
        )
        
        # Save results handler
        save_sql_btn.click(
            fn=save_results,
            inputs=[sql_output, doc_output, lineage_output],
            outputs=[save_result_text]
        )
    
    return app

if __name__ == "__main__":
    # Check if gradio is installed
    try:
        import gradio
    except ImportError:
        print("Gradio is not installed. Please install it with:")
        print("pip install gradio")
        exit(1)
    
    app = create_web_interface()
    app.launch(share=False)
