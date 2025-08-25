#!/usr/bin/env python3
"""
Client script for Advanced ETL to SQL Converter
Demonstrates usage of the converter for data stewards
"""

import asyncio
import json
import os
import argparse
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Import our advanced ETL to SQL converter
from advanced_etl_sql_converter import (
    AdvancedETLConverter, 
    ETLToolType, 
    DatabaseType, 
    LLMConfig,
    LLMProvider,
    ConversionResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ETLConversionCLI:
    """Command-line interface for ETL to SQL conversion"""
    
    def __init__(self):
        """Initialize the CLI"""
        self.parser = self._setup_argument_parser()
        
    def _setup_argument_parser(self):
        """Set up command-line argument parser"""
        parser = argparse.ArgumentParser(
            description="Convert ETL tool logic to ANSI SQL/Spark SQL",
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        parser.add_argument(
            "input_file",
            help="Path to the ETL code file to convert"
        )
        
        parser.add_argument(
            "--etl-tool",
            choices=[e.value for e in ETLToolType],
            required=True,
            help="Type of ETL tool (informatica, python_pandas, etc.)"
        )
        
        parser.add_argument(
            "--target-db",
            choices=[d.value for d in DatabaseType],
            default="spark_sql",
            help="Target database type (default: spark_sql)"
        )
        
        parser.add_argument(
            "--schema-file",
            help="Path to JSON file containing schema information"
        )
        
        parser.add_argument(
            "--rules-file",
            help="Path to text file containing business rules (one per line)"
        )
        
        parser.add_argument(
            "--output-dir",
            default="./output",
            help="Directory to save conversion results (default: ./output)"
        )
        
        parser.add_argument(
            "--llm-provider",
            choices=[p.value for p in LLMProvider],
            default="openai_gpt4",
            help="LLM provider to use (default: openai_gpt4)"
        )
        
        parser.add_argument(
            "--api-key",
            help="API key for LLM provider (if not provided, uses environment variable)"
        )
        
        parser.add_argument(
            "--model",
            help="Model to use (if not provided, uses provider default)"
        )
        
        parser.add_argument(
            "--no-lineage",
            action="store_true",
            help="Skip data lineage generation"
        )
        
        parser.add_argument(
            "--no-quality-checks",
            action="store_true",
            help="Skip data quality check generation"
        )
        
        return parser
    
    def parse_args(self):
        """Parse command-line arguments"""
        return self.parser.parse_args()
    
    @staticmethod
    def load_schema_file(file_path: str) -> Dict[str, Any]:
        """Load schema information from JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading schema file: {e}")
            return {}
    
    @staticmethod
    def load_rules_file(file_path: str) -> List[str]:
        """Load business rules from text file"""
        try:
            with open(file_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error loading rules file: {e}")
            return []
    
    @staticmethod
    def read_etl_code(file_path: str) -> str:
        """Read ETL code from file"""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading ETL code file: {e}")
            raise

    @staticmethod
    def save_results(result: ConversionResult, output_dir: str, input_file: str):
        """Save conversion results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a timestamp and base filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        base_filename = f"{base_name}_{timestamp}"
        
        # Save SQL code
        if result.sql_code:
            sql_path = os.path.join(output_dir, f"{base_filename}.sql")
            with open(sql_path, 'w') as f:
                f.write(result.sql_code)
            logger.info(f"SQL code saved to: {sql_path}")
        
        # Save documentation
        if result.documentation:
            doc_path = os.path.join(output_dir, f"{base_filename}_doc.md")
            with open(doc_path, 'w') as f:
                f.write(result.documentation)
            logger.info(f"Documentation saved to: {doc_path}")
        
        # Save lineage diagram
        if result.lineage_diagram:
            lineage_path = os.path.join(output_dir, f"{base_filename}_lineage.txt")
            with open(lineage_path, 'w') as f:
                f.write(result.lineage_diagram)
            logger.info(f"Data lineage saved to: {lineage_path}")
        
        # Save full results as JSON
        result_path = os.path.join(output_dir, f"{base_filename}_results.json")
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Full results saved to: {result_path}")

async def main():
    """Main entry point"""
    cli = ETLConversionCLI()
    args = cli.parse_args()
    
    try:
        # Read ETL code
        etl_code = cli.read_etl_code(args.input_file)
        
        # Load schema and rules if provided
        schema_info = cli.load_schema_file(args.schema_file) if args.schema_file else {}
        business_rules = cli.load_rules_file(args.rules_file) if args.rules_file else []
        
        # Set up LLM config
        api_key = args.api_key or os.environ.get(
            "OPENAI_API_KEY" if args.llm_provider == "openai_gpt4" else
            "ANTHROPIC_API_KEY" if args.llm_provider == "anthropic_claude" else
            "AZURE_OPENAI_API_KEY"
        )
        
        if not api_key:
            raise ValueError(f"API key not provided for {args.llm_provider}. "
                           f"Please provide it via --api-key or set the appropriate environment variable.")
        
        llm_config = LLMConfig(
            provider=LLMProvider(args.llm_provider),
            api_key=api_key,
            model_name=args.model or None,
        )
        
        # Initialize converter
        converter = AdvancedETLConverter(llm_config)
        
        print(f"🔄 Converting {args.input_file} ({args.etl_tool}) to {args.target_db}...")
        
        # Perform conversion
        result = await converter.convert(
            etl_code=etl_code,
            etl_tool=args.etl_tool,
            target_database=args.target_db,
            schema_info=schema_info,
            business_rules=business_rules,
            include_lineage=not args.no_lineage,
            include_data_quality=not args.no_quality_checks
        )
        
        if result.success:
            print(f"✅ Conversion successful! (Time: {result.execution_time:.2f}s)")
            
            # Save results
            cli.save_results(result, args.output_dir, args.input_file)
            
            # Print a preview of the SQL
            if result.sql_code:
                print("\n📊 SQL Preview:")
                preview_lines = result.sql_code.split('\n')[:15]
                print('\n'.join(preview_lines))
                if len(result.sql_code.split('\n')) > 15:
                    print("... (more lines in output file)")
        else:
            print(f"❌ Conversion failed: {result.error}")
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"❌ An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())
