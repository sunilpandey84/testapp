import React, { useState } from 'react';
import { Brain, Code2, Database, Zap, CheckCircle, AlertCircle, Loader, MessageSquare } from 'lucide-react';

const LLMETLDemo = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [etlInput, setEtlInput] = useState('');
  const [llmResponse, setLlmResponse] = useState('');
  const [finalSQL, setFinalSQL] = useState('');
  const [selectedTool, setSelectedTool] = useState('informatica');
  const [targetDB, setTargetDB] = useState('postgresql');

  const processingSteps = [
    { id: 'analysis', title: 'Code Analysis', desc: 'LLM analyzes ETL structure and logic', icon: Brain },
    { id: 'understanding', title: 'Business Logic Understanding', desc: 'Extracts business rules and transformations', icon: MessageSquare },
    { id: 'generation', title: 'SQL Generation', desc: 'Generates equivalent SQL code', icon: Code2 },
    { id: 'optimization', title: 'Database Optimization', desc: 'Optimizes for target database', icon: Database },
    { id: 'validation', title: 'Quality Validation', desc: 'Validates generated SQL', icon: CheckCircle }
  ];

  const sampleETLCode = {
    informatica: `<MAPPING NAME="customer_sales_analysis">
  <SOURCE NAME="CUSTOMERS" DATABASETYPE="Oracle">
    <TRANSFORMFIELD NAME="CUSTOMER_ID" DATATYPE="NUMBER"/>
    <TRANSFORMFIELD NAME="CUSTOMER_NAME" DATATYPE="VARCHAR2"/>
    <TRANSFORMFIELD NAME="REGION" DATATYPE="VARCHAR2"/>
    <TRANSFORMFIELD NAME="STATUS" DATATYPE="VARCHAR2"/>
  </SOURCE>
  <SOURCE NAME="SALES" DATABASETYPE="Oracle">
    <TRANSFORMFIELD NAME="SALE_ID" DATATYPE="NUMBER"/>
    <TRANSFORMFIELD NAME="CUSTOMER_ID" DATATYPE="NUMBER"/>
    <TRANSFORMFIELD NAME="AMOUNT" DATATYPE="NUMBER"/>
    <TRANSFORMFIELD NAME="SALE_DATE" DATATYPE="DATE"/>
  </SOURCE>
  <TRANSFORMATION TYPE="Filter" NAME="FLT_ACTIVE_CUSTOMERS">
    <CONDITION>STATUS = 'ACTIVE'</CONDITION>
  </TRANSFORMATION>
  <TRANSFORMATION TYPE="Joiner" NAME="JNR_CUST_SALES">
    <CONDITION>CUSTOMERS.CUSTOMER_ID = SALES.CUSTOMER_ID</CONDITION>
    <JOIN_TYPE>INNER</JOIN_TYPE>
  </TRANSFORMATION>
  <TRANSFORMATION TYPE="Expression" NAME="EXP_CALCULATIONS">
    <EXPRESSION PORT="SALES_QUARTER">
      DECODE(TO_CHAR(SALE_DATE, 'Q'), '1', 'Q1', '2', 'Q2', '3', 'Q3', '4', 'Q4')
    </EXPRESSION>
    <EXPRESSION PORT="CUSTOMER_TIER">
      IIF(AMOUNT > 10000, 'PREMIUM', IIF(AMOUNT > 1000, 'STANDARD', 'BASIC'))
    </EXPRESSION>
  </TRANSFORMATION>
  <TRANSFORMATION TYPE="Aggregator" NAME="AGG_SALES_SUMMARY">
    <GROUP_BY>REGION, SALES_QUARTER, CUSTOMER_TIER</GROUP_BY>
    <EXPRESSION PORT="TOTAL_SALES">SUM(AMOUNT)</EXPRESSION>
    <EXPRESSION PORT="AVG_SALES">AVG(AMOUNT)</EXPRESSION>
    <EXPRESSION PORT="CUSTOMER_COUNT">COUNT(DISTINCT CUSTOMER_ID)</EXPRESSION>
  </TRANSFORMATION>
</MAPPING>`,

    python: `import pandas as pd
import numpy as np
from datetime import datetime

def process_customer_sales_analysis(customers_df, sales_df):
    """
    Advanced ETL processing for customer sales analysis
    Implements complex business rules and transformations
    """
    
    # Data Quality Checks
    print("🔍 Performing data quality checks...")
    customers_df = customers_df.dropna(subset=['customer_id', 'customer_name'])
    sales_df = sales_df.dropna(subset=['customer_id', 'amount', 'sale_date'])
    
    # Filter active customers only
    active_customers = customers_df[customers_df['status'] == 'ACTIVE'].copy()
    
    # Business Rule: Calculate sales quarters
    sales_df['sales_quarter'] = sales_df['sale_date'].dt.quarter.map({
        1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'
    })
    
    # Business Rule: Customer tier classification
    def classify_customer_tier(amount):
        if amount > 10000:
            return 'PREMIUM'
        elif amount > 1000:
            return 'STANDARD'
        else:
            return 'BASIC'
    
    sales_df['customer_tier'] = sales_df['amount'].apply(classify_customer_tier)
    
    # Complex join with validation
    merged_df = pd.merge(
        active_customers,
        sales_df,
        on='customer_id',
        how='inner',
        validate='one_to_many'
    )
    
    # Advanced aggregation with multiple metrics
    result = merged_df.groupby(['region', 'sales_quarter', 'customer_tier']).agg({
        'amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique',
        'sale_date': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    result.columns = ['region', 'sales_quarter', 'customer_tier', 
                     'total_sales', 'avg_sales', 'transaction_count',
                     'unique_customers', 'first_sale_date', 'last_sale_date']
    
    # Business Rule: Add performance indicators
    result['performance_score'] = (
        result['total_sales'] * 0.4 + 
        result['avg_sales'] * 0.3 + 
        result['unique_customers'] * 100 * 0.3
    )
    
    return result.round(2)`
  };

  const mockLLMResponses = {
    analysis: {
      informatica: `🧠 **LLM Analysis Result:**

**Data Sources Identified:**
- CUSTOMERS table (Oracle): customer_id, customer_name, region, status
- SALES table (Oracle): sale_id, customer_id, amount, sale_date

**Transformations Detected:**
1. **Filter Transformation**: Active customers only (STATUS = 'ACTIVE')
2. **Joiner Transformation**: Inner join between customers and sales on customer_id
3. **Expression Transformation**: 
   - Quarter calculation using DECODE function
   - Customer tier classification using nested IIF
4. **Aggregator Transformation**: Group by region, quarter, tier with multiple aggregations

**Complexity Assessment:** HIGH
- Multiple complex transformations
- Nested business logic
- Cross-table aggregations
- Date/time manipulations

**SQL Strategy:** Will use CTEs for clarity, CASE statements for business logic, and window functions for optimization.`,

      python: `🧠 **LLM Analysis Result:**

**Functions Identified:**
- process_customer_sales_analysis(): Main ETL function with complex logic

**Operations Detected:**
1. **Data Quality**: dropna() operations for null handling
2. **Filtering**: Boolean indexing for active customers
3. **Date Operations**: Quarter extraction and mapping
4. **Business Logic**: Custom tier classification function
5. **Joins**: pd.merge() with validation
6. **Aggregations**: Multi-level groupby with multiple metrics
7. **Calculations**: Performance score formula

**Pandas Operations to Convert:**
- DataFrame filtering → WHERE clauses
- apply() functions → CASE statements  
- groupby().agg() → GROUP BY with multiple aggregations
- pd.merge() → JOIN operations

**Complexity Assessment:** HIGH
- Custom functions requiring CASE logic
- Multiple aggregation levels
- Complex business calculations
- Data validation requirements`
    },

    sql_generation: {
      postgresql: `-- 🔄 **Generated PostgreSQL SQL from LLM:**

-- Customer Sales Analysis Report
-- Converted from: {tool} ETL Logic
-- Target Database: PostgreSQL
-- Generated: ${new Date().toLocaleDateString()}

WITH active_customers AS (
    -- Filter: Only active customers
    SELECT 
        customer_id,
        customer_name,
        region,
        status
    FROM customers
    WHERE status = 'ACTIVE'
      AND customer_id IS NOT NULL
      AND customer_name IS NOT NULL
),

sales_enriched AS (
    -- Business Logic: Add calculated fields
    SELECT 
        s.sale_id,
        s.customer_id,
        s.amount,
        s.sale_date,
        
        -- Quarter calculation (converted from DECODE)
        CASE EXTRACT(QUARTER FROM s.sale_date)
            WHEN 1 THEN 'Q1'
            WHEN 2 THEN 'Q2' 
            WHEN 3 THEN 'Q3'
            WHEN 4 THEN 'Q4'
        END as sales_quarter,
        
        -- Customer tier classification (converted from nested IIF)
        CASE 
            WHEN s.amount > 10000 THEN 'PREMIUM'
            WHEN s.amount > 1000 THEN 'STANDARD'
            ELSE 'BASIC'
        END as customer_tier
        
    FROM sales s
    WHERE s.customer_id IS NOT NULL
      AND s.amount IS NOT NULL
      AND s.sale_date IS NOT NULL
      AND s.amount > 0  -- Data quality: positive amounts only
),

customer_sales_joined AS (
    -- Join: Customers with their sales (Inner Join)
    SELECT 
        c.customer_id,
        c.customer_name,
        c.region,
        s.sale_id,
        s.amount,
        s.sale_date,
        s.sales_quarter,
        s.customer_tier
    FROM active_customers c
    INNER JOIN sales_enriched s ON c.customer_id = s.customer_id
)

-- Final Aggregation: Sales summary by region, quarter, and tier
SELECT 
    region,
    sales_quarter,
    customer_tier,
    
    -- Aggregated metrics
    SUM(amount) as total_sales,
    AVG(amount) as avg_sales,
    COUNT(DISTINCT customer_id) as customer_count,
    COUNT(sale_id) as transaction_count,
    
    -- Additional insights
    MIN(sale_date) as first_sale_date,
    MAX(sale_date) as last_sale_date,
    
    -- Performance indicator
    ROUND(
        SUM(amount) * 0.4 + 
        AVG(amount) * 0.3 + 
        COUNT(DISTINCT customer_id) * 100 * 0.3,
        2
    ) as performance_score

FROM customer_sales_joined
GROUP BY region, sales_quarter, customer_tier
ORDER BY region, sales_quarter, total_sales DESC;

-- Quality Check: Ensure no data loss
-- Expected records: Should match source after filtering
-- Validation: Check for NULL values in key fields`
    }
  };

  const simulateProcessing = async () => {
    setIsProcessing(true);
    setActiveStep(0);
    setLlmResponse('');
    setFinalSQL('');

    // Step 1: Analysis
    await new Promise(resolve => setTimeout(resolve, 1500));
    setActiveStep(1);
    setLlmResponse(mockLLMResponses.analysis[selectedTool]);

    // Step 2: Understanding
    await new Promise(resolve => setTimeout(resolve, 1200));
    setActiveStep(2);

    // Step 3: Generation
    await new Promise(resolve => setTimeout(resolve, 1800));
    setActiveStep(3);

    // Step 4: Optimization
    await new Promise(resolve => setTimeout(resolve, 1000));
    setActiveStep(4);
    setFinalSQL(mockLLMResponses.sql_generation[targetDB]);

    // Step 5: Validation
    await new Promise(resolve => setTimeout(resolve, 800));
    setActiveStep(5);

    setIsProcessing(false);
  };

  const resetDemo = () => {
    setActiveStep(0);
    setIsProcessing(false);
    setLlmResponse('');
    setFinalSQL('');
  };

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 min-h-screen text-white">
      <div className="mb-8">
        <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent mb-4">
          🧠 LLM-Powered ETL Converter
        </h1>
        <p className="text-xl text-slate-300 mb-2">
          Watch as AI intelligently converts your ETL logic to optimized SQL
        </p>
        <div className="flex items-center gap-4 text-sm text-slate-400">
          <span className="flex items-center gap-1">
            <Brain className="w-4 h-4" />
            GPT-4 / Claude Integration
          </span>
          <span className="flex items-center gap-1">
            <Zap className="w-4 h-4" />
            Real-time Processing
          </span>
          <span className="flex items-center gap-1">
            <Database className="w-4 h-4" />
            Multi-DB Support
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-8">
        {/* Input Configuration */}
        <div className="bg-slate-800/40 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
          <h2 className="text-2xl font-semibold mb-4 text-slate-200">Configure Your Conversion</h2>
          
          {/* Tool Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-300 mb-2">Source ETL Tool</label>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setSelectedTool('informatica');
                  setEtlInput(sampleETLCode.informatica);
                }}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  selectedTool === 'informatica'
                    ? 'bg-blue-600 text-white shadow-lg'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                Informatica
              </button>
              <button
                onClick={() => {
                  setSelectedTool('python');
                  setEtlInput(sampleETLCode.python);
                }}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  selectedTool === 'python'
                    ? 'bg-green-600 text-white shadow-lg'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                Python/Pandas
              </button>
            </div>
          </div>

          {/* Target Database */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-300 mb-2">Target Database</label>
            <select
              value={targetDB}
              onChange={(e) => setTargetDB(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white"
            >
              <option value="postgresql">PostgreSQL</option>
              <option value="mysql">MySQL</option>
              <option value="oracle">Oracle</option>
              <option value="snowflake">Snowflake</option>
            </select>
          </div>

          {/* ETL Code Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-slate-300 mb-2">
              {selectedTool === 'informatica' ? 'Informatica XML Mapping' : 'Python ETL Code'}
            </label>
            <textarea
              value={etlInput || sampleETLCode[selectedTool]}
              onChange={(e) => setEtlInput(e.target.value)}
              className="w-full h-64 bg-slate-900/50 border border-slate-600 rounded-lg p-4 text-sm font-mono text-slate-200 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Your ETL code will appear here..."
            />
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={simulateProcessing}
              disabled={isProcessing}
              className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-slate-600 disabled:to-slate-600 text-white px-6 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2 shadow-lg disabled:shadow-none"
            >
              {isProcessing ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  Processing with AI...
                </>
              ) : (
                <>
                  <Brain className="w-4 h-4" />
                  Convert with LLM
                </>
              )}
            </button>
            <button
              onClick={resetDemo}
              className="px-4 py-3 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Processing Steps */}
        <div className="bg-slate-800/40 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
          <h2 className="text-2xl font-semibold mb-4 text-slate-200">AI Processing Pipeline</h2>
          
          <div className="space-y-4">
            {processingSteps.map((step, index) => {
              const Icon = step.icon;
              const isActive = activeStep === index;
              const isCompleted = activeStep > index;
              const isNext = activeStep === index - 1;

              return (
                <div
                  key={step.id}
                  className={`p-4 rounded-lg border-2 transition-all duration-500 ${
                    isActive
                      ? 'border-blue-500 bg-blue-500/10 shadow-lg'
                      : isCompleted
                      ? 'border-green-500 bg-green-500/10'
                      : isNext && isProcessing
                      ? 'border-yellow-500 bg-yellow-500/10'
                      : 'border-slate-600 bg-slate-700/30'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-full ${
                      isActive
                        ? 'bg-blue-500 text-white'
                        : isCompleted
                        ? 'bg-green-500 text-white'
                        : 'bg-slate-600 text-slate-300'
                    }`}>
                      {isCompleted ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : isActive && isProcessing ? (
                        <Loader className="w-4 h-4 animate-spin" />
                      ) : (
                        <Icon className="w-4 h-4" />
                      )}
                    </div>
                    <div>
                      <h3 className={`font-medium ${
                        isActive || isCompleted ? 'text-white' : 'text-slate-400'
                      }`}>
                        {step.title}
                      </h3>
                      <p className={`text-sm ${
                        isActive || isCompleted ? 'text-slate-300' : 'text-slate-500'
                      }`}>
                        {step.desc}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* LLM Analysis Response */}
      {llmResponse && (
        <div className="mb-6 bg-slate-800/40 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
          <h3 className="text-xl font-semibold mb-4 text-slate-200 flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-blue-400" />
            LLM Analysis & Understanding
          </h3>
          <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-600">
            <pre className="text-sm text-slate-300 whitespace-pre-wrap font-mono">
              {llmResponse}
            </pre>
          </div>
        </div>
      )}

      {/* Generated SQL Output */}
      {finalSQL && (
        <div className="bg-slate-800/40 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
          <h3 className="text-xl font-semibold mb-4 text-slate-200 flex items-center gap-2">
            <Code2 className="w-5 h-5 text-green-400" />
            Generated & Optimized SQL
          </h3>
          <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-600">
            <pre className="text-sm text-green-300 whitespace-pre-wrap font-mono overflow-x-auto">
              {finalSQL}
            </pre>
          </div>
          
          <div className="mt-4 p-4 bg-green-900/20 border border-green-700/50 rounded-lg">
            <div className="flex items-center gap-2 text-green-300 mb-2">
              <CheckCircle className="w-5 h-5" />
              <span className="font-medium">Conversion Completed Successfully!</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-slate-400">Source Tool:</span>
                <div className="text-green-400 font-medium capitalize">{selectedTool}</div>
              </div>
              <div>
                <span className="text-slate-400">Target DB:</span>
                <div className="text-green-400 font-medium uppercase">{targetDB}</div>
              </div>
              <div>
                <span className="text-slate-400">LLM Used:</span>
                <div className="text-green-400 font-medium">Claude-3-Sonnet</div>
              </div>
              <div>
                <span className="text-slate-400">Quality Score:</span>
                <div className="text-green-400 font-medium">98.5%</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Benefits Showcase */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
          <Brain className="w-8 h-8 text-blue-400 mb-3" />
          <h4 className="font-semibold text-slate-200 mb-2">Intelligent Analysis</h4>
          <p className="text-slate-400 text-sm">
            LLM understands complex ETL logic, business rules, and data transformations beyond simple pattern matching
          </p>
        </div>
        <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
          <Zap className="w-8 h-8 text-yellow-400 mb-3" />
          <h4 className="font-semibold text-slate-200 mb-2">Context-Aware Generation</h4>
          <p className="text-slate-400 text-sm">
            Generates SQL that preserves business logic, handles edge cases, and optimizes for your specific database
          </p>
        </div>
        <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
          <CheckCircle className="w-8 h-8 text-green-400 mb-3" />
          <h4 className="font-semibold text-slate-200 mb-2">Production Ready</h4>
          <p className="text-slate-400 text-sm">
            Includes data quality checks, error handling, performance optimizations, and comprehensive documentation
          </p>
        </div>
      </div>
    </div>
  );
};

export default LLMETLDemo;