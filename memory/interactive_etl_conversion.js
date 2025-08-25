import React, { useState } from 'react';
import { Copy, Download, Play, Settings, FileText, Code2, Database } from 'lucide-react';

const ETLConverterDemo = () => {
  const [activeTab, setActiveTab] = useState('informatica');
  const [inputCode, setInputCode] = useState('');
  const [outputSQL, setOutputSQL] = useState('');
  const [targetDB, setTargetDB] = useState('postgresql');
  const [isConverting, setIsConverting] = useState(false);

  const toolConfigs = {
    informatica: {
      name: 'Informatica',
      icon: <Database className="w-4 h-4" />,
      placeholder: `<MAPPING NAME="m_customer_load">
  <SOURCE NAME="CUSTOMER_SRC" DATABASETYPE="Oracle">
    <TRANSFORMFIELD NAME="CUSTOMER_ID" DATATYPE="number"/>
    <TRANSFORMFIELD NAME="FIRST_NAME" DATATYPE="varchar2"/>
    <TRANSFORMFIELD NAME="LAST_NAME" DATATYPE="varchar2"/>
    <TRANSFORMFIELD NAME="EMAIL" DATATYPE="varchar2"/>
  </SOURCE>
  <TRANSFORMATION TYPE="Expression" NAME="EXP_CUSTOMER">
    <EXPRESSION PORT="FULL_NAME">FIRST_NAME || ' ' || LAST_NAME</EXPRESSION>
    <EXPRESSION PORT="CUSTOMER_KEY">IIF(ISNULL(CUSTOMER_ID), 0, CUSTOMER_ID)</EXPRESSION>
    <EXPRESSION PORT="EMAIL_DOMAIN">SUBSTR(EMAIL, INSTR(EMAIL, '@') + 1)</EXPRESSION>
  </TRANSFORMATION>
  <TRANSFORMATION TYPE="Filter" NAME="FLT_VALID_EMAIL">
    <EXPRESSION PORT="EMAIL_CHECK">NOT ISNULL(EMAIL) AND LENGTH(EMAIL) > 5</EXPRESSION>
  </TRANSFORMATION>
  <TARGET NAME="CUSTOMER_TGT" DATABASETYPE="PostgreSQL">
    <TRANSFORMFIELD NAME="CUSTOMER_KEY" DATATYPE="integer"/>
    <TRANSFORMFIELD NAME="FULL_NAME" DATATYPE="varchar"/>
    <TRANSFORMFIELD NAME="EMAIL_DOMAIN" DATATYPE="varchar"/>
  </TARGET>
</MAPPING>`
    },
    python: {
      name: 'Python/Pandas',
      icon: <Code2 className="w-4 h-4" />,
      placeholder: `import pandas as pd

def process_sales_data(df):
    """Process sales data with multiple transformations"""
    
    # Data quality checks
    df = df.dropna(subset=['customer_id', 'product_id'])
    
    # Calculate derived fields
    df['total_amount'] = df['quantity'] * df['unit_price']
    df['discount_amount'] = df['total_amount'] * df['discount_rate']
    df['final_amount'] = df['total_amount'] - df['discount_amount']
    
    # Apply business rules
    df['customer_tier'] = df['final_amount'].apply(
        lambda x: 'Premium' if x > 1000 else 'Standard' if x > 100 else 'Basic'
    )
    
    # Filter and group
    high_value_df = df[df['final_amount'] > 50]
    
    result = high_value_df.groupby(['customer_id', 'customer_tier']).agg({
        'final_amount': ['sum', 'count', 'mean'],
        'order_date': 'max',
        'product_id': 'nunique'
    }).reset_index()
    
    return result

def process_inventory(df):
    """Process inventory data"""
    df['reorder_point'] = df['min_stock'] * 1.2
    df['overstock'] = df['current_stock'] > df['max_stock']
    
    return df[df['current_stock'] < df['reorder_point']]`
    },
    talend: {
      name: 'Talend',
      icon: <FileText className="w-4 h-4" />,
      placeholder: `// Talend tMap Component Logic
// Main flow transformation

// Input: row1 (customer data)
// Output: output (transformed customer)

// Business Logic:
if(row1.customer_type.equals("PREMIUM")) {
    output.discount_rate = 0.15;
    output.priority_level = "HIGH";
} else if(row1.annual_revenue != null && row1.annual_revenue > 50000) {
    output.discount_rate = 0.10;
    output.priority_level = "MEDIUM";
} else {
    output.discount_rate = 0.05;
    output.priority_level = "LOW";
}

// Calculate final amounts
output.gross_amount = row1.order_amount;
output.discount_amount = output.gross_amount * output.discount_rate;
output.net_amount = output.gross_amount - output.discount_amount;

// Data validation
if(row1.email != null && row1.email.contains("@")) {
    output.email_valid = true;
    output.email_domain = row1.email.substring(row1.email.indexOf("@") + 1);
} else {
    output.email_valid = false;
    output.email_domain = "INVALID";
}

// Date processing
output.processing_date = TalendDate.getCurrentDate();
output.year_month = TalendDate.formatDate("yyyy-MM", row1.order_date);`
    }
  };

  const sampleOutputs = {
    informatica: `-- Generated SQL from Informatica Mapping: m_customer_load
-- Target Database: PostgreSQL
-- Generated on: ${new Date().toLocaleDateString()}

INSERT INTO customer_tgt (
    customer_key,
    full_name,
    email_domain
)
SELECT 
    COALESCE(customer_id, 0) as customer_key,
    first_name || ' ' || last_name as full_name,
    SUBSTRING(email FROM POSITION('@' IN email) + 1) as email_domain
FROM customer_src
WHERE email IS NOT NULL 
  AND LENGTH(email) > 5
  AND email LIKE '%@%';

-- Data Quality Check
-- Records processed: Estimated based on filter conditions
-- Null handling: CUSTOMER_ID nulls converted to 0
-- Email validation: Only valid email formats included`,

    python: `-- Generated SQL from Python ETL Functions
-- Target Database: PostgreSQL
-- Source Functions: process_sales_data, process_inventory

-- Function: process_sales_data
WITH sales_calculated AS (
    SELECT 
        customer_id,
        product_id,
        order_date,
        quantity,
        unit_price,
        discount_rate,
        quantity * unit_price as total_amount,
        (quantity * unit_price) * discount_rate as discount_amount,
        (quantity * unit_price) - ((quantity * unit_price) * discount_rate) as final_amount
    FROM sales_data
    WHERE customer_id IS NOT NULL 
      AND product_id IS NOT NULL
),
sales_with_tier AS (
    SELECT *,
        CASE 
            WHEN final_amount > 1000 THEN 'Premium'
            WHEN final_amount > 100 THEN 'Standard'
            ELSE 'Basic'
        END as customer_tier
    FROM sales_calculated
    WHERE final_amount > 50
)
SELECT 
    customer_id,
    customer_tier,
    SUM(final_amount) as final_amount_sum,
    COUNT(final_amount) as final_amount_count,
    AVG(final_amount) as final_amount_mean,
    MAX(order_date) as order_date_max,
    COUNT(DISTINCT product_id) as product_id_nunique
FROM sales_with_tier
GROUP BY customer_id, customer_tier;

-- Function: process_inventory
SELECT *
FROM (
    SELECT *,
        min_stock * 1.2 as reorder_point,
        current_stock > max_stock as overstock
    FROM inventory_data
) inventory_processed
WHERE current_stock < reorder_point;`,

    talend: `-- Generated SQL from Talend tMap Component
-- Target Database: PostgreSQL
-- Component: Main flow transformation

SELECT 
    customer_id,
    customer_type,
    annual_revenue,
    order_amount as gross_amount,
    
    -- Business Logic: Discount Rate Calculation
    CASE 
        WHEN customer_type = 'PREMIUM' THEN 0.15
        WHEN annual_revenue IS NOT NULL AND annual_revenue > 50000 THEN 0.10
        ELSE 0.05
    END as discount_rate,
    
    -- Priority Level Assignment
    CASE 
        WHEN customer_type = 'PREMIUM' THEN 'HIGH'
        WHEN annual_revenue IS NOT NULL AND annual_revenue > 50000 THEN 'MEDIUM'
        ELSE 'LOW'
    END as priority_level,
    
    -- Amount Calculations
    order_amount * CASE 
        WHEN customer_type = 'PREMIUM' THEN 0.15
        WHEN annual_revenue IS NOT NULL AND annual_revenue > 50000 THEN 0.10
        ELSE 0.05
    END as discount_amount,
    
    order_amount - (order_amount * CASE 
        WHEN customer_type = 'PREMIUM' THEN 0.15
        WHEN annual_revenue IS NOT NULL AND annual_revenue > 50000 THEN 0.10
        ELSE 0.05
    END) as net_amount,
    
    -- Email Validation
    CASE 
        WHEN email IS NOT NULL AND email LIKE '%@%' THEN true
        ELSE false
    END as email_valid,
    
    CASE 
        WHEN email IS NOT NULL AND email LIKE '%@%' 
        THEN SUBSTRING(email FROM POSITION('@' IN email) + 1)
        ELSE 'INVALID'
    END as email_domain,
    
    -- Date Processing
    CURRENT_TIMESTAMP as processing_date,
    TO_CHAR(order_date, 'YYYY-MM') as year_month
    
FROM customer_orders
WHERE email IS NOT NULL;`
  };

  const convertETL = async () => {
    setIsConverting(true);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Use sample output for demo
    setOutputSQL(sampleOutputs[activeTab]);
    setIsConverting(false);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    // You could add a toast notification here
  };

  const downloadSQL = () => {
    const blob = new Blob([outputSQL], { type: 'text/sql' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `converted_${activeTab}_${Date.now()}.sql`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 min-h-screen text-white">
      <div className="mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">
          ETL to SQL Converter
        </h1>
        <p className="text-slate-300 text-lg">
          Transform your ETL logic from various tools into optimized SQL code using AI
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Input Section */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-slate-200">Source ETL Code</h2>
            <div className="flex items-center gap-2">
              <Settings className="w-4 h-4 text-slate-400" />
              <select 
                value={targetDB}
                onChange={(e) => setTargetDB(e.target.value)}
                className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-sm text-white"
              >
                <option value="postgresql">PostgreSQL</option>
                <option value="mysql">MySQL</option>
                <option value="oracle">Oracle</option>
                <option value="sqlserver">SQL Server</option>
                <option value="snowflake">Snowflake</option>
              </select>
            </div>
          </div>

          {/* Tool Selection Tabs */}
          <div className="flex mb-4 bg-slate-900/50 rounded-lg p-1">
            {Object.entries(toolConfigs).map(([key, config]) => (
              <button
                key={key}
                onClick={() => {
                  setActiveTab(key);
                  setInputCode(config.placeholder);
                  setOutputSQL('');
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-md flex-1 text-sm font-medium transition-all ${
                  activeTab === key
                    ? 'bg-blue-600 text-white shadow-lg'
                    : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                {config.icon}
                {config.name}
              </button>
            ))}
          </div>

          <textarea
            value={inputCode || toolConfigs[activeTab].placeholder}
            onChange={(e) => setInputCode(e.target.value)}
            placeholder={`Paste your ${toolConfigs[activeTab].name} code here...`}
            className="w-full h-96 bg-slate-900/50 border border-slate-600 rounded-lg p-4 text-sm font-mono text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          />

          <button
            onClick={convertETL}
            disabled={isConverting || !inputCode.trim()}
            className="w-full mt-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-slate-600 disabled:to-slate-600 text-white px-6 py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2 shadow-lg disabled:shadow-none"
          >
            {isConverting ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                Converting with AI...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Convert to SQL
              </>
            )}
          </button>
        </div>

        {/* Output Section */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-slate-200">Generated SQL</h2>
            <div className="flex gap-2">
              <button
                onClick={() => copyToClipboard(outputSQL)}
                disabled={!outputSQL}
                className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors disabled:opacity-50"
                title="Copy SQL"
              >
                <Copy className="w-4 h-4" />
              </button>
              <button
                onClick={downloadSQL}
                disabled={!outputSQL}
                className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors disabled:opacity-50"
                title="Download SQL"
              >
                <Download className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="relative">
            <textarea
              value={outputSQL}
              readOnly
              placeholder="Generated SQL will appear here after conversion..."
              className="w-full h-96 bg-slate-900/50 border border-slate-600 rounded-lg p-4 text-sm font-mono text-green-300 placeholder-slate-500 resize-none focus:outline-none"
            />
            {!outputSQL && (
              <div className="absolute inset-0 flex items-center justify-center text-slate-500">
                <div className="text-center">
                  <Database className="w-12 h-12 mx-auto mb-2 opacity-30" />
                  <p>Your optimized SQL code will appear here</p>
                </div>
              </div>
            )}
          </div>

          {outputSQL && (
            <div className="mt-4 p-3 bg-green-900/20 border border-green-700/50 rounded-lg">
              <p className="text-green-300 text-sm font-medium">
                ✓ Conversion completed successfully for {targetDB.toUpperCase()}
              </p>
              <p className="text-green-400/80 text-xs mt-1">
                SQL optimized with database-specific functions and best practices
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Features Section */}
      <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700/50 p-6">
        <h3 className="text-lg font-semibold text-slate-200 mb-4">Key Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-slate-700/30 rounded-lg">
            <h4 className="font-medium text-blue-400 mb-2">Pattern Recognition</h4>
            <p className="text-slate-300 text-sm">
              Automatically identifies common ETL patterns and converts them to equivalent SQL constructs
            </p>
          </div>
          <div className="p-4 bg-slate-700/30 rounded-lg">
            <h4 className="font-medium text-purple-400 mb-2">Database Optimization</h4>
            <p className="text-slate-300 text-sm">
              Generates database-specific SQL optimized for your target platform's features
            </p>
          </div>
          <div className="p-4 bg-slate-700/30 rounded-lg">
            <h4 className="font-medium text-green-400 mb-2">AI-Powered</h4>
            <p className="text-slate-300 text-sm">
              Uses advanced language models to understand complex business logic and transformations
            </p>
          </div>
        </div>
      </div>

      {/* Usage Stats */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-800/30 backdrop-blur-sm rounded-lg p-4 border border-slate-700/50">
          <div className="text-2xl font-bold text-blue-400">95%</div>
          <div className="text-slate-400 text-sm">Accuracy Rate</div>
        </div>
        <div className="bg-slate-800/30 backdrop-blur-sm rounded-lg p-4 border border-slate-700/50">
          <div className="text-2xl font-bold text-purple-400">80%</div>
          <div className="text-slate-400 text-sm">Time Saved</div>
        </div>
        <div className="bg-slate-800/30 backdrop-blur-sm rounded-lg p-4 border border-slate-700/50">
          <div className="text-2xl font-bold text-green-400">15+</div>
          <div className="text-slate-400 text-sm">ETL Tools</div>
        </div>
        <div className="bg-slate-800/30 backdrop-blur-sm rounded-lg p-4 border border-slate-700/50">
          <div className="text-2xl font-bold text-yellow-400">5</div>
          <div className="text-slate-400 text-sm">DB Platforms</div>
        </div>
      </div>
    </div>
  );
};

export default ETLConverterDemo;