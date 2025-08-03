# contract_creation_api.py - Add this to your existing FastAPI application

from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# New Pydantic models for contract creation
class ContractCreationRequest(BaseModel):
    query: str = Field(description="Natural language query for contract creation")
    table_name: Optional[str] = Field(None, description="Specific table name if provided")
    contract_type: Optional[str] = Field(None, description="Type of contract to create")
    include_governance: Optional[bool] = Field(True, description="Include governance rules")
    include_sla: Optional[bool] = Field(True, description="Include SLA requirements")
    output_format: Optional[str] = Field("markdown", description="Output format: markdown, json, yaml")


class SchemaField(BaseModel):
    name: str
    data_type: str
    required: bool = True
    description: str = ""
    constraints: Optional[List[str]] = []


class ContractTemplate(BaseModel):
    contract_name: str
    version: str = "1.0"
    owner: str = "Data Team"
    description: str = ""
    schema_fields: List[SchemaField] = []
    data_quality_rules: List[str] = []
    governance_rules: Dict[str, Any] = {}
    sla_requirements: Dict[str, Any] = {}


# Contract Creation Service
class ContractCreationService:
    """Service for creating data contracts and metadata documentation"""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    async def process_contract_request(self, request: ContractCreationRequest) -> Dict[str, Any]:
        """Process a contract creation request"""
        try:
            query_lower = request.query.lower()

            # Determine the type of contract creation request
            if "template" in query_lower:
                return await self._generate_template(request)
            elif "customer" in query_lower or "user" in query_lower:
                return await self._generate_customer_contract(request)
            elif "sales" in query_lower or "order" in query_lower:
                return await self._generate_sales_contract(request)
            elif "product" in query_lower or "inventory" in query_lower:
                return await self._generate_product_contract(request)
            elif any(keyword in query_lower for keyword in ["create", "generate", "document"]):
                return await self._generate_custom_contract(request)
            else:
                return await self._provide_guidance(request)

        except Exception as e:
            logger.error(f"Error processing contract request: {e}")
            return {
                "success": False,
                "error": f"Failed to process contract request: {str(e)}"
            }

    async def _generate_template(self, request: ContractCreationRequest) -> Dict[str, Any]:
        """Generate a generic contract template"""
        template_markdown = f"""# Data Contract Template

## Contract Information
- **Contract Name**: [Your Contract Name]
- **Version**: 1.0
- **Owner**: [Data Owner Name]
- **Created Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Last Updated**: {datetime.now().strftime('%Y-%m-%d')}

## Data Source Information
- **Source System**: [Source System Name]
- **Database/Schema**: [database.schema]
- **Table/Collection**: [table_name]

## Schema Definition

| Field Name | Data Type | Required | Description | Constraints |
|------------|-----------|----------|-------------|-------------|
| [field_1] | [VARCHAR(50)] | Yes | [Description of field] | [NOT NULL, UNIQUE] |
| [field_2] | [INTEGER] | No | [Description of field] | [DEFAULT 0] |
| [field_3] | [TIMESTAMP] | Yes | [Description of field] | [DEFAULT CURRENT_TIMESTAMP] |

## Data Quality Rules
- [ ] **Completeness**: All required fields must be populated
- [ ] **Uniqueness**: Primary key constraints must be maintained
- [ ] **Validity**: Data types and formats must be correct
- [ ] **Consistency**: Cross-field validation rules
- [ ] **Timeliness**: Data freshness requirements

## Data Governance

### Classification
- **Data Classification**: [Public/Internal/Confidential/Restricted]
- **PII Data**: [Yes/No]
- **Sensitive Data**: [Yes/No]

### Access Control
- **Read Access**: [Role/Group names]
- **Write Access**: [Role/Group names]
- **Admin Access**: [Role/Group names]

### Retention Policy
- **Retention Period**: [Time period]
- **Archival Policy**: [Archive strategy]
- **Deletion Policy**: [Deletion rules]

## Data Lineage
- **Upstream Sources**: [List source systems/tables]
- **Downstream Consumers**: [List consuming systems/applications]
- **Transformation Logic**: [Brief description of transformations]

## SLA Requirements

### Availability
- **Uptime Target**: 99.9%
- **Maintenance Windows**: [Scheduled maintenance times]

### Performance
- **Query Response Time**: < 100ms for simple queries
- **Batch Processing Time**: [Time requirements]
- **Throughput**: [Records per second/minute/hour]

### Data Freshness
- **Update Frequency**: [Real-time/Hourly/Daily/Weekly]
- **Maximum Lag**: [Acceptable delay]

## Monitoring and Alerting
- **Data Quality Monitoring**: [Monitoring strategy]
- **Performance Monitoring**: [Performance metrics]
- **Alert Conditions**: [Alert triggers]
- **Notification Recipients**: [Contact information]

## Change Management
- **Change Process**: [How changes are managed]
- **Approval Workflow**: [Approval requirements]
- **Version Control**: [Versioning strategy]

## Contact Information
- **Data Owner**: [Name and email]
- **Data Steward**: [Name and email]
- **Technical Contact**: [Name and email]
- **Business Contact**: [Name and email]

---
*Generated on {datetime.now().isoformat()}*
"""

        return {
            "success": True,
            "result_type": "markdown",
            "content": template_markdown,
            "metadata": {
                "title": "Data Contract Template",
                "filename": "data_contract_template",
                "type": "template",
                "created_at": datetime.now().isoformat()
            }
        }

    async def _generate_customer_contract(self, request: ContractCreationRequest) -> Dict[str, Any]:
        """Generate a customer data contract"""
        # Get customer-related data from database
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Query for customer-related elements
        cursor.execute("""
                       SELECT v_data_element_name, v_data_element_code, v_table_name
                       FROM business_element_mapping
                       WHERE v_data_element_name LIKE '%customer%'
                          OR v_data_element_name LIKE '%user%'
                          OR v_table_name LIKE '%customer%'
                       ORDER BY v_data_element_name
                       """)

        customer_elements = cursor.fetchall()
        conn.close()

        # Build schema information
        schema_section = ""
        if customer_elements:
            schema_section = "| Column Name | Data Type | Required | Description |\n|-------------|-----------|----------|-------------|\n"
            for element in customer_elements[:10]:  # Limit to first 10
                schema_section += f"| {element[0]} | VARCHAR(100) | Yes | {element[0].replace('_', ' ').title()} |\n"
        else:
            schema_section = """| Column Name | Data Type | Required | Description |
|-------------|-----------|----------|-------------|
| customer_id | INTEGER | Yes | Primary key, unique customer identifier |
| first_name | VARCHAR(50) | Yes | Customer's first name |
| last_name | VARCHAR(50) | Yes | Customer's last name |
| email | VARCHAR(100) | Yes | Customer's email address (unique) |
| phone | VARCHAR(20) | No | Customer's phone number |
| created_at | TIMESTAMP | Yes | Record creation timestamp |
| updated_at | TIMESTAMP | Yes | Last update timestamp |"""

        customer_contract = f"""# Customer Data Contract

## Contract Overview
- **Contract Name**: Customer Data Contract
- **Version**: 1.0
- **Owner**: Customer Data Team
- **Created Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Description**: Defines the structure, governance, and quality standards for customer data

## Data Source
- **Source System**: CRM System / Customer Database
- **Database**: customer_db
- **Primary Table**: customers

## Schema Definition

{schema_section}

## Data Quality Rules

### Mandatory Validations
- ✅ **Email Uniqueness**: Each email address must be unique across all customer records
- ✅ **Name Validation**: First and last names cannot contain numeric characters
- ✅ **Email Format**: Must follow valid email format (user@domain.com)
- ✅ **Phone Format**: Must follow standard phone number format if provided
- ✅ **Timestamp Validation**: All timestamps must be in UTC format

### Data Quality Metrics
- **Completeness**: > 95% for required fields
- **Validity**: > 99% for format validations
- **Uniqueness**: 100% for email addresses
- **Timeliness**: Data must be updated within 5 minutes of source change

## Data Governance

### Data Classification
- **Classification Level**: PII - Restricted
- **Contains PII**: Yes (names, email, phone)
- **GDPR Applicable**: Yes
- **Data Sensitivity**: High

### Access Control
- **Read Access**: Customer Service Team, Analytics Team, Marketing Team
- **Write Access**: Customer Service Team, Data Integration Service
- **Admin Access**: Data Engineering Team, DBA Team
- **Audit Access**: Compliance Team, Data Governance Team

### Privacy & Compliance
- **Right to be Forgotten**: Supported via anonymization process
- **Data Masking**: Required for non-production environments
- **Consent Management**: Tracked via consent_status field
- **Retention Period**: 7 years after account closure (regulatory requirement)

## Data Lineage

### Upstream Sources
- **CRM System**: Primary customer data entry point
- **Web Registration**: Online customer sign-ups
- **Mobile App**: Mobile customer registrations
- **Customer Service**: Manual updates and corrections

### Downstream Consumers
- **Analytics Database**: Customer behavior analysis
- **Marketing Platform**: Campaign targeting and personalization
- **Billing System**: Invoice and payment processing
- **Support System**: Customer service ticket management

### Data Flow
```
CRM/Web/Mobile → Customer Database → Analytics/Marketing/Billing
```

## SLA Requirements

### Availability
- **Uptime Target**: 99.95% (maximum 22 minutes downtime per month)
- **Maintenance Windows**: Sundays 2:00-4:00 AM EST
- **Disaster Recovery**: RTO 4 hours, RPO 15 minutes

### Performance
- **Query Response Time**: 
  - Simple lookups: < 50ms
  - Complex joins: < 200ms
  - Bulk operations: < 2 seconds
- **Concurrent Users**: Support up to 500 concurrent connections
- **Throughput**: 10,000 transactions per minute peak load

### Data Freshness
- **Real-time Updates**: Critical customer changes (contact info)
- **Batch Updates**: Non-critical data (preferences, segments)
- **Maximum Acceptable Lag**: 5 minutes for real-time, 1 hour for batch

## Monitoring & Quality Checks

### Automated Monitoring
- **Data Quality Dashboards**: Real-time quality metrics
- **Anomaly Detection**: Automated alerts for unusual patterns
- **Schema Drift Detection**: Monitor for unexpected schema changes
- **Volume Monitoring**: Track record counts and growth patterns

### Alert Conditions
- **Data Quality**: Quality score drops below 95%
- **Performance**: Query response time exceeds SLA thresholds
- **Availability**: Service downtime detected
- **Security**: Unauthorized access attempts

### Notification Recipients
- **Data Owner**: sarah.johnson@company.com
- **Data Steward**: mike.davis@company.com
- **On-call Engineer**: oncall-data@company.com

## Change Management

### Change Process
1. **Request**: Submit change request via Data Governance portal
2. **Impact Analysis**: Assess downstream system impacts
3. **Approval**: Require approval from Data Owner and affected teams
4. **Testing**: Validate changes in staging environment
5. **Deployment**: Deploy during scheduled maintenance window
6. **Validation**: Post-deployment validation and monitoring

### Version Control
- **Semantic Versioning**: Major.Minor.Patch format
- **Backward Compatibility**: Maintain for at least 2 major versions
- **Migration Strategy**: Automated migration scripts for schema changes

## Business Context

### Key Business Metrics
- **Customer Acquisition Cost (CAC)**
- **Customer Lifetime Value (CLV)**
- **Churn Rate**
- **Customer Satisfaction Score (CSAT)**

### Critical Business Processes
- **Customer Onboarding**: New customer registration and verification
- **Customer Support**: Issue resolution and account management
- **Marketing Campaigns**: Targeted marketing and personalization
- **Billing & Payments**: Invoice generation and payment processing

---
*This contract was generated on {datetime.now().isoformat()}*
*For questions or updates, contact the Data Governance Team*
"""

        return {
            "success": True,
            "result_type": "markdown",
            "content": customer_contract,
            "metadata": {
                "title": "Customer Data Contract",
                "filename": "customer_data_contract",
                "type": "data_contract",
                "table": "customers",
                "classification": "PII-Restricted",
                "created_at": datetime.now().isoformat()
            }
        }

    async def _generate_sales_contract(self, request: ContractCreationRequest) -> Dict[str, Any]:
        """Generate a sales/order data contract"""
        sales_contract = f"""# Sales Order Data Contract

## Contract Overview
- **Contract Name**: Sales Order Data Contract
- **Version**: 1.0
- **Owner**: Sales Operations Team
- **Created Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Description**: Defines structure and governance for sales order data

## Schema Definition

| Column Name | Data Type | Required | Description |
|-------------|-----------|----------|-------------|
| order_id | INTEGER | Yes | Primary key, unique order identifier |
| customer_id | INTEGER | Yes | Foreign key to customer table |
| order_date | TIMESTAMP | Yes | When the order was placed |
| order_status | VARCHAR(20) | Yes | Current status of the order |
| order_total | DECIMAL(10,2) | Yes | Total order amount including tax |
| subtotal | DECIMAL(10,2) | Yes | Order amount before tax and discounts |
| tax_amount | DECIMAL(10,2) | Yes | Total tax amount |
| discount_amount | DECIMAL(10,2) | No | Total discount applied |
| shipping_address | TEXT | Yes | Delivery address |
| payment_method | VARCHAR(50) | Yes | Payment method used |
| created_at | TIMESTAMP | Yes | Record creation timestamp |
| updated_at | TIMESTAMP | Yes | Last modification timestamp |

## Data Quality Rules

### Validation Rules
- ✅ **Order Total Calculation**: order_total = subtotal + tax_amount - discount_amount
- ✅ **Status Values**: Must be one of [pending, confirmed, shipped, delivered, cancelled]
- ✅ **Positive Amounts**: All monetary fields must be >= 0
- ✅ **Valid Customer**: customer_id must exist in customers table
- ✅ **Date Logic**: order_date cannot be in the future

### Business Rules
- Orders cannot be modified once status is 'shipped'
- Cancelled orders must have order_total = 0
- Tax calculation must comply with jurisdiction rules
- Discounts cannot exceed subtotal amount

## Data Lineage

### Upstream Sources
- **E-commerce Platform**: Online orders
- **POS System**: In-store purchases  
- **Call Center**: Phone orders
- **Mobile App**: Mobile purchases

### Downstream Consumers
- **Financial Reporting**: Revenue analytics
- **Inventory Management**: Stock level updates
- **Customer Service**: Order tracking and support
- **Marketing Analytics**: Purchase behavior analysis

## SLA & Performance
- **Availability**: 99.9% uptime
- **Query Performance**: < 200ms average response time
- **Data Freshness**: Real-time for order status updates
- **Backup**: Hourly incremental, daily full backup

---
*Generated on {datetime.now().isoformat()}*
"""

        return {
            "success": True,
            "result_type": "markdown",
            "content": sales_contract,
            "metadata": {
                "title": "Sales Order Data Contract",
                "filename": "sales_order_contract",
                "type": "data_contract",
                "created_at": datetime.now().isoformat()
            }
        }

    async def _generate_product_contract(self, request: ContractCreationRequest) -> Dict[str, Any]:
        """Generate a product/inventory data contract"""
        product_contract = f"""# Product Catalog Data Contract

## Contract Overview  
- **Contract Name**: Product Catalog Data Contract
- **Version**: 1.0
- **Owner**: Product Management Team
- **Created Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Description**: Defines structure and governance for product catalog data

## Schema Definition

| Column Name | Data Type | Required | Description |
|-------------|-----------|----------|-------------|
| product_id | INTEGER | Yes | Primary key, unique product identifier |
| sku | VARCHAR(50) | Yes | Stock keeping unit (unique) |
| product_name | VARCHAR(200) | Yes | Display name of the product |
| description | TEXT | No | Detailed product description |
| category_id | INTEGER | Yes | Product category classification |
| brand | VARCHAR(100) | Yes | Product brand name |
| price | DECIMAL(10,2) | Yes | Current selling price |
| cost | DECIMAL(10,2) | Yes | Product cost (restricted access) |
| inventory_count | INTEGER | Yes | Current stock level |
| is_active | BOOLEAN | Yes | Whether product is available for sale |
| created_at | TIMESTAMP | Yes | Product creation date |
| updated_at | TIMESTAMP | Yes | Last modification timestamp |

## Data Quality & Business Rules

### Quality Standards
- ✅ **SKU Uniqueness**: Each SKU must be globally unique
- ✅ **Price Validation**: Price must be greater than cost
- ✅ **Inventory Accuracy**: Stock count must be non-negative
- ✅ **Category Validation**: category_id must exist in categories table
- ✅ **Active Status**: Inactive products cannot have inventory > 0

### Derived Metrics
- **Profit Margin**: (price - cost) / price * 100
- **Inventory Value**: inventory_count * cost  
- **Turnover Rate**: Sales velocity calculation
- **Stock Status**: Low/Medium/High based on inventory levels

## Data Governance
- **Classification**: Business Confidential
- **Cost Access**: Restricted to Finance and Management only
- **Update Frequency**: Real-time for inventory, daily for pricing
- **Retention**: 10 years for historical product data

---
*Generated on {datetime.now().isoformat()}*
"""

        return {
            "success": True,
            "result_type": "markdown",
            "content": product_contract,
            "metadata": {
                "title": "Product Catalog Data Contract",
                "filename": "product_catalog_contract",
                "type": "data_contract",
                "created_at": datetime.now().isoformat()
            }
        }

    async def _generate_custom_contract(self, request: ContractCreationRequest) -> Dict[str, Any]:
        """Generate a custom contract based on the query"""
        # Extract table name or entity from query
        query = request.query.lower()
        table_name = "custom_table"

        # Try to extract table name from query
        import re
        table_match = re.search(r'(?:for|create|table|entity)\s+(\w+)', query)
        if table_match:
            table_name = table_match.group(1)

        custom_contract = f"""# {table_name.title()} Data Contract

## Contract Information
- **Contract Name**: {table_name.title()} Data Contract
- **Version**: 1.0
- **Owner**: [To be assigned]
- **Created Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Description**: Auto-generated contract for {table_name} based on request: "{request.query}"

## Schema Definition (Template)

| Column Name | Data Type | Required | Description |
|-------------|-----------|----------|-------------|
| id | INTEGER | Yes | Primary key identifier |
| name | VARCHAR(100) | Yes | Name or title field |
| description | TEXT | No | Detailed description |
| status | VARCHAR(20) | Yes | Current status |
| created_at | TIMESTAMP | Yes | Creation timestamp |
| updated_at | TIMESTAMP | Yes | Last update timestamp |

## Customization Required ⚠️

This is a template contract. Please customize the following:

1. **Schema Fields**: Add/modify columns based on actual requirements
2. **Data Types**: Adjust data types and constraints
3. **Business Rules**: Define specific validation rules
4. **Ownership**: Assign data owner and steward
5. **Classification**: Set appropriate data classification level
6. **SLA Requirements**: Define performance and availability needs

## Next Steps

1. Review and customize schema definition
2. Define data quality rules
3. Establish governance policies
4. Set up monitoring and alerting
5. Get stakeholder approval

## Template Sections to Complete

### Data Quality Rules
- [ ] Define validation rules
- [ ] Set completeness thresholds  
- [ ] Establish uniqueness constraints
- [ ] Define format validations

### Data Governance
- [ ] Set data classification level
- [ ] Define access control policies
- [ ] Establish retention policies
- [ ] Set up compliance requirements

### SLA Requirements  
- [ ] Define availability targets
- [ ] Set performance benchmarks
- [ ] Establish data freshness requirements
- [ ] Define backup and recovery needs

---
*Template generated on {datetime.now().isoformat()}*
*Please customize before implementation*
"""

        return {
            "success": True,
            "result_type": "markdown",
            "content": custom_contract,
            "metadata": {
                "title": f"{table_name.title()} Data Contract",
                "filename": f"{table_name}_contract_template",
                "type": "custom_contract",
                "requires_customization": True,
                "created_at": datetime.now().isoformat()
            }
        }

    async def _provide_guidance(self, request: ContractCreationRequest) -> Dict[str, Any]:
        """Provide guidance when request is unclear"""
        guidance = f"""# Contract Creation Guidance

## Your Request
"{request.query}"

## I can help you create various types of contracts:

### 🏢 **Business Entity Contracts**
- Customer data contracts
- Product/inventory contracts  
- Sales/order contracts
- Employee/HR contracts

### 📋 **Template Types**
- Generic data contract template
- API contract specifications
- Data pipeline contracts
- Schema documentation

### 🔧 **Custom Contracts**
- Specific table documentation
- Custom entity definitions
- Integration contracts

## Example Requests

| Request Type | Example |
|--------------|---------|
| **Customer Data** | "Create data contract for customer table" |
| **Sales Data** | "Generate contract for sales orders" |
| **Product Data** | "Document product catalog schema" |
| **Template** | "Create data contract template" |
| **Custom** | "Generate contract for user_preferences table" |

## What I Need From You

To create a comprehensive contract, please specify:

1. **Entity/Table Name**: What data are you documenting?
2. **Purpose**: How is this data used?
3. **Scope**: What level of detail do you need?
4. **Audience**: Who will use this contract?

### Try These Specific Requests:
- "Create a customer data contract"
- "Generate a template for product data"  
- "Document the orders table schema"
- "Create governance rules for PII data"

---
*Need help? Just ask me to create a contract for any specific data entity!*
"""

        return {
            "success": True,
            "result_type": "markdown",
            "content": guidance,
            "metadata": {
                "title": "Contract Creation Guidance",
                "filename": "contract_guidance",
                "type": "guidance",
                "created_at": datetime.now().isoformat()
            }
        }


# Add these endpoints to your existing FastAPI application:

@app.post("/api/contract/create")
async def create_contract(request: ContractCreationRequest):
    """Create a data contract based on natural language request"""
    try:
        logger.info(f"Processing contract creation request: {request.query}")

        # Initialize contract service
        contract_service = ContractCreationService(db_manager_global)

        # Process the request
        result = await contract_service.process_contract_request(request)

        logger.info(f"Contract creation completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error in contract creation: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to create contract",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/api/contract/templates")
async def list_contract_templates():
    """List available contract templates"""
    templates = [
        {
            "id": "generic_template",
            "name": "Generic Data Contract Template",
            "description": "Basic template for any data contract",
            "category": "template"
        },
        {
            "id": "customer_contract",
            "name": "Customer Data Contract",
            "description": "Specialized contract for customer/user data",
            "category": "business_entity"
        },
        {
            "id": "sales_contract",
            "name": "Sales Order Contract",
            "description": "Contract for sales and order data",
            "category": "business_entity"
        },
        {
            "id": "product_contract",
            "name": "Product Catalog Contract",
            "description": "Contract for product and inventory data",
            "category": "business_entity"
        }
    ]

    return {
        "success": True,
        "templates": templates,
        "total_count": len(templates)
    }


@app.post("/api/contract/validate")
async def validate_contract(contract_data: Dict[str, Any]):
    """Validate a contract structure and content"""
    try:
        # Basic validation logic
        required_sections = [
            "contract_name", "version", "owner", "schema_definition"
        ]

        missing_sections = []
        for section in required_sections:
            if section not in contract_data:
                missing_sections.append(section)

        validation_result = {
            "valid": len(missing_sections) == 0,
            "missing_sections": missing_sections,
            "recommendations": [],
            "score": 0
        }

        # Calculate validation score
        score = max(0, 100 - (len(missing_sections) * 25))
        validation_result["score"] = score

        # Add recommendations
        if missing_sections:
            validation_result["recommendations"].append(
                f"Add missing sections: {', '.join(missing_sections)}"
            )

        if score < 75:
            validation_result["recommendations"].append(
                "Consider adding more detailed governance and SLA sections"
            )

        return validation_result

    except Exception as e:
        logger.error(f"Error validating contract: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Contract validation failed: {str(e)}"
        )