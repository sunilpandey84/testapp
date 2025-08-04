import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Copy, Download, RefreshCw, Database, FileText, MessageSquare, Settings } from 'lucide-react';

// Message Types
const MESSAGE_TYPES = {
  TEXT: 'text',
  VISUALIZATION: 'visualization',
  FEEDBACK_REQUEST: 'feedback_request',
  ERROR: 'error',
  SYSTEM: 'system',
  MARKDOWN: 'markdown',
  TABLE: 'table'
};

// Chat Modes
const CHAT_MODES = {
  LINEAGE: 'lineage',
  CONTRACT: 'contract'
};

// Chat Message Class
class ChatMessage {
  constructor(role, content, messageType = MESSAGE_TYPES.TEXT, metadata = {}) {
    this.id = Date.now() + Math.random();
    this.role = role;
    this.content = content;
    this.timestamp = new Date();
    this.messageType = messageType;
    this.metadata = metadata;
  }
}

// Mode Selector Component
const ModeSelector = ({ currentMode, onModeChange, disabled }) => {
  return (
    <div className="mode-selector">
      <div className="mode-buttons">
        <button
          className={`mode-btn ${currentMode === CHAT_MODES.LINEAGE ? 'active' : ''}`}
          onClick={() => onModeChange(CHAT_MODES.LINEAGE)}
          disabled={disabled}
        >
          <Database size={20} />
          <span>Data Lineage</span>
          <small>Trace & analyze data flows</small>
        </button>
        <button
          className={`mode-btn ${currentMode === CHAT_MODES.CONTRACT ? 'active' : ''}`}
          onClick={() => onModeChange(CHAT_MODES.CONTRACT)}
          disabled={disabled}
        >
          <FileText size={20} />
          <span>Contract Creation</span>
          <small>Generate data contracts & metadata</small>
        </button>
      </div>
      <div className="mode-indicator">
        <span className="indicator-dot" style={{
          backgroundColor: currentMode === CHAT_MODES.LINEAGE ? '#3b82f6' : '#10b981'
        }}></span>
        <span className="indicator-text">
          {currentMode === CHAT_MODES.LINEAGE ? 'Lineage Mode' : 'Contract Mode'}
        </span>
      </div>
    </div>
  );
};

// Markdown Message Component
const MarkdownMessage = ({ content, metadata }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleDownload = () => {
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${metadata?.filename || 'contract'}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="markdown-message">
      <div className="markdown-header">
        <div className="markdown-title">
          <FileText size={16} />
          <span>{metadata?.title || 'Generated Content'}</span>
        </div>
        <div className="markdown-actions">
          <button onClick={handleCopy} className="action-btn" title="Copy to clipboard">
            <Copy size={14} />
            {copied ? 'Copied!' : 'Copy'}
          </button>
          <button onClick={handleDownload} className="action-btn" title="Download as markdown">
            <Download size={14} />
            Download
          </button>
        </div>
      </div>
      <div className="markdown-content">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    </div>
  );
};

// Contract Creation Quick Actions
const ContractQuickActions = ({ onActionClick, disabled }) => {
  const contractActions = [
    "Create data contract for customer table",
    "Generate metadata for sales pipeline",
    "Create contract template",
    "Document existing data source"
  ];

  return (
    <div className="quick-actions">
      <p><strong>💡 Contract creation suggestions:</strong></p>
      <div className="quick-buttons">
        {contractActions.map((action, index) => (
          <button
            key={index}
            className="quick-action-btn contract-action"
            onClick={() => onActionClick(action)}
            disabled={disabled}
          >
            <FileText size={16} />
            {action.substring(0, 25)}...
          </button>
        ))}
      </div>
    </div>
  );
};

// Lineage Quick Actions (existing)
const LineageQuickActions = ({ onActionClick, disabled }) => {
  const lineageActions = [
    "trace customer_id lineage",
    "show available contracts",
    "upstream dependencies for order_total",
    "help me understand data lineage"
  ];

  return (
    <div className="quick-actions">
      <p><strong>💡 Lineage analysis suggestions:</strong></p>
      <div className="quick-buttons">
        {lineageActions.map((action, index) => (
          <button
            key={index}
            className="quick-action-btn lineage-action"
            onClick={() => onActionClick(action)}
            disabled={disabled}
          >
            <Database size={16} />
            {action.substring(0, 25)}...
          </button>
        ))}
      </div>
    </div>
  );
};

// Status Indicator Component
const StatusIndicator = ({ status, mode }) => {
  const statusConfig = {
    processing: { color: '#17a2b8', text: 'Processing...' },
    waiting: { color: '#6c757d', text: 'Waiting for input...' },
    ready: { color: '#28a745', text: 'Ready' },
    error: { color: '#dc3545', text: 'Error' }
  };

  const config = statusConfig[status] || statusConfig.ready;
  const modeColor = mode === CHAT_MODES.LINEAGE ? '#3b82f6' : '#10b981';

  return (
    <div className="status-indicator">
      <span
        className="status-dot"
        style={{ backgroundColor: config.color }}
      ></span>
      <div className="status-info">
        <span className="status-text">{config.text}</span>
        <small style={{ color: modeColor }}>
          {mode === CHAT_MODES.LINEAGE ? 'Lineage Mode' : 'Contract Mode'}
        </small>
      </div>
    </div>
  );
};

// Enhanced Message Component
const MessageComponent = ({ message, onFeedbackSelection }) => {
  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const renderMessageContent = () => {
    switch (message.messageType) {
      case MESSAGE_TYPES.MARKDOWN:
        return <MarkdownMessage content={message.content} metadata={message.metadata} />;
      
      case MESSAGE_TYPES.TABLE:
        return (
          <div className="table-message">
            <div className="table-header">
              <span>📊 Generated Table</span>
            </div>
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        );

      case MESSAGE_TYPES.VISUALIZATION:
        return (
          <>
            <div>📊 Here's the lineage analysis results for your query:</div>
            <LineageVisualization result={message.metadata.result} />
          </>
        );

      case MESSAGE_TYPES.FEEDBACK_REQUEST:
        return (
          <>
            <div>{message.content}</div>
            <FeedbackInterface
              feedbackData={message.metadata}
              onSelection={onFeedbackSelection}
            />
          </>
        );

      default:
        return <div>{message.content}</div>;
    }
  };

  return (
    <div className={`message ${message.role}-message`}>
      <div className="message-content">
        {renderMessageContent()}
      </div>
      <div className="message-timestamp">
        {formatTime(message.timestamp)}
      </div>
    </div>
  );
};

// Simplified Lineage Visualization (keeping existing structure)
const LineageVisualization = ({ result }) => {
  const [activeTab, setActiveTab] = useState('elements');
  const elements = result.nodes || result.data_elements || result.elements || [];
  const relationships = result.edges || result.relationships || [];

  return (
    <div className="visualization-container">
      <div className="metrics-row">
        <div className="metric-card">
          <h4>{elements.length || 0}</h4>
          <small>Data Elements</small>
        </div>
        <div className="metric-card">
          <h4>{relationships.length || 0}</h4>
          <small>Relationships</small>
        </div>
        <div className="metric-card">
          <h4>{result.complexity_score || 0}/10</h4>
          <small>Complexity</small>
        </div>
      </div>

      <div className="tabbed-content">
        <div className="tab-navigation">
          <button
            className={`tab-button ${activeTab === 'elements' ? 'active' : ''}`}
            onClick={() => setActiveTab('elements')}
          >
            📊 Elements ({elements.length})
          </button>
          <button
            className={`tab-button ${activeTab === 'relationships' ? 'active' : ''}`}
            onClick={() => setActiveTab('relationships')}
          >
            🔗 Relationships ({relationships.length})
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'elements' && (
            <div className="simple-table">
              <h4>📊 Data Elements</h4>
              {elements.length > 0 ? (
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Table</th>
                      <th>Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {elements.map((element, index) => (
                      <tr key={index}>
                        <td>{element.name || element.element_name || 'N/A'}</td>
                        <td>{element.table || element.table_name || 'N/A'}</td>
                        <td>
                          <span className="type-badge">
                            {element.type || 'Unknown'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p>No elements found</p>
              )}
            </div>
          )}

          {activeTab === 'relationships' && (
            <div className="simple-table">
              <h4>🔗 Relationships</h4>
              {relationships.length > 0 ? (
                <table>
                  <thead>
                    <tr>
                      <th>Source</th>
                      <th>Target</th>
                      <th>Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {relationships.map((rel, index) => (
                      <tr key={index}>
                        <td>{rel.source || rel.source_element || 'N/A'}</td>
                        <td>{rel.target || rel.target_element || 'N/A'}</td>
                        <td>
                          <span className="relationship-badge">
                            {rel.type || rel.relationship_type || 'Unknown'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p>No relationships found</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Feedback Interface Component (keeping existing)
const FeedbackInterface = ({ feedbackData, onSelection }) => {
  const queryResults = feedbackData.query_results || {};

  if (queryResults.available_elements) {
    return (
      <div className="feedback-interface">
        <p><strong>🎯 Please select a data element:</strong></p>
        <div className="feedback-buttons">
          {queryResults.available_elements.map((element, index) => (
            <button
              key={element.name}
              className="feedback-btn element-btn"
              onClick={() => onSelection({ selected_index: index })}
            >
              📊 {element.name}
            </button>
          ))}
        </div>
      </div>
    );
  }

  if (queryResults.available_contracts) {
    return (
      <div className="feedback-interface">
        <p><strong>📋 Please select a data contract:</strong></p>
        <div className="feedback-buttons">
          {queryResults.available_contracts.map((contract, index) => (
            <button
              key={contract.name}
              className="feedback-btn contract-btn"
              onClick={() => onSelection({ selected_index: index })}
            >
              📄 {contract.name}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return null;
};

// Main App Component
const DualChatApp = () => {
  const [currentMode, setCurrentMode] = useState(CHAT_MODES.LINEAGE);
  const [lineageMessages, setLineageMessages] = useState([
    new ChatMessage(
      'system',
      '🤖 Hello! I\'m your Data Lineage Assistant. I can help you trace data lineage, analyze contracts, and explore data relationships.',
      MESSAGE_TYPES.SYSTEM
    )
  ]);
  const [contractMessages, setContractMessages] = useState([
    new ChatMessage(
      'system',
      '🤖 Hello! I\'m your Contract Creation Assistant. I can help you create data contracts, generate metadata documentation, and build schema definitions.',
      MESSAGE_TYPES.SYSTEM
    )
  ]);

  const [inputValue, setInputValue] = useState('');
  const [processing, setProcessing] = useState(false);
  const [awaitingFeedback, setAwaitingFeedback] = useState(false);
  const [sessionId] = useState(() => Date.now().toString());
  const messagesEndRef = useRef(null);

  // API Base URL
  const API_BASE_URL = 'http://localhost:8000/api';

  // Get current messages based on mode
  const getCurrentMessages = () => {
    return currentMode === CHAT_MODES.LINEAGE ? lineageMessages : contractMessages;
  };

  const setCurrentMessages = (messages) => {
    if (currentMode === CHAT_MODES.LINEAGE) {
      setLineageMessages(messages);
    } else {
      setContractMessages(messages);
    }
  };

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [lineageMessages, contractMessages, currentMode]);

  // Handle mode change
  const handleModeChange = (newMode) => {
    if (processing) return;
    setCurrentMode(newMode);
    setAwaitingFeedback(false);
    setInputValue('');
  };

  // API Functions for Lineage
  const processLineageQuery = async (query) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/lineage/query`, {
        query: query,
        session_id: sessionId
      });
      return response.data;
    } catch (error) {
      console.error('Lineage API Error:', error);
      throw new Error('Failed to process lineage query. Please check your connection.');
    }
  };

  // API Functions for Contract Creation (mock implementation)
  const processContractQuery = async (query) => {
    // Simulate API call - replace with actual contract creation endpoint
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Mock contract creation response
    if (query.toLowerCase().includes('customer')) {
      return {
        success: true,
        result_type: 'markdown',
        content: `# Customer Data Contract

## Table: customers

### Description
This contract defines the structure and governance rules for the customer data table.

### Schema Definition

| Column Name | Data Type | Required | Description |
|-------------|-----------|----------|-------------|
| customer_id | INTEGER | Yes | Primary key, unique customer identifier |
| first_name | VARCHAR(50) | Yes | Customer's first name |
| last_name | VARCHAR(50) | Yes | Customer's last name |
| email | VARCHAR(100) | Yes | Customer's email address (unique) |
| phone | VARCHAR(20) | No | Customer's phone number |
| created_at | TIMESTAMP | Yes | Record creation timestamp |
| updated_at | TIMESTAMP | Yes | Last update timestamp |

### Data Quality Rules
- Email must be unique across all records
- Phone number format validation required
- Names cannot contain numeric characters
- All timestamps must be in UTC

### Data Lineage
- **Source Systems**: CRM System, Web Registration
- **Downstream Systems**: Analytics DB, Marketing Platform
- **Update Frequency**: Real-time via CDC

### Data Governance
- **Owner**: Customer Data Team
- **Steward**: John Smith (john.smith@company.com)
- **Classification**: PII - Restricted
- **Retention**: 7 years after account closure

### SLA
- **Availability**: 99.9%
- **Latency**: < 100ms for reads
- **Backup**: Daily snapshots, 30-day retention

Generated on: ${new Date().toISOString()}`,
        metadata: {
          title: 'Customer Data Contract',
          filename: 'customer_data_contract',
          type: 'data_contract'
        }
      };
    } else if (query.toLowerCase().includes('template')) {
      return {
        success: true,
        result_type: 'markdown',
        content: `# Data Contract Template

## Basic Information
- **Contract Name**: [Contract Name]
- **Version**: 1.0
- **Owner**: [Data Owner]
- **Created Date**: ${new Date().toLocaleDateString()}

## Table Definition
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| [field_name] | [data_type] | [Yes/No] | [Description] |

## Data Quality Rules
- [Rule 1]
- [Rule 2]
- [Rule 3]

## Governance
- **Classification**: [Public/Internal/Confidential/Restricted]
- **Retention Period**: [Time period]
- **Access Control**: [Access rules]

## SLA Requirements
- **Availability**: [Percentage]
- **Performance**: [Requirements]
- **Backup**: [Backup strategy]`,
        metadata: {
          title: 'Data Contract Template',
          filename: 'data_contract_template',
          type: 'template'
        }
      };
    } else {
      return {
        success: true,
        result_type: 'table',
        content: `## Generated Metadata Summary

| Attribute | Value |
|-----------|-------|
| Query Processed | ${query} |
| Processing Time | ${new Date().toLocaleTimeString()} |
| Status | Success |
| Type | Metadata Generation |

### Recommendations
- Define clear data quality rules
- Establish ownership and governance
- Document data lineage
- Set up monitoring and alerting`,
        metadata: {
          title: 'Metadata Summary',
          type: 'summary'
        }
      };
    }
  };

  // Message Handlers
  const addMessage = (message) => {
    const currentMessages = getCurrentMessages();
    setCurrentMessages([...currentMessages, message]);
  };

  const handleUserMessage = async (userInput) => {
    if (!userInput.trim() || processing) return;

    setProcessing(true);

    // Add user message
    const userMessage = new ChatMessage('user', userInput);
    addMessage(userMessage);

    try {
      let result;
      
      if (currentMode === CHAT_MODES.LINEAGE) {
        result = await processLineageQuery(userInput);
        
        if (result.human_input_required) {
          const feedbackMessage = new ChatMessage(
            'assistant',
            result.message || 'I need more information to help you...',
            MESSAGE_TYPES.FEEDBACK_REQUEST,
            result
          );
          addMessage(feedbackMessage);
          setAwaitingFeedback(true);
        } else if (result.error) {
          const errorMessage = new ChatMessage(
            'assistant',
            `❌ I encountered an issue: ${result.error}\n\nWould you like me to help you rephrase your query?`,
            MESSAGE_TYPES.ERROR
          );
          addMessage(errorMessage);
        } else {
          const successMessage = new ChatMessage(
            'assistant',
            '✅ Great! I found the lineage information you requested. Here\'s what I discovered:'
          );
          addMessage(successMessage);

          const vizMessage = new ChatMessage(
            'assistant',
            '',
            MESSAGE_TYPES.VISUALIZATION,
            { result }
          );
          addMessage(vizMessage);
        }
      } else {
        // Contract mode
        result = await processContractQuery(userInput);
        
        if (result.success) {
          const messageType = result.result_type === 'markdown' ? MESSAGE_TYPES.MARKDOWN : MESSAGE_TYPES.TABLE;
          const contractMessage = new ChatMessage(
            'assistant',
            result.content,
            messageType,
            result.metadata
          );
          addMessage(contractMessage);
        } else {
          const errorMessage = new ChatMessage(
            'assistant',
            `❌ Contract creation failed: ${result.error}`,
            MESSAGE_TYPES.ERROR
          );
          addMessage(errorMessage);
        }
      }
    } catch (error) {
      const errorMessage = new ChatMessage(
        'assistant',
        `❌ I'm sorry, something went wrong: ${error.message}\n\nPlease try rephrasing your question.`,
        MESSAGE_TYPES.ERROR
      );
      addMessage(errorMessage);
    } finally {
      setProcessing(false);
    }
  };

  const handleSend = () => {
    handleUserMessage(inputValue);
    setInputValue('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearCurrentChat = () => {
    const systemMessage = new ChatMessage(
      'system',
      currentMode === CHAT_MODES.LINEAGE 
        ? '🤖 Chat cleared! How can I help you with data lineage today?'
        : '🤖 Chat cleared! How can I help you create contracts and metadata today?',
      MESSAGE_TYPES.SYSTEM
    );
    setCurrentMessages([systemMessage]);
    setAwaitingFeedback(false);
  };

  const currentStatus = processing ? 'processing' : awaitingFeedback ? 'waiting' : 'ready';
  const currentMessages = getCurrentMessages();

  return (
    <div className="app">
      <style jsx>{`
        .app {
          height: 100vh;
          display: flex;
          flex-direction: column;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        }
        
        .app-header {
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(10px);
          padding: 1rem 2rem;
          border-bottom: 1px solid rgba(0, 0, 0, 0.1);
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .app-header h1 {
          margin: 0;
          background: linear-gradient(135deg, #667eea, #764ba2);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          font-size: 1.8rem;
          font-weight: 700;
        }
        
        .app-body {
          flex: 1;
          display: flex;
          gap: 1rem;
          padding: 1rem;
          overflow: hidden;
        }
        
        .sidebar {
          width: 300px;
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(10px);
          border-radius: 12px;
          padding: 1.5rem;
          overflow-y: auto;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .sidebar-section {
          margin-bottom: 2rem;
        }
        
        .sidebar-section h3 {
          margin: 0 0 1rem 0;
          color: #374151;
          font-size: 1rem;
          font-weight: 600;
        }
        
        .mode-selector {
          margin-bottom: 2rem;
        }
        
        .mode-buttons {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          margin-bottom: 1rem;
        }
        
        .mode-btn {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          padding: 1rem;
          border: 2px solid transparent;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.8);
          cursor: pointer;
          transition: all 0.2s ease;
          text-align: left;
        }
        
        .mode-btn:hover {
          background: rgba(255, 255, 255, 1);
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .mode-btn.active {
          border-color: #3b82f6;
          background: rgba(59, 130, 246, 0.1);
        }
        
        .mode-btn span {
          font-weight: 600;
          margin: 0.25rem 0;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .mode-btn small {
          color: #6b7280;
          font-size: 0.85rem;
        }
        
        .mode-indicator {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 0.75rem;
          background: rgba(255, 255, 255, 0.8);
          border-radius: 6px;
        }
        
        .indicator-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        
        .status-indicator {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.75rem;
          background: rgba(255, 255, 255, 0.8);
          border-radius: 8px;
        }
        
        .status-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }
        
        .status-info {
          display: flex;
          flex-direction: column;
        }
        
        .status-text {
          font-weight: 500;
          color: #374151;
        }
        
        .sidebar-btn {
          
          width: 100%;
          padding: 0.75rem 1rem;
          border: none;
          border-radius: 6px;
          background: linear-gradient(135deg, #667eea, #764ba2);
          color: white;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
          margin-bottom: 0.5rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .sidebar-btn:hover {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .chat-main {
          flex: 1;
          display: flex;
          flex-direction: column;
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(10px);
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .chat-container {
          flex: 1;
          overflow: hidden;
        }
        
        .messages-container {
          height: 100%;
          overflow-y: auto;
          padding: 1.5rem;
          scroll-behavior: smooth;
        }
        
        .message {
          margin-bottom: 1.5rem;
          display: flex;
          flex-direction: column;
        }
        
        .user-message {
          align-items: flex-end;
        }
        
        .assistant-message, .system-message {
          align-items: flex-start;
        }
        
        .message-content {
          max-width: 80%;
          padding: 1rem 1.25rem;
          border-radius: 18px;
          position: relative;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .user-message .message-content {
          background: linear-gradient(135deg, #667eea, #764ba2);
          color: white;
          border-bottom-right-radius: 4px;
        }
        
        .assistant-message .message-content {
          background: #f3f4f6;
          color: #374151;
          border-bottom-left-radius: 4px;
        }
        
        .system-message .message-content {
          background: rgba(16, 185, 129, 0.1);
          color: #065f46;
          border: 1px solid rgba(16, 185, 129, 0.2);
        }
        
        .message-timestamp {
          font-size: 0.75rem;
          color: #9ca3af;
          margin-top: 0.5rem;
          padding: 0 0.5rem;
        }
        
        .markdown-message {
          background: #ffffff;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .markdown-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem 1rem;
          background: #f9fafb;
          border-bottom: 1px solid #e5e7eb;
        }
        
        .markdown-title {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-weight: 600;
          color: #374151;
        }
        
        .markdown-actions {
          display: flex;
          gap: 0.5rem;
        }
        
        .action-btn {
          display: flex;
          align-items: center;
          gap: 0.25rem;
          padding: 0.25rem 0.5rem;
          border: 1px solid #d1d5db;
          border-radius: 4px;
          background: white;
          color: #6b7280;
          font-size: 0.75rem;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .action-btn:hover {
          background: #f3f4f6;
          border-color: #9ca3af;
          color: #374151;
        }
        
        .markdown-content {
          padding: 1rem;
          max-height: 400px;
          overflow-y: auto;
        }
        
        .markdown-content h1 {
          color: #1f2937;
          border-bottom: 2px solid #e5e7eb;
          padding-bottom: 0.5rem;
          margin-bottom: 1rem;
        }
        
        .markdown-content h2 {
          color: #374151;
          margin: 1.5rem 0 0.75rem 0;
        }
        
        .markdown-content h3 {
          color: #4b5563;
          margin: 1rem 0 0.5rem 0;
        }
        
        .markdown-content table {
          width: 100%;
          border-collapse: collapse;
          margin: 1rem 0;
          font-size: 0.9rem;
        }
        
        .markdown-content th,
        .markdown-content td {
          border: 1px solid #e5e7eb;
          padding: 0.5rem 0.75rem;
          text-align: left;
        }
        
        .markdown-content th {
          background: #f9fafb;
          font-weight: 600;
          color: #374151;
        }
        
        .markdown-content tbody tr:nth-child(even) {
          background: #f9fafb;
        }
        
        .markdown-content code {
          background: #f3f4f6;
          padding: 0.125rem 0.25rem;
          border-radius: 3px;
          font-size: 0.85em;
          color: #e11d48;
        }
        
        .markdown-content pre {
          background: #1f2937;
          color: #f9fafb;
          padding: 1rem;
          border-radius: 6px;
          overflow-x: auto;
          margin: 1rem 0;
        }
        
        .markdown-content ul, .markdown-content ol {
          margin: 0.5rem 0;
          padding-left: 1.5rem;
        }
        
        .markdown-content li {
          margin: 0.25rem 0;
        }
        
        .table-message {
          background: #ffffff;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          overflow: hidden;
        }
        
        .table-header {
          padding: 0.75rem 1rem;
          background: #f9fafb;
          border-bottom: 1px solid #e5e7eb;
          font-weight: 600;
          color: #374151;
        }
        
        .visualization-container {
          background: #ffffff;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 1rem;
          margin-top: 0.5rem;
        }
        
        .metrics-row {
          display: flex;
          gap: 1rem;
          margin-bottom: 1.5rem;
          flex-wrap: wrap;
        }
        
        .metric-card {
          flex: 1;
          min-width: 120px;
          padding: 1rem;
          background: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 6px;
          text-align: center;
        }
        
        .metric-card h4 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 700;
          color: #1f2937;
        }
        
        .metric-card small {
          color: #6b7280;
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        
        .tabbed-content {
          border: 1px solid #e5e7eb;
          border-radius: 6px;
          overflow: hidden;
        }
        
        .tab-navigation {
          display: flex;
          background: #f9fafb;
          border-bottom: 1px solid #e5e7eb;
        }
        
        .tab-button {
          flex: 1;
          padding: 0.75rem 1rem;
          border: none;
          background: transparent;
          color: #6b7280;
          cursor: pointer;
          transition: all 0.2s ease;
          font-weight: 500;
        }
        
        .tab-button:hover {
          background: #f3f4f6;
          color: #374151;
        }
        
        .tab-button.active {
          background: #ffffff;
          color: #3b82f6;
          border-bottom: 2px solid #3b82f6;
        }
        
        .tab-content {
          padding: 1rem;
        }
        
        .simple-table {
          overflow-x: auto;
        }
        
        .simple-table table {
          width: 100%;
          border-collapse: collapse;
          margin-top: 0.5rem;
        }
        
        .simple-table th,
        .simple-table td {
          border: 1px solid #e5e7eb;
          padding: 0.5rem 0.75rem;
          text-align: left;
        }
        
        .simple-table th {
          background: #f9fafb;
          font-weight: 600;
          color: #374151;
        }
        
        .simple-table tbody tr:nth-child(even) {
          background: #f9fafb;
        }
        
        .type-badge, .relationship-badge {
          display: inline-block;
          padding: 0.125rem 0.5rem;
          background: #dbeafe;
          color: #1e40af;
          border-radius: 12px;
          font-size: 0.75rem;
          font-weight: 500;
        }
        
        .feedback-interface {
          margin-top: 1rem;
          padding: 1rem;
          background: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 6px;
        }
        
        .feedback-buttons {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin-top: 0.75rem;
        }
        
        .feedback-btn {
          padding: 0.5rem 1rem;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          background: white;
          color: #374151;
          cursor: pointer;
          transition: all 0.2s ease;
          font-weight: 500;
        }
        
        .feedback-btn:hover {
          background: #f3f4f6;
          border-color: #9ca3af;
        }
        
        .input-container {
          padding: 1.5rem;
          border-top: 1px solid #e5e7eb;
          background: rgba(255, 255, 255, 0.95);
        }
        
        .quick-actions {
          margin-bottom: 1rem;
        }
        
        .quick-buttons {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin-top: 0.5rem;
        }
        
        .quick-action-btn {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 0.75rem;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          background: white;
          color: #6b7280;
          cursor: pointer;
          transition: all 0.2s ease;
          font-size: 0.875rem;
        }
        
        .quick-action-btn:hover:not(:disabled) {
          background: #f3f4f6;
          border-color: #9ca3af;
          color: #374151;
        }
        
        .quick-action-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .lineage-action:hover:not(:disabled) {
          border-color: #3b82f6;
          color: #3b82f6;
        }
        
        .contract-action:hover:not(:disabled) {
          border-color: #10b981;
          color: #10b981;
        }
        
        .input-row {
          display: flex;
          gap: 0.75rem;
          align-items: center;
        }
        
        .chat-input {
          flex: 1;
          padding: 0.875rem 1rem;
          border: 2px solid #e5e7eb;
          border-radius: 24px;
          background: white;
          font-size: 1rem;
          transition: all 0.2s ease;
          outline: none;
        }
        
        .chat-input:focus {
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        .chat-input:disabled {
          background: #f9fafb;
          color: #9ca3af;
          cursor: not-allowed;
        }
        
        .send-button {
          padding: 0.875rem 1.5rem;
          border: none;
          border-radius: 24px;
          background: linear-gradient(135deg, #667eea, #764ba2);
          color: white;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .send-button:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .send-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }
        
        .typing-indicator {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 1rem;
          color: #6b7280;
          font-style: italic;
        }
        
        .typing-dots {
          display: flex;
          gap: 0.25rem;
        }
        
        .typing-dot {
          width: 6px;
          height: 6px;
          background: #9ca3af;
          border-radius: 50%;
          animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typing {
          0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.5;
          }
          40% {
            transform: scale(1);
            opacity: 1;
          }
        }
        
        .stat {
          display: flex;
          justify-content: space-between;
          margin-bottom: 0.5rem;
          font-size: 0.875rem;
        }
        
        .stat-label {
          color: #6b7280;
        }
        
        .stat-value {
          font-weight: 600;
          color: #374151;
        }
        
        details {
          margin-top: 1rem;
        }
        
        summary {
          cursor: pointer;
          font-weight: 600;
          color: #374151;
          padding: 0.5rem 0;
        }
        
        summary:hover {
          color: #1f2937;
        }
        
        .help-content {
          padding: 1rem 0;
          color: #6b7280;
        }
        
        .help-content p {
          margin: 0.5rem 0;
        }
        
        .help-content ul {
          margin: 0.5rem 0;
          padding-left: 1.25rem;
        }
        
        .help-content li {
          margin: 0.25rem 0;
          font-size: 0.875rem;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
          .app-body {
            flex-direction: column;
          }
          
          .sidebar {
            width: 100%;
            order: 2;
            max-height: 300px;
          }
          
          .chat-main {
            order: 1;
            height: 60vh;
          }
          
          .mode-buttons {
            flex-direction: row;
          }
          
          .metrics-row {
            flex-direction: column;
          }
          
          .message-content {
            max-width: 95%;
          }
        }
      `}</style>

      {/* Header */}
      <header className="app-header">
        <h1>🤖 Smart Data Assistant - Dual Mode Interface</h1>
      </header>

      <div className="app-body">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <h3>🎛️ Mode Selection</h3>
            <ModeSelector 
              currentMode={currentMode}
              onModeChange={handleModeChange}
              disabled={processing}
            />
          </div>

          <div className="sidebar-section">
            <h3>📊 Status</h3>
            <StatusIndicator status={currentStatus} mode={currentMode} />
          </div>

          <div className="sidebar-section">
            <h3>📈 Chat Stats</h3>
            <div className="stat">
              <span className="stat-label">Current Messages:</span>
              <span className="stat-value">{currentMessages.length}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Lineage Messages:</span>
              <span className="stat-value">{lineageMessages.length}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Contract Messages:</span>
              <span className="stat-value">{contractMessages.length}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Session ID:</span>
              <span className="stat-value">{sessionId.slice(0, 8)}...</span>
            </div>
          </div>

          <div className="sidebar-section">
            <h3>⚡ Actions</h3>
            <button className="sidebar-btn" onClick={clearCurrentChat}>
              <RefreshCw size={16} />
              Clear {currentMode === CHAT_MODES.LINEAGE ? 'Lineage' : 'Contract'} Chat
            </button>
          </div>

          <details className="sidebar-section">
            <summary>❓ Help & Examples</summary>
            <div className="help-content">
              {currentMode === CHAT_MODES.LINEAGE ? (
                <>
                  <p><strong>Lineage Examples:</strong></p>
                  <ul>
                    <li>trace customer_id lineage</li>
                    <li>show Customer Data Pipeline</li>
                    <li>upstream deps for order_total</li>
                    <li>bidirectional lineage for email</li>
                  </ul>
                </>
              ) : (
                <>
                  <p><strong>Contract Examples:</strong></p>
                  <ul>
                    <li>Create data contract for customer table</li>
                    <li>Generate metadata template</li>
                    <li>Document sales pipeline schema</li>
                    <li>Create governance rules for PII data</li>
                  </ul>
                </>
              )}
              <p><strong>Tips:</strong></p>
              <ul>
                <li>Be specific about your requirements</li>
                <li>Use natural language</li>
                <li>Ask follow-up questions</li>
                <li>Switch modes anytime</li>
              </ul>
            </div>
          </details>
        </aside>

        {/* Main Chat Area */}
        <main className="chat-main">
          <div className="chat-container">
            <div className="messages-container">
              {currentMessages.map(message => (
                <MessageComponent
                  key={message.id}
                  message={message}
                  onFeedbackSelection={() => {}} // Placeholder for lineage feedback
                />
              ))}
              {processing && (
                <div className="typing-indicator">
                  <div className="typing-dots">
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                  </div>
                  <span>
                    {currentMode === CHAT_MODES.LINEAGE 
                      ? 'Analyzing lineage...' 
                      : 'Generating contract...'}
                  </span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area */}
          <div className="input-container">
            {!awaitingFeedback && (
              currentMode === CHAT_MODES.LINEAGE ? (
                <LineageQuickActions
                  onActionClick={handleUserMessage}
                  disabled={processing}
                />
              ) : (
                <ContractQuickActions
                  onActionClick={handleUserMessage}
                  disabled={processing}
                />
              )
            )}

            <div className="input-row">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  currentMode === CHAT_MODES.LINEAGE
                    ? "Ask me about data lineage, contracts, or element tracing..."
                    : "Ask me to create contracts, generate metadata, or document schemas..."
                }
                disabled={processing || awaitingFeedback}
                className="chat-input"
              />
              <button
                onClick={handleSend}
                disabled={processing || awaitingFeedback || !inputValue.trim()}
                className="send-button"
              >
                {currentMode === CHAT_MODES.LINEAGE ? (
                  <>
                    <Database size={16} />
                    Analyze
                  </>
                ) : (
                  <>
                    <FileText size={16} />
                    Generate
                  </>
                )}
              </button>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default DualChatApp;