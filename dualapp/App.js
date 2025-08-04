// App.js - Main React Component with Tabular Results
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import ReactMarkdown from 'react-markdown';
import { Copy, Download, RefreshCw, Database, FileText, MessageSquare, Settings, BotMessageSquare } from 'lucide-react';

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
          <Database size={20}/>
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

// Typing Indicator Component
const TypingIndicator = () => (
  <div className="typing-indicator">
    <div className="typing-dots">
      <div className="typing-dot"></div>
      <div className="typing-dot"></div>
      <div className="typing-dot"></div>
    </div>
    <span>Assistant is thinking...</span>
  </div>
);

// Data Elements Table Component
const DataElementsTable = ({ elements }) => {
  if (!elements || elements.length === 0) {
    return (
      <div className="no-data-message">
        <p>No data elements found</p>
      </div>
    );
  }

  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>
            <th>Element Name</th>
            <th>Element Code</th>
            <th>Table</th>
            <th>Type</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {elements.map((element, index) => (
            <tr key={index}>
              <td className="element-name">{element.name || element.element_name || 'N/A'}</td>
              <td className="element-code">{element.code || element.element_code || 'N/A'}</td>
              <td className="table-name">{element.table || element.table_name || 'N/A'}</td>
              <td className="element-type">
                <span className={`type-badge ${element.type || 'default'}`}>
                  {element.type || 'Unknown'}
                </span>
              </td>
              <td className="description">{element.description || 'No description available'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// Relationships Table Component
const RelationshipsTable = ({ relationships }) => {
  if (!relationships || relationships.length === 0) {
    return (
      <div className="no-data-message">
        <p>No relationships found</p>
      </div>
    );
  }

  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>
            <th>Source</th>
            <th>Target</th>
            <th>Relationship Type</th>
            <th>Direction</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {relationships.map((rel, index) => (
            <tr key={index}>
              <td className="source-element">{rel.source || rel.source_element || 'N/A'}</td>
              <td className="target-element">{rel.target || rel.target_element || 'N/A'}</td>
              <td className="relationship-type">
                <span className={`relationship-badge ${rel.type || 'default'}`}>
                  {rel.type || rel.relationship_type || 'Unknown'}
                </span>
              </td>
              <td className="direction">
                <span className="direction-indicator">
                  {rel.direction === 'bidirectional' ? '↔️' : '→'}
                  {rel.direction || 'unidirectional'}
                </span>
              </td>
              <td className="confidence">
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${(rel.confidence || 0.5) * 100}%` }}
                  ></div>
                  <span className="confidence-text">
                    {Math.round((rel.confidence || 0.5) * 100)}%
                  </span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// Transformations Table Component
const TransformationsTable = ({ transformations }) => {
  if (!transformations || transformations.length === 0) {
    return (
      <div className="no-data-message">
        <p>No transformations found</p>
      </div>
    );
  }

  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>
            <th>Transformation Name</th>
            <th>Input Elements</th>
            <th>Output Elements</th>
            <th>Operation Type</th>
            <th>Business Logic</th>
          </tr>
        </thead>
        <tbody>
          {transformations.map((transform, index) => (
            <tr key={index}>
              <td className="transform-name">{transform.name || transform.transformation_name || 'N/A'}</td>
              <td className="input-elements">
                <div className="element-list">
                  {Array.isArray(transform.inputs) ?
                    transform.inputs.map((input, i) => (
                      <span key={i} className="element-tag input-tag">{input}</span>
                    )) :
                    <span className="element-tag input-tag">{transform.inputs || transform.input_elements || 'N/A'}</span>
                  }
                </div>
              </td>
              <td className="output-elements">
                <div className="element-list">
                  {Array.isArray(transform.outputs) ?
                    transform.outputs.map((output, i) => (
                      <span key={i} className="element-tag output-tag">{output}</span>
                    )) :
                    <span className="element-tag output-tag">{transform.outputs || transform.output_elements || 'N/A'}</span>
                  }
                </div>
              </td>
              <td className="operation-type">
                <span className={`operation-badge ${transform.operation_type || 'default'}`}>
                  {transform.operation_type || transform.type || 'Unknown'}
                </span>
              </td>
              <td className="business-logic">
                <div className="logic-description">
                  {transform.business_logic || transform.description || 'No logic description available'}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// Updated Lineage Visualization Component with Enhanced Executive Summary


// Enhanced Executive Summary Formatter Component (inline for convenience)
const ExecutiveSummaryFormatter = ({ summary }) => {
  // Function to detect and format different types of content
  const formatSummaryContent = (text) => {
    if (!text) return null;

    // Split text into paragraphs
    const paragraphs = text.split('\n\n').filter(p => p.trim());

    return paragraphs.map((paragraph, index) => {
      // Check if paragraph contains bullet-like patterns
      if (paragraph.includes('•') || paragraph.includes('-') || paragraph.includes('*') ||
          paragraph.match(/^\s*\d+\./) || paragraph.includes('\n-') || paragraph.includes('\n•')) {
        return formatBulletPoints(paragraph, index);
      }

      // Check if it's a section header (contains colon and is relatively short)
      if (paragraph.includes(':') && paragraph.length < 150 && !paragraph.includes('.')) {
        return formatSectionHeader(paragraph, index);
      }

      // Regular paragraph with inline highlighting
      return formatRegularParagraph(paragraph, index);
    });
  };

  // Format bullet points
  const formatBulletPoints = (text, key) => {
    const lines = text.split('\n').filter(line => line.trim());

    return (
      <div key={key} className="summary-bullet-section">
        {lines.map((line, lineIndex) => {
          // Remove bullet markers and format content
          const cleanLine = line.replace(/^[\s]*[\d]*[•\-\*\.]\s*/, '').trim();
          if (!cleanLine) return null;

          return (
            <div key={lineIndex} className="summary-bullet-item">
              <span className="bullet-icon">🔹</span>
              <span className="bullet-content">
                {highlightInlineContent(cleanLine)}
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  // Format section headers
  const formatSectionHeader = (text, key) => {
    const colonIndex = text.indexOf(':');
    const header = text.substring(0, colonIndex).trim();
    const content = text.substring(colonIndex + 1).trim();

    return (
      <div key={key} className="summary-section-header">
        <h4 className="section-title">
          <span className="section-icon">📋</span>
          {header}
        </h4>
        {content && (
          <p className="section-content">
            {highlightInlineContent(content)}
          </p>
        )}
      </div>
    );
  };

  // Format regular paragraphs
  const formatRegularParagraph = (text, key) => {
    return (
      <p key={key} className="summary-paragraph">
        {highlightInlineContent(text)}
      </p>
    );
  };

  // Highlight important content within text
  const highlightInlineContent = (text) => {
    // Define patterns to highlight with priority order
    const patterns = [
      // Table names (ALL_CAPS or PascalCase)
      { regex: /\b[A-Z][A-Z_]{2,}[A-Z]?\b/g, className: 'table-name', priority: 1 },
      // Element codes (usually with underscores)
      { regex: /\b[a-zA-Z]+_[a-zA-Z_]+\b/g, className: 'element-code', priority: 2 },
      // Quoted strings (data elements)
      { regex: /'([^']+)'/g, className: 'data-element', captureGroup: 1, priority: 3 },
      { regex: /"([^"]+)"/g, className: 'data-element', captureGroup: 1, priority: 3 },
      // Numbers with units or percentages
      { regex: /\b\d+(\.\d+)?%\b/g, className: 'metric-highlight', priority: 4 },
      { regex: /\b\d+\s*(elements?|relationships?|transformations?|tables?|records?|rows?)\b/gi, className: 'metric-highlight', priority: 4 },
      // Relationship types
      { regex: /\b(upstream|downstream|bidirectional|depends on|feeds into|derived from|sources from)\b/gi, className: 'relationship-type', priority: 5 },
      // Operation types
      { regex: /\b(transformation|aggregation|filtering|joining|calculation|mapping|validation)\b/gi, className: 'transformation', priority: 6 },
    ];

    let components = [];
    let processedText = text;
    let offset = 0;

    // Find all matches with their positions
    const allMatches = [];
    patterns.forEach(pattern => {
      let match;
      const regex = new RegExp(pattern.regex.source, pattern.regex.flags);
      while ((match = regex.exec(text)) !== null) {
        allMatches.push({
          start: match.index,
          end: match.index + match[0].length,
          text: pattern.captureGroup ? match[pattern.captureGroup] : match[0],
          originalText: match[0],
          className: pattern.className,
          priority: pattern.priority
        });
      }
    });

    // Sort by position, then by priority for overlapping matches
    allMatches.sort((a, b) => {
      if (a.start !== b.start) return a.start - b.start;
      return a.priority - b.priority;
    });

    // Remove overlapping matches (keep higher priority)
    const cleanMatches = [];
    allMatches.forEach(match => {
      const hasOverlap = cleanMatches.some(existing =>
        (match.start < existing.end && match.end > existing.start)
      );
      if (!hasOverlap) {
        cleanMatches.push(match);
      }
    });

    // If no matches found, return original text
    if (cleanMatches.length === 0) {
      return text;
    }

    // Build components with highlighted text
    let lastIndex = 0;
    cleanMatches.forEach((match, index) => {
      // Add text before the match
      if (match.start > lastIndex) {
        components.push(text.substring(lastIndex, match.start));
      }

      // Add highlighted match
      components.push(
        <span key={`highlight-${index}`} className={match.className}>
          {match.text}
        </span>
      );

      lastIndex = match.end;
    });

    // Add remaining text
    if (lastIndex < text.length) {
      components.push(text.substring(lastIndex));
    }

    return components;
  };

  return (
    <div className="enhanced-summary-content">
      {formatSummaryContent(summary)}
    </div>
  );
};

// Updated Lineage Visualization Component
const LineageVisualization = ({ result }) => {
  const [activeTab, setActiveTab] = useState('elements');

  // Extract data from result
  const elements = result.nodes || result.data_elements || result.elements || [];
  const relationships = result.edges || result.relationships || [];
  const transformations = result.transformations || [];

  return (
    <div className="visualization-container">
      {/* Metrics Row */}
      <div className="metrics-row">
        <div className="metric-card">
          <h6>{result.lineage_type?.replace('_', ' ').toUpperCase() || 'ANALYSIS'}</h6>
          <small>Analysis Type</small>
        </div>
        <div className="metric-card">
          <h4>{elements.length || 0}</h4>
          <small>Data Elements</small>
        </div>
        <div className="metric-card">
          <h4>{relationships.length || 0}</h4>
          <small>Relationships</small>
        </div>
        <div className="metric-card">
          <h4>{transformations.length || 0}</h4>
          <small>Transformations</small>
        </div>
        <div className="metric-card">
          <h4>{result.complexity_score || 'N/A'}</h4>
          <small>Complexity Score</small>
        </div>
      </div>

      {/* Enhanced Executive Summary */}
      {result.executive_summary && (
        <details className="summary-section" open>
          <summary>📋 Executive Summary</summary>
          <ExecutiveSummaryFormatter summary={result.executive_summary} />
        </details>
      )}

      {/* Tabbed Content */}
      <div className="tabbed-content">
        <div className="tab-navigation">
          <button
            className={`tab-button ${activeTab === 'elements' ? 'active' : ''}`}
            onClick={() => setActiveTab('elements')}
          >
            📊 Data Elements ({elements.length})
          </button>
          <button
            className={`tab-button ${activeTab === 'relationships' ? 'active' : ''}`}
            onClick={() => setActiveTab('relationships')}
          >
            🔗 Relationships ({relationships.length})
          </button>
          <button
            className={`tab-button ${activeTab === 'transformations' ? 'active' : ''}`}
            onClick={() => setActiveTab('transformations')}
          >
            ⚙️ Transformations ({transformations.length})
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'elements' && (
            <div className="tab-panel">
              <h4>📊 Data Elements</h4>
              <DataElementsTable elements={elements} />
            </div>
          )}

          {activeTab === 'relationships' && (
            <div className="tab-panel">
              <h4>🔗 Data Relationships</h4>
              <RelationshipsTable relationships={relationships} />
            </div>
          )}

          {activeTab === 'transformations' && (
            <div className="tab-panel">
              <h4>⚙️ Data Transformations</h4>
              <TransformationsTable transformations={transformations} />
            </div>
          )}
        </div>
      </div>

      {/* Additional Analysis Details */}
      {result.analysis_details && (
        <details className="analysis-section">
          <summary>🔍 Technical Analysis Details</summary>
          <div className="analysis-content">
            <pre>{JSON.stringify(result.analysis_details, null, 2)}</pre>
          </div>
        </details>
      )}
    </div>
  );
};

//export default LineageVisualization;

// Feedback Interface Component
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

  if (queryResults.ambiguous_elements) {
    return (
      <div className="feedback-interface">
        <p><strong>🔍 Multiple matches found. Please choose:</strong></p>
        <div className="feedback-buttons vertical">
          {queryResults.ambiguous_elements.map((element, index) => (
            <button
              key={element.element_code}
              className="feedback-btn ambiguous-btn"
              onClick={() => onSelection({ selected_index: index })}
            >
              🎯 {element.element_name} (Code: {element.element_code})
            </button>
          ))}
        </div>
      </div>
    );
  }

  return null;
};

// Message Component
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

// Quick Actions Component
const QuickActions = ({ onActionClick, disabled }) => {
  const getQuickActions = () => [
    "trace customer_id lineage",
    "show available contracts",
    "upstream dependencies for order_total",
    "help me understand data lineage"
  ];

  return (
    <div className="quick-actions">
      <p><strong>💡 Quick suggestions:</strong></p>
      <div className="quick-buttons">
        {getQuickActions().map((action, index) => (
          <button
            key={index}
            className="quick-action-btn"
            onClick={() => onActionClick(action)}
            disabled={disabled}
          >
            💬 {action.substring(0, 20)}...
          </button>
        ))}
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
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

  // API Base URL - Update this to match your backend
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
  // Message Handlers
  const addMessage = (message) => {
    const currentMessages = getCurrentMessages();
    setCurrentMessages([...currentMessages, message]);
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

  const resumeWithFeedback = async (feedback) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/lineage/feedback`, {
        feedback: feedback,
        session_id: sessionId
      });
      return response.data;
    } catch (error) {
      console.error('API Error:', error);
      throw new Error('Failed to process feedback. Please try again.');
    }
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
          // Need human feedback
          const feedbackMessage = new ChatMessage(
              'assistant',
              result.message || 'I need more information to help you...',
              MESSAGE_TYPES.FEEDBACK_REQUEST,
              result
          );
          addMessage(feedbackMessage);
          setAwaitingFeedback(true);
        } else if (result.error) {
          // Error occurred
          const errorMessage = new ChatMessage(
              'assistant',
              `❌ I encountered an issue: ${result.error}\n\nWould you like me to help you rephrase your query?`,
              MESSAGE_TYPES.ERROR
          );
          addMessage(errorMessage);
        } else {
          // Success - show results
          const successMessage = new ChatMessage(
              'assistant',
              '✅ Great! I found the lineage information you requested. Here\'s what I discovered:'
          );
          addMessage(successMessage);

          const vizMessage = new ChatMessage(
              'assistant',
              '',
              MESSAGE_TYPES.VISUALIZATION,
              {result}
          );
          addMessage(vizMessage);
        }
      }
    else {
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
    }
    catch (error) {
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

  const handleFeedbackSelection = async (feedback) => {
    setAwaitingFeedback(false);

    // Add user selection message
    const selectionMessage = new ChatMessage(
      'user',
      `✅ Selected option ${feedback.selected_index + 1}`
    );
    addMessage(selectionMessage);

    try {
      const result = await resumeWithFeedback(feedback);

      if (result.human_input_required) {
        // Still need more input
        const feedbackMessage = new ChatMessage(
          'assistant',
          result.message || 'I need more information...',
          MESSAGE_TYPES.FEEDBACK_REQUEST,
          result
        );
        addMessage(feedbackMessage);
        setAwaitingFeedback(true);
      } else if (result.error) {
        const errorMessage = new ChatMessage(
          'assistant',
          `❌ ${result.error}`,
          MESSAGE_TYPES.ERROR
        );
        addMessage(errorMessage);
      } else {
        // Success - add visualization
        const vizMessage = new ChatMessage(
          'assistant',
          '',
          MESSAGE_TYPES.VISUALIZATION,
          { result }
        );
        addMessage(vizMessage);
      }
    } catch (error) {
      const errorMessage = new ChatMessage(
        'assistant',
        `❌ Error processing selection: ${error.message}`,
        MESSAGE_TYPES.ERROR
      );
      addMessage(errorMessage);
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

  const exportChat = () => {
    const chatExport = currentMessages.map(msg => ({
      timestamp: msg.timestamp.toISOString(),
      role: msg.role,
      content: msg.content,
      type: msg.messageType
    }));

    const dataStr = JSON.stringify(chatExport, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `lineage_chat_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };



  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1> Finance Metadata Discovery Assistant</h1>
      </header>

      <div className="app-body">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <h3>🤖 Agent Selection</h3>
            <ModeSelector
              currentMode={currentMode}
              onModeChange={handleModeChange}
              disabled={processing}
            />
          </div>

          <div className="sidebar-section">
            <h3>📊 Agent Status</h3>
            <StatusIndicator status={currentStatus} mode={currentMode} />
          </div>

          <div className="sidebar-section">
          <h3>💬 Chat Stats</h3>
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
                  <button className="sidebar-btn export-btn" onClick={exportChat}>
              💾 Export Chat
            </button>
          </div>
          <details className="sidebar-section">
            <summary>❓ Help & Examples</summary>
            <div className="help-content">
              <p><strong>Example Queries:</strong></p>
              <ul>
                <li>trace customer_id lineage</li>
                <li>show Customer Data Pipeline</li>
                <li>upstream deps for order_total</li>
                <li>bidirectional lineage for email</li>
              </ul>
              <p><strong>Tips:</strong></p>
              <ul>
                <li>Be specific about data elements</li>
                <li>Use natural language</li>
                <li>Ask follow-up questions</li>
                <li>Request clarifications anytime</li>
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
                  onFeedbackSelection={handleFeedbackSelection}
                />
              ))}
              {processing && <TypingIndicator />}
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
                placeholder="Ask me about data lineage, contracts, or element tracing..."
                disabled={processing || awaitingFeedback}
                className="chat-input"
              />
              <button
                onClick={handleSend}
                disabled={processing || awaitingFeedback || !inputValue.trim()}
                className="send-button"
              >
                🚀 Send
              </button>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;