// App.js - Main React Component with Tabular Results
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// Message Types
const MESSAGE_TYPES = {
  TEXT: 'text',
  VISUALIZATION: 'visualization',
  FEEDBACK_REQUEST: 'feedback_request',
  ERROR: 'error',
  SYSTEM: 'system'
};

// Chat Message Class
class ChatMessage {
  constructor(role, content, messageType = MESSAGE_TYPES.TEXT, metadata = {}) {
    this.id = Date.now() + Math.random();
    this.role = role; // 'user', 'assistant', 'system'
    this.content = content;
    this.timestamp = new Date();
    this.messageType = messageType;
    this.metadata = metadata;
  }
}

// Status Indicator Component
const StatusIndicator = ({ status }) => {
  const statusConfig = {
    processing: { color: '#17a2b8', text: 'Processing...' },
    waiting: { color: '#6c757d', text: 'Waiting for input...' },
    ready: { color: '#28a745', text: 'Ready' },
    error: { color: '#dc3545', text: 'Error' }
  };

  const config = statusConfig[status] || statusConfig.ready;

  return (
    <div className="status-indicator">
      <span
        className="status-dot"
        style={{ backgroundColor: config.color }}
      ></span>
      {config.text}
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

// Lineage Visualization Component (Updated without Network Graph)
const LineageVisualization = ({ result }) => {
  const [activeTab, setActiveTab] = useState('elements');

  // Extract data from result
  const elements = result.nodes || result.data_elements || result.elements || [];
  const relationships = result.edges || result.relationships || [];
  const transformations = result.transformations || [];

  return (
    <div className="visualization-container">
      {/* Metrics */}
      <div className="metrics-row">
        <div className="metric-card">
          <h6>{result.lineage_type?.replace('_', ' ').toUpperCase() || 'N/A'}</h6>
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
          <h4>{result.complexity_score || 0}/10</h4>
          <small>Complexity</small>
        </div>
      </div>

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

      {/* Executive Summary */}
      {result.executive_summary && (
        <details className="summary-section" open>
          <summary>📋 Executive Summary</summary>
          <div className="summary-content">
            {result.executive_summary}
          </div>
        </details>
      )}

      {/* Additional Analysis Details */}
      {result.analysis_details && (
        <details className="analysis-section">
          <summary>🔍 Analysis Details</summary>
          <div className="analysis-content">
            <pre>{JSON.stringify(result.analysis_details, null, 2)}</pre>
          </div>
        </details>
      )}
    </div>
  );
};

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
  const [messages, setMessages] = useState([
    new ChatMessage(
      'system',
      '🤖 Hello! I\'m your Data Lineage Assistant. I can help you trace data lineage, analyze contracts, and explore data relationships. What would you like to explore today?',
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

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // API Functions
  const processQuery = async (query) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/lineage/query`, {
        query: query,
        session_id: sessionId
      });
      return response.data;
    } catch (error) {
      console.error('API Error:', error);
      throw new Error('Failed to process query. Please check your connection.');
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

  // Message Handlers
  const addMessage = (message) => {
    setMessages(prev => [...prev, message]);
  };

  const handleUserMessage = async (userInput) => {
    if (!userInput.trim() || processing) return;

    setProcessing(true);

    // Add user message
    const userMessage = new ChatMessage('user', userInput);
    addMessage(userMessage);

    try {
      const result = await processQuery(userInput);

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
          { result }
        );
        addMessage(vizMessage);
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

  const clearChat = () => {
    setMessages([
      new ChatMessage(
        'system',
        '🤖 Chat cleared! How can I help you with data lineage today?',
        MESSAGE_TYPES.SYSTEM
      )
    ]);
    setAwaitingFeedback(false);
  };

  const exportChat = () => {
    const chatExport = messages.map(msg => ({
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

  const currentStatus = processing ? 'processing' : awaitingFeedback ? 'waiting' : 'ready';

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1>🤖 Data Lineage Chat Assistant</h1>
      </header>

      <div className="app-body">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <h3>🎛️ Chat Controls</h3>
            <StatusIndicator status={currentStatus} />
          </div>

          <div className="sidebar-section">
            <h3>📊 Session Stats</h3>
            <div className="stat">
              <span className="stat-label">Messages:</span>
              <span className="stat-value">{messages.length}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Session ID:</span>
              <span className="stat-value">{sessionId.slice(0, 8)}...</span>
            </div>
          </div>

          <div className="sidebar-section">
            <h3>⚡ Quick Actions</h3>
            <button className="sidebar-btn clear-btn" onClick={clearChat}>
              🗑️ Clear Chat
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
              {messages.map(message => (
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
              <QuickActions
                onActionClick={handleUserMessage}
                disabled={processing}
              />
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