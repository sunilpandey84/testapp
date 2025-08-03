// App.js - Main React Component
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';


const CHAT_MODES = {
  LINEAGE: 'lineage',
  CONTRACT: 'contract'
};

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

// Network Graph Component
const NetworkGraph = ({ nodes, edges }) => {
  if (!nodes || !edges || nodes.length === 0 || edges.length === 0) {
    return (
      <div className="no-data-message">
        <p>No data to visualize</p>
      </div>
    );
  }

  // Create layout positions (simplified spring layout)
  const positions = {};
  nodes.forEach((node, index) => {
    const angle = (2 * Math.PI * index) / nodes.length;
    const radius = Math.min(nodes.length * 20, 200);
    positions[node.id] = {
      x: radius * Math.cos(angle),
      y: radius * Math.sin(angle)
    };
  });

  // Prepare edge traces
  const edgeX = [];
  const edgeY = [];

  edges.forEach(edge => {
    if (positions[edge.source] && positions[edge.target]) {
      edgeX.push(positions[edge.source].x, positions[edge.target].x, null);
      edgeY.push(positions[edge.source].y, positions[edge.target].y, null);
    }
  });

  // Prepare node traces
  const nodeX = nodes.map(node => positions[node.id]?.x || 0);
  const nodeY = nodes.map(node => positions[node.id]?.y || 0);
  const nodeColors = nodes.map(node =>
    node.type === 'source' ? 'lightblue' : 'lightcoral'
  );
  const nodeText = nodes.map(node => node.name);
  const hoverText = nodes.map(node => `${node.name}<br>Table: ${node.table}`);

  const data = [
    {
      x: edgeX,
      y: edgeY,
      mode: 'lines',
      line: { width: 2, color: 'gray' },
      hoverinfo: 'none',
      type: 'scatter'
    },
    {
      x: nodeX,
      y: nodeY,
      mode: 'markers+text',
      marker: {
        size: 15,
        color: nodeColors,
        line: { width: 2, color: 'black' }
      },
      text: nodeText,
      textposition: 'middle center',
      hovertext: hoverText,
      hoverinfo: 'text',
      type: 'scatter'
    }
  ];

  const layout = {
    showlegend: false,
    hovermode: 'closest',
    margin: { b: 20, l: 5, r: 5, t: 40 },
    xaxis: { showgrid: false, zeroline: false, showticklabels: false },
    yaxis: { showgrid: false, zeroline: false, showticklabels: false },
    height: 400,
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)'
  };

  return (
    <div className="network-graph">
      <Plot data={data} layout={layout} style={{ width: '100%' }} />
    </div>
  );
};

// Lineage Visualization Component
const LineageVisualization = ({ result }) => {
  return (
    <div className="visualization-container">
      {/* Metrics */}
      <div className="metrics-row">
        <div className="metric-card">
          <h4>{result.lineage_type?.replace('_', ' ').toUpperCase() || 'N/A'}</h4>
          <small>Analysis Type</small>
        </div>
        <div className="metric-card">
          <h4>{result.nodes?.length || 0}</h4>
          <small>Data Elements</small>
        </div>
        <div className="metric-card">
          <h4>{result.edges?.length || 0}</h4>
          <small>Relationships</small>
        </div>
        <div className="metric-card">
          <h4>{result.complexity_score || 0}/10</h4>
          <small>Complexity</small>
        </div>
      </div>

      {/* Network Visualization */}
      {result.nodes && result.edges && (
        <div className="network-section">
          <h4>🕸️ Lineage Network:</h4>
          <NetworkGraph nodes={result.nodes} edges={result.edges} />
        </div>
      )}

      {/* Executive Summary */}
      {result.executive_summary && (
        <details className="summary-section">
          <summary>📋 Executive Summary</summary>
          <div className="summary-content">
            {result.executive_summary}
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
            <div>📊 Here's the lineage visualization for your query:</div>
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