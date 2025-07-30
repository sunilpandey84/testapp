import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// Message Types - Enhanced for agentic system
const MESSAGE_TYPES = {
  TEXT: 'text',
  VISUALIZATION: 'visualization',
  FEEDBACK_REQUEST: 'feedback_request',
  CONTRACT_SUMMARY: 'contract_summary',
  ELEMENT_ANALYSIS: 'element_analysis',
  QUALITY_ASSESSMENT: 'quality_assessment',
  ERROR: 'error',
  SYSTEM: 'system',
  AGENT_REASONING: 'agent_reasoning'
};

// Enhanced Chat Message Class
class ChatMessage {
  constructor(role, content, messageType = MESSAGE_TYPES.TEXT, metadata = {}) {
    this.id = Date.now() + Math.random();
    this.role = role;
    this.content = content;
    this.timestamp = new Date();
    this.messageType = messageType;
    this.metadata = metadata;
    this.agentType = metadata.agentType || null; // Track which agent generated this
    this.complexityScore = metadata.complexityScore || null;
    this.recommendations = metadata.recommendations || [];
  }
}

// Agent Status Indicator Component
const AgentStatusIndicator = ({ status, currentAgent }) => {
  const statusConfig = {
    processing: { icon: '⚙️', color: '#17a2b8', text: `${currentAgent} Agent Processing...` },
    waiting: { icon: '⏳', color: '#6c757d', text: 'Waiting for input...' },
    ready: { icon: '✅', color: '#28a745', text: 'Ready' },
    error: { icon: '❌', color: '#dc3545', text: 'Error' },
    analyzing: { icon: '🧠', color: '#ffc107', text: 'AI Agent Analyzing...' },
    summarizing: { icon: '📊', color: '#6f42c1', text: 'Summarization Agent Working...' }
  };

  const config = statusConfig[status] || statusConfig.ready;

  return (
    <div className="agent-status-indicator" role="status" aria-label={config.text}>
      <span className="status-icon">{config.icon}</span>
      <span className="status-text">{config.text}</span>
      {currentAgent && (
        <div className="current-agent">
          <small>Current: {currentAgent.replace('_', ' ').toUpperCase()}</small>
        </div>
      )}
    </div>
  );
};

// Agent Reasoning Component - Show LLM reasoning
const AgentReasoningDisplay = ({ reasoning, agentType }) => {
  if (!reasoning) return null;

  return (
    <details className="agent-reasoning">
      <summary>🧠 {agentType} Agent Reasoning</summary>
      <div className="reasoning-content">
        <pre>{typeof reasoning === 'string' ? reasoning : JSON.stringify(reasoning, null, 2)}</pre>
      </div>
    </details>
  );
};

// Enhanced Contract Summary Component
const ContractSummaryVisualization = ({ result }) => {
  const [activeTab, setActiveTab] = useState('overview');

  // Extract contract summary data
  const contractDetails = result.contract_details || {};
  const qualityAnalysis = result.quality_analysis || {};
  const statistics = result.statistics || {};
  const systemAnalysis = result.system_analysis || {};
  const recommendations = result.recommendations || [];

  return (
    <div className="contract-summary-container">
      {/* Quality Score Header */}
      <div className="quality-header">
        <div className="quality-score-card">
          <div className="score-circle" data-score={qualityAnalysis.quality_score || 0}>
            <span className="score-number">{qualityAnalysis.quality_score || 0}</span>
            <span className="score-label">Quality Score</span>
          </div>
          <div className="quality-level">
            <span className={`level-badge ${(qualityAnalysis.quality_level || '').toLowerCase()}`}>
              {qualityAnalysis.quality_level || 'Unknown'}
            </span>
          </div>
        </div>

        <div className="contract-info">
          <h3>{contractDetails.contract_name || 'Contract Analysis'}</h3>
          <p>{contractDetails.description || 'No description available'}</p>
          <div className="contract-meta">
            <span>📊 {statistics.pipeline_count || 0} Pipelines</span>
            <span>🔄 {statistics.transformation_count || 0} Transformations</span>
            <span>🎯 {systemAnalysis.source_system || 'N/A'} → {systemAnalysis.target_system || 'N/A'}</span>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="summary-tabs">
        <button
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          📋 Overview
        </button>
        <button
          className={`tab-btn ${activeTab === 'quality' ? 'active' : ''}`}
          onClick={() => setActiveTab('quality')}
        >
          🔍 Quality Issues
        </button>
        <button
          className={`tab-btn ${activeTab === 'statistics' ? 'active' : ''}`}
          onClick={() => setActiveTab('statistics')}
        >
          📊 Statistics
        </button>
        <button
          className={`tab-btn ${activeTab === 'recommendations' ? 'active' : ''}`}
          onClick={() => setActiveTab('recommendations')}
        >
          💡 Recommendations
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'overview' && (
          <div className="overview-panel">
            <div className="info-grid">
              <div className="info-card">
                <h4>Contract Details</h4>
                <ul>
                  <li><strong>Code:</strong> {contractDetails.contract_code || 'N/A'}</li>
                  <li><strong>Source Owner:</strong> {contractDetails.source_owner || 'N/A'}</li>
                  <li><strong>Ingestion Owner:</strong> {contractDetails.ingestion_owner || 'N/A'}</li>
                </ul>
              </div>

              <div className="info-card">
                <h4>System Architecture</h4>
                <ul>
                  <li><strong>Source System:</strong> {systemAnalysis.source_system || 'N/A'}</li>
                  <li><strong>Target System:</strong> {systemAnalysis.target_system || 'N/A'}</li>
                  <li><strong>Data Flow:</strong> {systemAnalysis.data_flow_complexity || 'Unknown'}</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'quality' && (
          <div className="quality-panel">
            <div className="quality-issues">
              <h4>Quality Issues Found</h4>
              {qualityAnalysis.quality_issues && qualityAnalysis.quality_issues.length > 0 ? (
                <ul className="issues-list">
                  {qualityAnalysis.quality_issues.map((issue, index) => (
                    <li key={index} className="issue-item">
                      <span className="issue-icon">⚠️</span>
                      {issue}
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="no-issues">
                  <span className="success-icon">✅</span>
                  No quality issues found!
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'statistics' && (
          <div className="statistics-panel">
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-number">{statistics.pipeline_count || 0}</div>
                <div className="stat-label">Total Pipelines</div>
              </div>
              <div className="stat-card">
                <div className="stat-number">{statistics.unique_source_types || 0}</div>
                <div className="stat-label">Source Types</div>
              </div>
              <div className="stat-card">
                <div className="stat-number">{statistics.unique_target_types || 0}</div>
                <div className="stat-label">Target Types</div>
              </div>
              <div className="stat-card">
                <div className="stat-number">{statistics.transformation_count || 0}</div>
                <div className="stat-label">Transformations</div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'recommendations' && (
          <div className="recommendations-panel">
            <h4>AI-Generated Recommendations</h4>
            {recommendations.length > 0 ? (
              <div className="recommendations-list">
                {recommendations.map((rec, index) => (
                  <div key={index} className="recommendation-item">
                    <span className="rec-icon">💡</span>
                    <span className="rec-text">{rec}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-recommendations">
                <span>ℹ️</span>
                No specific recommendations at this time.
              </div>
            )}
          </div>
        )}
      </div>

      {/* LLM Insights */}
      {result.llm_insights && (
        <details className="llm-insights">
          <summary>🤖 AI Analysis Insights</summary>
          <div className="insights-content">
            <div className="formatted-insights">
              {result.llm_insights.split('\n\n').map((paragraph, index) => (
                <p key={index}>{paragraph}</p>
              ))}
            </div>
          </div>
        </details>
      )}
    </div>
  );
};

// Enhanced Feedback Interface with Agent Context
const FeedbackInterface = ({ feedbackData, onSelection }) => {
  const queryResults = feedbackData.query_results || {};
  const agentContext = feedbackData.llm_analysis || {};

  // Show agent reasoning if available
  const renderAgentReasoning = () => {
    if (agentContext.llm_reasoning) {
      return (
        <div className="agent-context">
          <details>
            <summary>🧠 Why am I asking this?</summary>
            <div className="context-explanation">
              {agentContext.llm_reasoning}
            </div>
          </details>
        </div>
      );
    }
    return null;
  };

  if (queryResults.available_elements) {
    return (
      <div className="feedback-interface enhanced">
        {renderAgentReasoning()}
        <p><strong>🎯 Available Data Elements:</strong></p>
        <div className="feedback-grid">
          {queryResults.available_elements.map((element, index) => (
            <button
              key={element.name}
              className="feedback-btn element-btn enhanced"
              onClick={() => onSelection({ selected_index: index })}
              title={`Select ${element.name} for lineage analysis`}
            >
              <span className="btn-icon">📊</span>
              <span className="btn-text">{element.name}</span>
              <small className="btn-detail">Data Element</small>
            </button>
          ))}
        </div>
        <div className="selection-helper">
          <small>💡 Tip: Choose the data element you want to trace through the system</small>
        </div>
      </div>
    );
  }

  if (queryResults.available_contracts) {
    return (
      <div className="feedback-interface enhanced">
        {renderAgentReasoning()}
        <p><strong>📋 Available Data Contracts:</strong></p>
        <div className="feedback-grid">
          {queryResults.available_contracts.map((contract, index) => (
            <button
              key={contract.name}
              className="feedback-btn contract-btn enhanced"
              onClick={() => onSelection({ selected_index: index })}
              title={`Analyze ${contract.name} contract`}
            >
              <span className="btn-icon">📄</span>
              <span className="btn-text">{contract.name}</span>
              <small className="btn-detail">Data Contract</small>
            </button>
          ))}
        </div>

        {/* Special option for overall analysis */}
        <div className="special-actions">
          <button
            className="feedback-btn overall-btn"
            onClick={() => onSelection({ selected_name: 'all contracts' })}
            title="Analyze all contracts for ecosystem overview"
          >
            <span className="btn-icon">🌐</span>
            <span className="btn-text">All Contracts</span>
            <small className="btn-detail">Ecosystem Analysis</small>
          </button>
        </div>

        <div className="selection-helper">
          <small>💡 Tip: Select a specific contract or choose "All Contracts" for ecosystem analysis</small>
        </div>
      </div>
    );
  }

  if (queryResults.ambiguous_elements) {
    return (
      <div className="feedback-interface enhanced">
        {renderAgentReasoning()}
        <p><strong>🔍 Multiple matches found. Please choose the specific element:</strong></p>
        <div className="feedback-buttons vertical">
          {queryResults.ambiguous_elements.map((element, index) => (
            <button
              key={element.element_code}
              className="feedback-btn ambiguous-btn enhanced"
              onClick={() => onSelection({ selected_index: index })}
              title={`Code: ${element.element_code} | Table: ${element.table_name}`}
            >
              <div className="ambiguous-item">
                <span className="element-name">🎯 {element.element_name}</span>
                <span className="element-details">
                  <small>Code: {element.element_code}</small>
                  <small>Table: {element.table_name}</small>
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return null;
};

// Enhanced Message Component with Agent Context
const MessageComponent = ({ message, onFeedbackSelection }) => {
  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const renderMessageContent = () => {
    switch (message.messageType) {
      case MESSAGE_TYPES.CONTRACT_SUMMARY:
        return (
          <>
            <div className="message-header">
              <span className="agent-badge">📊 Contract Analysis Agent</span>
              <span>Contract analysis completed successfully!</span>
            </div>
            <ContractSummaryVisualization result={message.metadata.result} />
          </>
        );

      case MESSAGE_TYPES.VISUALIZATION:
        const analysisType = message.metadata.result?.analysis_type || message.metadata.result?.lineage_type;
        const agentName = analysisType === 'contract_summarization' ?
          'Contract Summarization Agent' :
          analysisType === 'element_based' ? 'Element Analysis Agent' : 'Contract Analysis Agent';

        return (
          <>
            <div className="message-header">
              <span className="agent-badge">🤖 {agentName}</span>
              <span>Analysis completed! Here are your results:</span>
            </div>
            {analysisType === 'contract_summarization' ? (
              <ContractSummaryVisualization result={message.metadata.result} />
            ) : (
              <LineageVisualization result={message.metadata.result} />
            )}
          </>
        );

      case MESSAGE_TYPES.FEEDBACK_REQUEST:
        return (
          <>
            <div className="message-header">
              <span className="agent-badge">🧠 Coordinator Agent</span>
              <span>{message.content}</span>
            </div>
            <FeedbackInterface
              feedbackData={message.metadata}
              onSelection={onFeedbackSelection}
            />
          </>
        );

      case MESSAGE_TYPES.AGENT_REASONING:
        return (
          <>
            <div>{message.content}</div>
            <AgentReasoningDisplay
              reasoning={message.metadata.reasoning}
              agentType={message.metadata.agentType}
            />
          </>
        );

      default:
        return (
          <div>
            {message.agentType && (
              <div className="message-header">
                <span className="agent-badge">🤖 {message.agentType.replace('_', ' ').toUpperCase()}</span>
              </div>
            )}
            <div>{message.content}</div>
          </div>
        );
    }
  };

  return (
    <div className={`message ${message.role}-message enhanced`}>
      <div className="message-content">
        {renderMessageContent()}
      </div>
      <div className="message-meta">
        <div className="message-timestamp">
          {formatTime(message.timestamp)}
        </div>
        {message.complexityScore && (
          <div className="complexity-indicator">
            <span>Complexity: {message.complexityScore}/10</span>
          </div>
        )}
      </div>
    </div>
  );
};

// Enhanced Quick Actions with Agent-Specific Examples
const QuickActions = ({ onActionClick, disabled }) => {
  const getQuickActions = () => [
    {
      text: "trace customer_id lineage",
      icon: "🔍",
      description: "Element-based analysis",
      category: "element"
    },
    {
      text: "summarize Customer Data Pipeline",
      icon: "📊",
      description: "Contract quality analysis",
      category: "summary"
    },
    {
      text: "analyze all contracts quality",
      icon: "🌐",
      description: "Ecosystem overview",
      category: "ecosystem"
    },
    {
      text: "upstream dependencies for order_total",
      icon: "⬆️",
      description: "Dependency mapping",
      category: "dependency"
    }
  ];

  return (
    <div className="quick-actions enhanced">
      <p><strong>💡 Try these agent-powered queries:</strong></p>
      <div className="quick-buttons-grid">
        {getQuickActions().map((action, index) => (
          <button
            key={index}
            className={`quick-action-btn ${action.category}`}
            onClick={() => onActionClick(action.text)}
            disabled={disabled}
            title={action.description}
          >
            <span className="action-icon">{action.icon}</span>
            <span className="action-text">{action.text}</span>
            <small className="action-desc">{action.description}</small>
          </button>
        ))}
      </div>
    </div>
  );
};

// Keep existing LineageVisualization, DataElementsTable, etc. components as they are...
// (Including all the table components and LineageVisualization from your original code)

// For brevity, I'm including just the essential parts. The table components remain the same:
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

const LineageVisualization = ({ result }) => {
  const [activeTab, setActiveTab] = useState('elements');

  const elements = result.nodes || result.data_elements || result.elements || [];
  const relationships = result.edges || result.relationships || [];
  const transformations = result.transformations || [];

  return (
    <div className="visualization-container">
      <div className="metrics-row">
        <div className="metric-card">
          <h3>{result.lineage_type?.replace('_', ' ').toUpperCase() || 'ANALYSIS'}</h3>
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
          <h4>{result.complexity_score || 'N/A'}</h4>
          <small>Complexity Score</small>
        </div>
      </div>

      {result.executive_summary && (
        <details className="summary-section" open>
          <summary>📋 Executive Summary</summary>
          <div className="enhanced-summary-content">
            {result.executive_summary.split('\n\n').map((paragraph, index) => (
              <p key={index}>{paragraph}</p>
            ))}
          </div>
        </details>
      )}

      <div className="tabbed-content">
        <div className="tab-navigation">
          <button
            className={`tab-button ${activeTab === 'elements' ? 'active' : ''}`}
            onClick={() => setActiveTab('elements')}
          >
            📊 Data Elements ({elements.length})
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'elements' && (
            <div className="tab-panel">
              <h4>📊 Data Elements</h4>
              <DataElementsTable elements={elements} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Main App Component with Enhanced Agent Integration
const App = () => {
  const [messages, setMessages] = useState([
    new ChatMessage(
      'system',
      '🤖 Hello! I\'m your AI-powered Data Lineage Assistant with specialized agents for different analysis types:\n\n' +
      '📊 **Contract Summarization Agent** - Analyzes data quality and provides recommendations\n' +
      '🔍 **Element Analysis Agent** - Traces data lineage across systems\n' +
      '📋 **Contract Analysis Agent** - Reviews pipeline configurations\n' +
      '🧠 **Coordinator Agent** - Routes your queries to the right specialist\n\n' +
      'What would you like to explore today?',
      MESSAGE_TYPES.SYSTEM
    )
  ]);

  const [inputValue, setInputValue] = useState('');
  const [processing, setProcessing] = useState(false);
  const [currentAgent, setCurrentAgent] = useState(null);
  const [awaitingFeedback, setAwaitingFeedback] = useState(false);
  const [sessionId] = useState(() => Date.now().toString());
  const [workflowState, setWorkflowState] = useState(null);
  const messagesEndRef = useRef(null);

  const API_BASE_URL = 'http://localhost:8000/api';

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const processQuery = async (query) => {
    try {
      setCurrentAgent('coordinator');
      const response = await axios.post(`${API_BASE_URL}/lineage/query`, {
        query: query,
        session_id: sessionId
      });
      return response.data;
    } catch (error) {
      console.error('API Error:', error);
      setCurrentAgent(null);
      throw new Error(error.response?.data?.detail || 'Failed to process query. Please check your connection.');
    }
  };

  const resumeWithFeedback = async (feedback) => {
    try {
      setCurrentAgent('human_approval');
      const response = await axios.post(`${API_BASE_URL}/lineage/feedback`, {
        feedback: feedback,
        session_id: sessionId,
        thread_id: workflowState?.thread_id
      });
      return response.data;
    } catch (error) {
      console.error('API Error:', error);
      setCurrentAgent(null);
      throw new Error(error.response?.data?.detail || 'Failed to process feedback. Please try again.');
    }
  };

  const addMessage = (message) => {
    setMessages(prev => [...prev, message]);
  };

  const detectAnalysisType = (result) => {
    if (result.analysis_type === 'contract_summarization' || result.analysis_type === 'overall_contract_ecosystem') {
      return MESSAGE_TYPES.CONTRACT_SUMMARY;
    }
    return MESSAGE_TYPES.VISUALIZATION;
  };

  const getAgentFromResult = (result) => {
    const analysisType = result.analysis_type || result.lineage_type;
    switch (analysisType) {
      case 'contract_summarization':
      case 'overall_contract_ecosystem':
        return 'contract_summarization';
      case 'element_based':
        return 'element_analysis';
      case 'contract_based':
        return 'contract_analysis';
      default:
        return 'coordinator';
    }
  };

  const handleUserMessage = async (userInput) => {
    if (!userInput.trim() || processing) return;

    setProcessing(true);
    setCurrentAgent('coordinator');

    const userMessage = new ChatMessage('user', userInput);
    addMessage(userMessage);

    try {
      const result = await processQuery(userInput);

      if (result.human_input_required) {
        setWorkflowState({ thread_id: result.thread_id });
        setCurrentAgent('human_approval');

        const feedbackMessage = new ChatMessage(
          'assistant',
          result.message || 'I need more information to help you...',
          MESSAGE_TYPES.FEEDBACK_REQUEST,
          { ...result, agentType: 'coordinator' }
        );
        addMessage(feedbackMessage);
        setAwaitingFeedback(true);
      } else if (result.error) {
        setCurrentAgent(null);
        const errorMessage = new ChatMessage(
          'assistant',
          `❌ I encountered an issue: ${result.error}\n\nWould you like me to help you rephrase your query?`,
          MESSAGE_TYPES.ERROR
        );
        addMessage(errorMessage);
      } else {
        const agentType = getAgentFromResult(result);
        setCurrentAgent(agentType);

        const messageType = detectAnalysisType(result);
        const successMessage = new ChatMessage(
          'assistant',
          '✅ Analysis completed successfully! Here\'s what I discovered:',
          MESSAGE_TYPES.TEXT,
          { agentType }
        );
        addMessage(successMessage);

        const vizMessage = new ChatMessage(
          'assistant',
          '',
          messageType,
          {
            result,
            agentType,
            complexityScore: result.complexity_score,
            recommendations: result.recommendations
          }
        );
        addMessage(vizMessage);

        setCurrentAgent(null);
      }
    } catch (error) {
      setCurrentAgent(null);
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
    setProcessing(true);

    const selectionMessage = new ChatMessage(
      'user',
      `✅ Selected option ${(feedback.selected_index !== undefined ? feedback.selected_index + 1 : feedback.selected_name) || 'custom'}`
    );
    addMessage(selectionMessage);

    try {
      const result = await resumeWithFeedback(feedback);

      if (result.human_input_required) {
        setCurrentAgent('human_approval');
        const feedbackMessage = new ChatMessage(
          'assistant',
          result.message || 'I need more information...',
          MESSAGE_TYPES.FEEDBACK_REQUEST,
          { ...result, agentType: 'coordinator' }
        );
        addMessage(feedbackMessage);
        setAwaitingFeedback(true);
      } else if (result.error) {
        setCurrentAgent(null);
        const errorMessage = new ChatMessage(
          'assistant',
          `❌ ${result.error}`,
          MESSAGE_TYPES.ERROR
        );
        addMessage(errorMessage);
      } else {
        const agentType = getAgentFromResult(result);
        setCurrentAgent(agentType);

        const messageType = detectAnalysisType(result);
        const vizMessage = new ChatMessage(
          'assistant',
          '',
          messageType,
          {
            result,
            agentType,
            complexityScore: result.complexity_score,
            recommendations: result.recommendations
          }
        );
        addMessage(vizMessage);

        setCurrentAgent(null);
      }
    } catch (error) {
      setCurrentAgent(null);
      const errorMessage = new ChatMessage(
        'assistant',
        `Error processing selection: ${error.message}`,
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

  const clearChat = () => {
    setMessages([
      new ChatMessage(
        'system',
        '🤖 Chat cleared! The AI agents are ready to assist you with data lineage analysis.',
        MESSAGE_TYPES.SYSTEM
      )
    ]);
    setAwaitingFeedback(false);
    setCurrentAgent(null);
    setWorkflowState(null);
  };

  const exportChat = () => {
    const chatExport = messages.map(msg => ({
      timestamp: msg.timestamp.toISOString(),
      role: msg.role,
      content: msg.content,
      type: msg.messageType,
      agentType: msg.agentType,
      complexityScore: msg.complexityScore,
      recommendations: msg.recommendations
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

  const getCurrentStatus = () => {
    if (processing && currentAgent) {
      switch (currentAgent) {
        case 'contract_summarization':
          return 'summarizing';
        case 'element_analysis':
        case 'contract_analysis':
          return 'analyzing';
        default:
          return 'processing';
      }
    }
    return processing ? 'processing' : awaitingFeedback ? 'waiting' : 'ready';
  };

  return (
    <div className="app enhanced">
      {/* Enhanced Header */}
      <header className="app-header enhanced">
        <div className="header-content">
          <h1>🤖 AI-Powered Data Lineage Assistant</h1>
          <div className="header-subtitle">
            <span>Multi-Agent System for Intelligent Data Analysis</span>
          </div>
        </div>
        <AgentStatusIndicator
          status={getCurrentStatus()}
          currentAgent={currentAgent}
        />
      </header>

      <div className="app-body enhanced">
        {/* Enhanced Sidebar */}
        <aside className="sidebar enhanced">
          <div className="sidebar-section">
            <h3>🤖 Active Agents</h3>
            <div className="agents-list">
              <div className={`agent-item ${currentAgent === 'coordinator' ? 'active' : ''}`}>
                <span className="agent-icon">🧠</span>
                <span className="agent-name">Coordinator</span>
                <span className="agent-status">{currentAgent === 'coordinator' ? 'Active' : 'Ready'}</span>
              </div>
              <div className={`agent-item ${currentAgent === 'contract_summarization' ? 'active' : ''}`}>
                <span className="agent-icon">📊</span>
                <span className="agent-name">Contract Analysis</span>
                <span className="agent-status">{currentAgent === 'contract_summarization' ? 'Working' : 'Standby'}</span>
              </div>
              <div className={`agent-item ${currentAgent === 'element_analysis' ? 'active' : ''}`}>
                <span className="agent-icon">🔍</span>
                <span className="agent-name">Element Tracing</span>
                <span className="agent-status">{currentAgent === 'element_analysis' ? 'Tracing' : 'Standby'}</span>
              </div>
              <div className={`agent-item ${currentAgent === 'contract_analysis' ? 'active' : ''}`}>
                <span className="agent-icon">📋</span>
                <span className="agent-name">Pipeline Analysis</span>
                <span className="agent-status">{currentAgent === 'contract_analysis' ? 'Analyzing' : 'Standby'}</span>
              </div>
            </div>
          </div>

          <div className="sidebar-section">
            <h3>📊 Session Analytics</h3>
            <div className="analytics-grid">
              <div className="stat">
                <span className="stat-label">Messages:</span>
                <span className="stat-value">{messages.length}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Queries:</span>
                <span className="stat-value">{messages.filter(m => m.role === 'user').length}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Agent Activations:</span>
                <span className="stat-value">{messages.filter(m => m.agentType).length}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Session ID:</span>
                <span className="stat-value">{sessionId.slice(0, 8)}...</span>
              </div>
            </div>
          </div>

          <div className="sidebar-section">
            <h3>⚡ Agent Capabilities</h3>
            <div className="capabilities-list">
              <div className="capability-item">
                <span className="cap-icon">📊</span>
                <div className="cap-content">
                  <strong>Contract Quality Analysis</strong>
                  <p>Assess metadata quality, identify issues, and provide improvement recommendations</p>
                </div>
              </div>
              <div className="capability-item">
                <span className="cap-icon">🔍</span>
                <div className="cap-content">
                  <strong>Element Lineage Tracing</strong>
                  <p>Trace data elements across systems with upstream/downstream analysis</p>
                </div>
              </div>
              <div className="capability-item">
                <span className="cap-icon">📋</span>
                <div className="cap-content">
                  <strong>Pipeline Architecture</strong>
                  <p>Analyze ETL pipelines, dependencies, and transformation logic</p>
                </div>
              </div>
            </div>
          </div>

          <div className="sidebar-section">
            <h3>🛠️ Actions</h3>
            <button className="sidebar-btn clear-btn" onClick={clearChat}>
              🗑️ Clear Chat
            </button>
            <button className="sidebar-btn export-btn" onClick={exportChat}>
              📥 Export Session
            </button>
          </div>

          <details className="sidebar-section help-section">
            <summary>❓ Agent Usage Guide</summary>
            <div className="help-content">
              <h4>🧠 Coordinator Agent</h4>
              <p>Automatically routes your queries to the right specialist agent based on natural language understanding.</p>

              <h4>📊 Contract Analysis Agent</h4>
              <p><strong>Trigger words:</strong> "summarize", "quality", "analyze contract"</p>
              <p><strong>Examples:</strong></p>
              <ul>
                <li>"Summarize Customer Data Pipeline"</li>
                <li>"Analyze contract quality issues"</li>
                <li>"Show all contracts statistics"</li>
              </ul>

              <h4>🔍 Element Tracing Agent</h4>
              <p><strong>Trigger words:</strong> "trace", element names, "lineage"</p>
              <p><strong>Examples:</strong></p>
              <ul>
                <li>"Trace customer_id lineage"</li>
                <li>"Upstream dependencies for order_total"</li>
                <li>"Bidirectional lineage for email"</li>
              </ul>

              <h4>📋 Pipeline Analysis Agent</h4>
              <p><strong>Trigger words:</strong> "contract", "pipeline", contract names</p>
              <p><strong>Examples:</strong></p>
              <ul>
                <li>"Show Customer Data Pipeline"</li>
                <li>"Analyze payment processing contract"</li>
              </ul>
            </div>
          </details>
        </aside>

        {/* Enhanced Main Chat Area */}
        <main className="chat-main enhanced">
          <div className="chat-container">
            <div className="messages-container">
              {messages.map(message => (
                <MessageComponent
                  key={message.id}
                  message={message}
                  onFeedbackSelection={handleFeedbackSelection}
                />
              ))}

              {processing && (
                <div className="typing-indicator enhanced">
                  <div className="agent-working">
                    <span className="working-icon">🤖</span>
                    <span className="working-text">
                      {currentAgent ?
                        `${currentAgent.replace('_', ' ').toUpperCase()} Agent is working...` :
                        'AI Agent is processing your request...'
                      }
                    </span>
                  </div>
                  <div className="typing-dots">
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Enhanced Input Area */}
          <div className="input-container enhanced">
            {!awaitingFeedback && (
              <QuickActions
                onActionClick={handleUserMessage}
                disabled={processing}
              />
            )}

            <div className="input-row enhanced">
              <div className="input-wrapper">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={
                    awaitingFeedback
                      ? "Please use the buttons above to make your selection..."
                      : "Ask me about data lineage, contracts, or element tracing... (The AI agents will understand!)"
                  }
                  disabled={processing || awaitingFeedback}
                  className="chat-input"
                />
                <div className="input-helpers">
                  <div className="character-count">
                    <small>{inputValue.length}/500</small>
                  </div>
                  {currentAgent && (
                    <div className="active-agent-indicator">
                      <small>🤖 {currentAgent.replace('_', ' ').toUpperCase()}</small>
                    </div>
                  )}
                </div>
              </div>

              <button
                onClick={handleSend}
                disabled={processing || awaitingFeedback || !inputValue.trim()}
                className="send-button enhanced"
                title={
                  processing ? "Agent is working..." :
                  awaitingFeedback ? "Please make a selection above" :
                  !inputValue.trim() ? "Enter a message" :
                  "Send to AI Agent"
                }
              >
                {processing ? (
                  <span className="loading-spinner">⚙️</span>
                ) : (
                  <span>🚀 Send</span>
                )}
              </button>
            </div>

            {/* Agent Routing Preview */}
            {inputValue.trim() && !processing && !awaitingFeedback && (
              <div className="routing-preview">
                <small>
                  🧠 Coordinator will route to: {
                    inputValue.toLowerCase().includes('summariz') || inputValue.toLowerCase().includes('quality') ? '📊 Contract Analysis' :
                    inputValue.toLowerCase().includes('trace') || inputValue.toLowerCase().includes('lineage') ? '🔍 Element Tracing' :
                    inputValue.toLowerCase().includes('contract') || inputValue.toLowerCase().includes('pipeline') ? '📋 Pipeline Analysis' :
                    '🤖 Auto-detect'
                  } Agent
                </small>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Enhanced Status Bar */}
      <footer className="status-bar">
        <div className="status-left">
          <span className="status-item">
            🤖 {messages.filter(m => m.agentType).length} Agent Activations
          </span>
          <span className="status-item">
            💬 {messages.filter(m => m.role === 'user').length} User Queries
          </span>
          {workflowState?.thread_id && (
            <span className="status-item">
              🔗 Thread: {workflowState.thread_id.slice(0, 8)}...
            </span>
          )}
        </div>
        <div className="status-right">
          <span className="status-item">
            {processing ? '⚙️ Processing' : awaitingFeedback ? '⏳ Awaiting Input' : '✅ Ready'}
          </span>
        </div>
      </footer>
    </div>
  );
};

export default App;