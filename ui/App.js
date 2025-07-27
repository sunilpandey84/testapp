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
    processing: { icon: 'loader', color: '#17a2b8', text: 'Processing...' },
    waiting: { icon: 'clock', color: '#6c757d', text: 'Waiting for input...' },
    ready: { icon: 'check-circle', color: '#28a745', text: 'Ready' },
    error: { icon: 'alert-circle', color: '#dc3545', text: 'Error' }
  };

  const config = statusConfig[status] || statusConfig.ready;

  return (
    <div className="status-indicator" role="status" aria-label={config.text}>
      <svg className="status-icon" width="12" height="12" viewBox="0 0 24 24">
        {/* Use proper SVG icons instead of emoji */}
      </svg>
      <span>{config.text}</span>
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


//Enhanced Executive Summary Formatter Component (inline for convenience)
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

// const ExecutiveSummaryFormatter = ({ summary }) => {
//   const [expandedSections, setExpandedSections] = useState({});
//
//   const toggleSection = (sectionId) => {
//     setExpandedSections(prev => ({
//       ...prev,
//       [sectionId]: !prev[sectionId]
//     }));
//   };
//
//   // Parse the summary text into structured sections
//   const parseStructuredSummary = (text) => {
//     if (!text) return [];
//
//     const sections = [];
//     const lines = text.split('\n').filter(line => line.trim());
//     let currentSection = null;
//     let currentSubsection = null;
//
//     lines.forEach(line => {
//       const trimmedLine = line.trim();
//
//       // Main section headers (🔹*text**)
//       if (trimmedLine.match(/^🔹\*.*\*\*$/)) {
//         if (currentSection) {
//           sections.push(currentSection);
//         }
//         currentSection = {
//           id: `section-${sections.length}`,
//           title: trimmedLine.replace(/^🔹\*/, '').replace(/\*\*$/, '').trim(),
//           type: 'main',
//           content: [],
//           subsections: []
//         };
//         currentSubsection = null;
//       }
//       // Subsection headers (🔹**text**)
//       else if (trimmedLine.match(/^🔹\*\*.*\*\*$/)) {
//         const subsectionTitle = trimmedLine.replace(/^🔹\*\*/, '').replace(/\*\*$/, '').trim();
//         currentSubsection = {
//           id: `subsection-${currentSection?.subsections?.length || 0}`,
//           title: subsectionTitle,
//           type: 'subsection',
//           content: [],
//           items: []
//         };
//         if (currentSection) {
//           currentSection.subsections.push(currentSubsection);
//         }
//       }
//       // Bullet points or recommendations (🔹**text or 🔹*text)
//       else if (trimmedLine.match(/^🔹\*\*?[^*]/)) {
//         const content = trimmedLine.replace(/^🔹\*\*?/, '').trim();
//         const item = {
//           type: 'bullet',
//           content: content,
//           priority: trimmedLine.includes('**') ? 'high' : 'medium'
//         };
//
//         if (currentSubsection) {
//           currentSubsection.items.push(item);
//         } else if (currentSection) {
//           currentSection.content.push(item);
//         }
//       }
//       // Regular content
//       else if (trimmedLine && !trimmedLine.startsWith('🔹')) {
//         const item = {
//           type: 'paragraph',
//           content: trimmedLine
//         };
//
//         if (currentSubsection) {
//           currentSubsection.content.push(item);
//         } else if (currentSection) {
//           currentSection.content.push(item);
//         }
//       }
//     });
//
//     if (currentSection) {
//       sections.push(currentSection);
//     }
//
//     return sections;
//   };
//
//   // Highlight important terms in text
//   const highlightTerms = (text) => {
//     const patterns = [
//       { regex: /\b[A-Z][A-Z_]{2,}[A-Z]?\b/g, className: 'table-name' },
//       { regex: /\b[a-zA-Z]+_[a-zA-Z_]+\b/g, className: 'element-code' },
//       { regex: /'([^']+)'/g, className: 'data-element', captureGroup: 1 },
//       { regex: /"([^"]+)"/g, className: 'data-element', captureGroup: 1 },
//       { regex: /\b\d+(\.\d+)?%\b/g, className: 'metric-highlight' },
//       { regex: /\b\d+\s*(elements?|relationships?|transformations?|tables?|records?|rows?)\b/gi, className: 'metric-highlight' },
//       { regex: /\b(upstream|downstream|bidirectional|depends on|feeds into|derived from|sources from)\b/gi, className: 'relationship-type' },
//       { regex: /\b(transformation|aggregation|filtering|joining|calculation|mapping|validation)\b/gi, className: 'transformation' },
//       { regex: /\b(High Risk|Medium Risk|Low Risk)\b/gi, className: 'risk-level' }
//     ];
//
//     let result = text;
//     const allMatches = [];
//
//     patterns.forEach(pattern => {
//       let match;
//       const regex = new RegExp(pattern.regex.source, pattern.regex.flags);
//       while ((match = regex.exec(text)) !== null) {
//         allMatches.push({
//           start: match.index,
//           end: match.index + match[0].length,
//           text: pattern.captureGroup ? match[pattern.captureGroup] : match[0],
//           className: pattern.className
//         });
//       }
//     });
//
//     // Sort by position and remove overlaps
//     allMatches.sort((a, b) => a.start - b.start);
//     const cleanMatches = [];
//     allMatches.forEach(match => {
//       const hasOverlap = cleanMatches.some(existing =>
//         (match.start < existing.end && match.end > existing.start)
//       );
//       if (!hasOverlap) {
//         cleanMatches.push(match);
//       }
//     });
//
//     if (cleanMatches.length === 0) return text;
//
//     const components = [];
//     let lastIndex = 0;
//
//     cleanMatches.forEach((match, index) => {
//       if (match.start > lastIndex) {
//         components.push(text.substring(lastIndex, match.start));
//       }
//       components.push(
//         <span key={`highlight-${index}`} className={match.className}>
//           {match.text}
//         </span>
//       );
//       lastIndex = match.end;
//     });
//
//     if (lastIndex < text.length) {
//       components.push(text.substring(lastIndex));
//     }
//
//     return components;
//   };
//
//   const getSectionIcon = (title) => {
//     const titleLower = title.toLowerCase();
//     if (titleLower.includes('key findings') || titleLower.includes('insights')) return '🔍';
//     if (titleLower.includes('data flow') || titleLower.includes('patterns')) return '🔄';
//     if (titleLower.includes('risk') || titleLower.includes('recommendations')) return '⚠️';
//     if (titleLower.includes('next steps') || titleLower.includes('governance')) return '📋';
//     return '📊';
//   };
//
//   const getRiskLevel = (content) => {
//     if (content.toLowerCase().includes('high risk')) return 'high';
//     if (content.toLowerCase().includes('medium risk')) return 'medium';
//     if (content.toLowerCase().includes('low risk')) return 'low';
//     return 'info';
//   };
//
//   const sections = parseStructuredSummary(summary);
//
//   return (
//     <div className="enhanced-executive-summary">
//       <style jsx>{`
//         .enhanced-executive-summary {
//           font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
//           line-height: 1.6;
//           color: #2c2c2c;
//         }
//
//         .summary-header {
//           background: linear-gradient(135deg, #d71921 0%, #b51319 100%);
//           color: white;
//           padding: 1.5rem;
//           border-radius: 12px 12px 0 0;
//           margin-bottom: 0;
//         }
//
//         .summary-header h2 {
//           margin: 0;
//           font-size: 1.5rem;
//           font-weight: 700;
//           display: flex;
//           align-items: center;
//           gap: 0.75rem;
//         }
//
//         .summary-sections {
//           background: white;
//           border-radius: 0 0 12px 12px;
//           border: 1px solid #e5e5e5;
//           border-top: none;
//         }
//
//         .summary-section {
//           border-bottom: 1px solid #f0f0f0;
//           transition: all 0.3s ease;
//         }
//
//         .summary-section:last-child {
//           border-bottom: none;
//         }
//
//         .section-header {
//           display: flex;
//           align-items: center;
//           justify-content: space-between;
//           padding: 1.25rem 1.5rem;
//           cursor: pointer;
//           background: #fafafa;
//           border: none;
//           width: 100%;
//           text-align: left;
//           font-size: 1.1rem;
//           font-weight: 600;
//           color: #d71921;
//           transition: all 0.2s ease;
//         }
//
//         .section-header:hover {
//           background: #f5f5f5;
//           color: #b51319;
//         }
//
//         .section-title {
//           display: flex;
//           align-items: center;
//           gap: 0.75rem;
//           flex: 1;
//         }
//
//         .section-content {
//           padding: 0 1.5rem 1.5rem 1.5rem;
//           background: white;
//         }
//
//         .section-content.collapsed {
//           display: none;
//         }
//
//         .subsection {
//           margin: 1.5rem 0;
//           border-left: 4px solid #f5b800;
//           padding-left: 1rem;
//           background: #fffbf0;
//           border-radius: 0 8px 8px 0;
//         }
//
//         .subsection-title {
//           font-weight: 700;
//           color: #d71921;
//           margin-bottom: 1rem;
//           font-size: 1.05rem;
//           display: flex;
//           align-items: center;
//           gap: 0.5rem;
//         }
//
//         .content-item {
//           margin: 0.75rem 0;
//         }
//
//         .content-item.paragraph {
//           text-align: justify;
//           margin-bottom: 1rem;
//         }
//
//         .bullet-item {
//           display: flex;
//           align-items: flex-start;
//           gap: 0.75rem;
//           margin: 0.75rem 0;
//           padding: 0.5rem;
//           border-radius: 6px;
//           transition: background 0.2s ease;
//         }
//
//         .bullet-item:hover {
//           background: #f8f9fa;
//         }
//
//         .bullet-item.high {
//           background: #fff5f5;
//           border-left: 4px solid #dc3545;
//         }
//
//         .bullet-item.medium {
//           background: #fff8e1;
//           border-left: 4px solid #ff9800;
//         }
//
//         .bullet-item.low {
//           background: #f3f8f3;
//           border-left: 4px solid #28a745;
//         }
//
//         .bullet-icon {
//           color: #d71921;
//           font-weight: bold;
//           margin-top: 0.1rem;
//           flex-shrink: 0;
//         }
//
//         .bullet-content {
//           flex: 1;
//         }
//
//         .expand-icon {
//           transition: transform 0.3s ease;
//           color: #666;
//         }
//
//         .expand-icon.expanded {
//           transform: rotate(180deg);
//         }
//
//         .table-name {
//           background: linear-gradient(135deg, #003d6b 0%, #0056b3 100%);
//           color: white;
//           padding: 0.2rem 0.6rem;
//           border-radius: 6px;
//           font-weight: 700;
//           font-family: 'Courier New', monospace;
//           font-size: 0.9rem;
//           letter-spacing: 0.5px;
//           white-space: nowrap;
//           display: inline-block;
//           margin: 0 0.2rem;
//           box-shadow: 0 2px 8px rgba(0, 61, 107, 0.3);
//         }
//
//         .element-code {
//           background: linear-gradient(135deg, #f5b800 0%, #d4a000 100%);
//           color: #2c2c2c;
//           padding: 0.2rem 0.6rem;
//           border-radius: 6px;
//           font-weight: 700;
//           font-family: 'Courier New', monospace;
//           font-size: 0.9rem;
//           letter-spacing: 0.5px;
//           white-space: nowrap;
//           display: inline-block;
//           margin: 0 0.2rem;
//           box-shadow: 0 2px 8px rgba(245, 184, 0, 0.3);
//         }
//
//         .data-element {
//           background: linear-gradient(135deg, #d71921 0%, #b51319 100%);
//           color: white;
//           padding: 0.2rem 0.6rem;
//           border-radius: 6px;
//           font-weight: 700;
//           font-size: 0.9rem;
//           letter-spacing: 0.5px;
//           white-space: nowrap;
//           display: inline-block;
//           margin: 0 0.2rem;
//           box-shadow: 0 2px 8px rgba(215, 25, 33, 0.3);
//         }
//
//         .relationship-type {
//           background: #28a745;
//           color: white;
//           padding: 0.2rem 0.6rem;
//           border-radius: 6px;
//           font-weight: 600;
//           font-size: 0.85rem;
//           text-transform: uppercase;
//           letter-spacing: 0.5px;
//           white-space: nowrap;
//           display: inline-block;
//           margin: 0 0.2rem;
//         }
//
//         .transformation {
//           background: #fd7e14;
//           color: white;
//           padding: 0.2rem 0.6rem;
//           border-radius: 6px;
//           font-weight: 600;
//           font-size: 0.85rem;
//           text-transform: uppercase;
//           letter-spacing: 0.5px;
//           white-space: nowrap;
//           display: inline-block;
//           margin: 0 0.2rem;
//         }
//
//         .metric-highlight {
//           background: linear-gradient(135deg, #2c2c2c 0%, #666666 100%);
//           color: white;
//           padding: 0.2rem 0.6rem;
//           border-radius: 6px;
//           font-weight: 700;
//           font-size: 0.9rem;
//           white-space: nowrap;
//           display: inline-block;
//           margin: 0 0.2rem;
//         }
//
//         .risk-level {
//           padding: 0.3rem 0.8rem;
//           border-radius: 20px;
//           font-weight: 700;
//           font-size: 0.8rem;
//           text-transform: uppercase;
//           letter-spacing: 0.5px;
//           display: inline-block;
//           margin: 0 0.5rem 0 0;
//         }
//
//         .risk-level.high {
//           background: #dc3545;
//           color: white;
//         }
//
//         .risk-level.medium {
//           background: #ff9800;
//           color: white;
//         }
//
//         .risk-level.low {
//           background: #28a745;
//           color: white;
//         }
//
//         .section-summary {
//           background: #f8f9fa;
//           border: 1px solid #e9ecef;
//           border-radius: 8px;
//           padding: 1rem;
//           margin: 1rem 0;
//           font-style: italic;
//           color: #495057;
//         }
//       `}</style>
//
//       <div className="summary-header">
//         <h2>
//           <span>📋</span>
//           Executive Summary
//         </h2>
//       </div>
//
//       <div className="summary-sections">
//         {sections.map((section) => (
//           <div key={section.id} className="summary-section">
//             <button
//               className="section-header"
//               onClick={() => toggleSection(section.id)}
//             >
//               <div className="section-title">
//                 <span>{getSectionIcon(section.title)}</span>
//                 <span>{section.title}</span>
//               </div>
//               <span className={`expand-icon ${expandedSections[section.id] ? 'expanded' : ''}`}>
//                 ▼
//               </span>
//             </button>
//
//             <div className={`section-content ${expandedSections[section.id] ? '' : 'collapsed'}`}>
//               {/* Main section content */}
//               {section.content.map((item, index) => (
//                 <div key={index} className={`content-item ${item.type}`}>
//                   {item.type === 'bullet' ? (
//                     <div className={`bullet-item ${getRiskLevel(item.content)}`}>
//                       <span className="bullet-icon">
//                         {item.priority === 'high' ? '🔴' : '🔸'}
//                       </span>
//                       <div className="bullet-content">
//                         {highlightTerms(item.content)}
//                       </div>
//                     </div>
//                   ) : (
//                     <div>
//                       {highlightTerms(item.content)}
//                     </div>
//                   )}
//                 </div>
//               ))}
//
//               {/* Subsections */}
//               {section.subsections.map((subsection) => (
//                 <div key={subsection.id} className="subsection">
//                   <div className="subsection-title">
//                     <span>📌</span>
//                     {subsection.title}
//                   </div>
//
//                   {subsection.content.map((item, index) => (
//                     <div key={index} className={`content-item ${item.type}`}>
//                       {highlightTerms(item.content)}
//                     </div>
//                   ))}
//
//                   {subsection.items.map((item, index) => (
//                     <div key={index} className={`bullet-item ${getRiskLevel(item.content)}`}>
//                       <span className="bullet-icon">
//                         {item.priority === 'high' ? '🔴' : '🔸'}
//                       </span>
//                       <div className="bullet-content">
//                         {highlightTerms(item.content)}
//                       </div>
//                     </div>
//                   ))}
//                 </div>
//               ))}
//             </div>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };
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
      case MESSAGE_TYPES.VISUALIZATION:
        return (
          <>
            <div>Here's the lineage analysis results for your query:</div>
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
      'Hello! I\'m your Data Lineage Assistant. I can help you trace data lineage, analyze contracts, and explore data relationships. What would you like to explore today?',
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
        `Error processing selection: ${error.message}`,
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
        ' Chat cleared! How can I help you with data lineage today?',
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
            <h3>Chat Controls</h3>
            <StatusIndicator status={currentStatus} />
          </div>

          <div className="sidebar-section">
            <h3>Session Stats</h3>
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
              Clear Chat
            </button>
            <button className="sidebar-btn export-btn" onClick={exportChat}>
              Export Chat
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
