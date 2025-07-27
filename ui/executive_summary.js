import React, { useState } from 'react';

const ExecutiveSummaryFormatter = ({ summary }) => {
  const [expandedSections, setExpandedSections] = useState({});

  const toggleSection = (sectionId) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  // Parse the summary text into structured sections
  const parseStructuredSummary = (text) => {
    if (!text) return [];

    const sections = [];
    const lines = text.split('\n').filter(line => line.trim());
    let currentSection = null;
    let currentSubsection = null;

    lines.forEach(line => {
      const trimmedLine = line.trim();

      // Main section headers (🔹*text**)
      if (trimmedLine.match(/^🔹\*.*\*\*$/)) {
        if (currentSection) {
          sections.push(currentSection);
        }
        currentSection = {
          id: `section-${sections.length}`,
          title: trimmedLine.replace(/^🔹\*/, '').replace(/\*\*$/, '').trim(),
          type: 'main',
          content: [],
          subsections: []
        };
        currentSubsection = null;
      }
      // Subsection headers (🔹**text**)
      else if (trimmedLine.match(/^🔹\*\*.*\*\*$/)) {
        const subsectionTitle = trimmedLine.replace(/^🔹\*\*/, '').replace(/\*\*$/, '').trim();
        currentSubsection = {
          id: `subsection-${currentSection?.subsections?.length || 0}`,
          title: subsectionTitle,
          type: 'subsection',
          content: [],
          items: []
        };
        if (currentSection) {
          currentSection.subsections.push(currentSubsection);
        }
      }
      // Bullet points or recommendations (🔹**text or 🔹*text)
      else if (trimmedLine.match(/^🔹\*\*?[^*]/)) {
        const content = trimmedLine.replace(/^🔹\*\*?/, '').trim();
        const item = {
          type: 'bullet',
          content: content,
          priority: trimmedLine.includes('**') ? 'high' : 'medium'
        };

        if (currentSubsection) {
          currentSubsection.items.push(item);
        } else if (currentSection) {
          currentSection.content.push(item);
        }
      }
      // Regular content
      else if (trimmedLine && !trimmedLine.startsWith('🔹')) {
        const item = {
          type: 'paragraph',
          content: trimmedLine
        };

        if (currentSubsection) {
          currentSubsection.content.push(item);
        } else if (currentSection) {
          currentSection.content.push(item);
        }
      }
    });

    if (currentSection) {
      sections.push(currentSection);
    }

    return sections;
  };

  // Highlight important terms in text
  const highlightTerms = (text) => {
    const patterns = [
      { regex: /\b[A-Z][A-Z_]{2,}[A-Z]?\b/g, className: 'table-name' },
      { regex: /\b[a-zA-Z]+_[a-zA-Z_]+\b/g, className: 'element-code' },
      { regex: /'([^']+)'/g, className: 'data-element', captureGroup: 1 },
      { regex: /"([^"]+)"/g, className: 'data-element', captureGroup: 1 },
      { regex: /\b\d+(\.\d+)?%\b/g, className: 'metric-highlight' },
      { regex: /\b\d+\s*(elements?|relationships?|transformations?|tables?|records?|rows?)\b/gi, className: 'metric-highlight' },
      { regex: /\b(upstream|downstream|bidirectional|depends on|feeds into|derived from|sources from)\b/gi, className: 'relationship-type' },
      { regex: /\b(transformation|aggregation|filtering|joining|calculation|mapping|validation)\b/gi, className: 'transformation' },
      { regex: /\b(High Risk|Medium Risk|Low Risk)\b/gi, className: 'risk-level' }
    ];

    let result = text;
    const allMatches = [];

    patterns.forEach(pattern => {
      let match;
      const regex = new RegExp(pattern.regex.source, pattern.regex.flags);
      while ((match = regex.exec(text)) !== null) {
        allMatches.push({
          start: match.index,
          end: match.index + match[0].length,
          text: pattern.captureGroup ? match[pattern.captureGroup] : match[0],
          className: pattern.className
        });
      }
    });

    // Sort by position and remove overlaps
    allMatches.sort((a, b) => a.start - b.start);
    const cleanMatches = [];
    allMatches.forEach(match => {
      const hasOverlap = cleanMatches.some(existing =>
        (match.start < existing.end && match.end > existing.start)
      );
      if (!hasOverlap) {
        cleanMatches.push(match);
      }
    });

    if (cleanMatches.length === 0) return text;

    const components = [];
    let lastIndex = 0;

    cleanMatches.forEach((match, index) => {
      if (match.start > lastIndex) {
        components.push(text.substring(lastIndex, match.start));
      }
      components.push(
        <span key={`highlight-${index}`} className={match.className}>
          {match.text}
        </span>
      );
      lastIndex = match.end;
    });

    if (lastIndex < text.length) {
      components.push(text.substring(lastIndex));
    }

    return components;
  };

  const getSectionIcon = (title) => {
    const titleLower = title.toLowerCase();
    if (titleLower.includes('key findings') || titleLower.includes('insights')) return '🔍';
    if (titleLower.includes('data flow') || titleLower.includes('patterns')) return '🔄';
    if (titleLower.includes('risk') || titleLower.includes('recommendations')) return '⚠️';
    if (titleLower.includes('next steps') || titleLower.includes('governance')) return '📋';
    return '📊';
  };

  const getRiskLevel = (content) => {
    if (content.toLowerCase().includes('high risk')) return 'high';
    if (content.toLowerCase().includes('medium risk')) return 'medium';
    if (content.toLowerCase().includes('low risk')) return 'low';
    return 'info';
  };

  const sections = parseStructuredSummary(summary);

  return (
    <div className="enhanced-executive-summary">
      <style jsx>{`
        .enhanced-executive-summary {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          line-height: 1.6;
          color: #2c2c2c;
        }

        .summary-header {
          background: linear-gradient(135deg, #d71921 0%, #b51319 100%);
          color: white;
          padding: 1.5rem;
          border-radius: 12px 12px 0 0;
          margin-bottom: 0;
        }

        .summary-header h2 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 700;
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .summary-sections {
          background: white;
          border-radius: 0 0 12px 12px;
          border: 1px solid #e5e5e5;
          border-top: none;
        }

        .summary-section {
          border-bottom: 1px solid #f0f0f0;
          transition: all 0.3s ease;
        }

        .summary-section:last-child {
          border-bottom: none;
        }

        .section-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 1.25rem 1.5rem;
          cursor: pointer;
          background: #fafafa;
          border: none;
          width: 100%;
          text-align: left;
          font-size: 1.1rem;
          font-weight: 600;
          color: #d71921;
          transition: all 0.2s ease;
        }

        .section-header:hover {
          background: #f5f5f5;
          color: #b51319;
        }

        .section-title {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          flex: 1;
        }

        .section-content {
          padding: 0 1.5rem 1.5rem 1.5rem;
          background: white;
        }

        .section-content.collapsed {
          display: none;
        }

        .subsection {
          margin: 1.5rem 0;
          border-left: 4px solid #f5b800;
          padding-left: 1rem;
          background: #fffbf0;
          border-radius: 0 8px 8px 0;
        }

        .subsection-title {
          font-weight: 700;
          color: #d71921;
          margin-bottom: 1rem;
          font-size: 1.05rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .content-item {
          margin: 0.75rem 0;
        }

        .content-item.paragraph {
          text-align: justify;
          margin-bottom: 1rem;
        }

        .bullet-item {
          display: flex;
          align-items: flex-start;
          gap: 0.75rem;
          margin: 0.75rem 0;
          padding: 0.5rem;
          border-radius: 6px;
          transition: background 0.2s ease;
        }

        .bullet-item:hover {
          background: #f8f9fa;
        }

        .bullet-item.high {
          background: #fff5f5;
          border-left: 4px solid #dc3545;
        }

        .bullet-item.medium {
          background: #fff8e1;
          border-left: 4px solid #ff9800;
        }

        .bullet-item.low {
          background: #f3f8f3;
          border-left: 4px solid #28a745;
        }

        .bullet-icon {
          color: #d71921;
          font-weight: bold;
          margin-top: 0.1rem;
          flex-shrink: 0;
        }

        .bullet-content {
          flex: 1;
        }

        .expand-icon {
          transition: transform 0.3s ease;
          color: #666;
        }

        .expand-icon.expanded {
          transform: rotate(180deg);
        }

        .table-name {
          background: linear-gradient(135deg, #003d6b 0%, #0056b3 100%);
          color: white;
          padding: 0.2rem 0.6rem;
          border-radius: 6px;
          font-weight: 700;
          font-family: 'Courier New', monospace;
          font-size: 0.9rem;
          letter-spacing: 0.5px;
          white-space: nowrap;
          display: inline-block;
          margin: 0 0.2rem;
          box-shadow: 0 2px 8px rgba(0, 61, 107, 0.3);
        }

        .element-code {
          background: linear-gradient(135deg, #f5b800 0%, #d4a000 100%);
          color: #2c2c2c;
          padding: 0.2rem 0.6rem;
          border-radius: 6px;
          font-weight: 700;
          font-family: 'Courier New', monospace;
          font-size: 0.9rem;
          letter-spacing: 0.5px;
          white-space: nowrap;
          display: inline-block;
          margin: 0 0.2rem;
          box-shadow: 0 2px 8px rgba(245, 184, 0, 0.3);
        }

        .data-element {
          background: linear-gradient(135deg, #d71921 0%, #b51319 100%);
          color: white;
          padding: 0.2rem 0.6rem;
          border-radius: 6px;
          font-weight: 700;
          font-size: 0.9rem;
          letter-spacing: 0.5px;
          white-space: nowrap;
          display: inline-block;
          margin: 0 0.2rem;
          box-shadow: 0 2px 8px rgba(215, 25, 33, 0.3);
        }

        .relationship-type {
          background: #28a745;
          color: white;
          padding: 0.2rem 0.6rem;
          border-radius: 6px;
          font-weight: 600;
          font-size: 0.85rem;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          white-space: nowrap;
          display: inline-block;
          margin: 0 0.2rem;
        }

        .transformation {
          background: #fd7e14;
          color: white;
          padding: 0.2rem 0.6rem;
          border-radius: 6px;
          font-weight: 600;
          font-size: 0.85rem;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          white-space: nowrap;
          display: inline-block;
          margin: 0 0.2rem;
        }

        .metric-highlight {
          background: linear-gradient(135deg, #2c2c2c 0%, #666666 100%);
          color: white;
          padding: 0.2rem 0.6rem;
          border-radius: 6px;
          font-weight: 700;
          font-size: 0.9rem;
          white-space: nowrap;
          display: inline-block;
          margin: 0 0.2rem;
        }

        .risk-level {
          padding: 0.3rem 0.8rem;
          border-radius: 20px;
          font-weight: 700;
          font-size: 0.8rem;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          display: inline-block;
          margin: 0 0.5rem 0 0;
        }

        .risk-level.high {
          background: #dc3545;
          color: white;
        }

        .risk-level.medium {
          background: #ff9800;
          color: white;
        }

        .risk-level.low {
          background: #28a745;
          color: white;
        }

        .section-summary {
          background: #f8f9fa;
          border: 1px solid #e9ecef;
          border-radius: 8px;
          padding: 1rem;
          margin: 1rem 0;
          font-style: italic;
          color: #495057;
        }
      `}</style>

      <div className="summary-header">
        <h2>
          <span>📋</span>
          Executive Summary
        </h2>
      </div>

      <div className="summary-sections">
        {sections.map((section) => (
          <div key={section.id} className="summary-section">
            <button
              className="section-header"
              onClick={() => toggleSection(section.id)}
            >
              <div className="section-title">
                <span>{getSectionIcon(section.title)}</span>
                <span>{section.title}</span>
              </div>
              <span className={`expand-icon ${expandedSections[section.id] ? 'expanded' : ''}`}>
                ▼
              </span>
            </button>

            <div className={`section-content ${expandedSections[section.id] ? '' : 'collapsed'}`}>
              {/* Main section content */}
              {section.content.map((item, index) => (
                <div key={index} className={`content-item ${item.type}`}>
                  {item.type === 'bullet' ? (
                    <div className={`bullet-item ${getRiskLevel(item.content)}`}>
                      <span className="bullet-icon">
                        {item.priority === 'high' ? '🔴' : '🔸'}
                      </span>
                      <div className="bullet-content">
                        {highlightTerms(item.content)}
                      </div>
                    </div>
                  ) : (
                    <div>
                      {highlightTerms(item.content)}
                    </div>
                  )}
                </div>
              ))}

              {/* Subsections */}
              {section.subsections.map((subsection) => (
                <div key={subsection.id} className="subsection">
                  <div className="subsection-title">
                    <span>📌</span>
                    {subsection.title}
                  </div>

                  {subsection.content.map((item, index) => (
                    <div key={index} className={`content-item ${item.type}`}>
                      {highlightTerms(item.content)}
                    </div>
                  ))}

                  {subsection.items.map((item, index) => (
                    <div key={index} className={`bullet-item ${getRiskLevel(item.content)}`}>
                      <span className="bullet-icon">
                        {item.priority === 'high' ? '🔴' : '🔸'}
                      </span>
                      <div className="bullet-content">
                        {highlightTerms(item.content)}
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Demo component with sample data
const ExecutiveSummaryDemo = () => {
  const sampleSummary = `🔹*Executive Summary: Lineage Analysis of Order Contract (C002)**
🔹*1. Key Findings and Insights:**
This lineage analysis of the order contract (C002) reveals a partially understood data pipeline with significant gaps and potential risks. The analysis identified two core ETL processes: Q002 (loading order facts) and Q003 (aggregating sales data). While the data flow between Q002 and Q003 is clear, a critical upstream query (Q001) remains undefined, representing a major knowledge gap. Q002 exhibits an unusually high dependency (50x) on this missing query, suggesting a potential design flaw or critical data preparation step that is currently undocumented.

🔹*2. Data Flow Patterns:**
The data flows linearly from the orders and products source tables through Q002 to the fact_orders table, and then from fact_orders through Q003 to the agg_sales table. Q003 is dependent on Q002, performing a SUM_AGGREGATION to generate sales summaries. The critical path is the flow of order_amount through aggregation to the final sales_summary.

🔹*3. Risk Assessment and Recommendations:**
🔹**High Risk: Missing Upstream Query (Q001):** The undefined Q001 poses the most significant risk. Its identification and documentation are paramount to understanding the complete data lineage, potential bottlenecks, and data quality issues.
🔹**Medium Risk: Data Quality and Validation:** The lack of documented data validation and cleansing steps within Q001 and Q002 introduces risks of inaccurate data entering the pipeline.
🔹**Low Risk: Redundant Mappings:** The redundant mappings are a data quality issue within the input provided for the lineage analysis, not a problem within the pipeline itself.

🔹*Recommendations:**
🔹**Prioritize Q001 Investigation:** Immediately investigate and document Q001's functionality, inputs, outputs, and data quality checks.
🔹**Implement Data Validation:** Implement robust data validation and cleansing steps in Q001 and Q002 to ensure data accuracy.
🔹**Enhance Error Handling:** Implement comprehensive error logging and handling mechanisms throughout the pipeline.

🔹*4. Next Steps for Data Governance:**
🔹**Complete Lineage Mapping:** Extend the lineage analysis to include the complete data flow, starting from the origin of the data.
🔹**Metadata Enrichment:** Enrich the metadata associated with each transformation step.
🔹**Regular Monitoring:** Establish a regular schedule for lineage analysis to proactively identify issues.`;

  return (
    <div style={{ padding: '2rem', backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      <EnhancedExecutiveSummary summary={sampleSummary} />
    </div>
  );
};

export default ExecutiveSummaryDemo;