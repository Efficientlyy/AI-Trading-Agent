import React, { useEffect, useState } from 'react';
// @ts-ignore
import { Card, Alert, Badge, Collapse, Table, Typography, Space, Button, Tooltip } from 'antd';
// @ts-ignore
import { CheckCircleOutlined, CloseCircleOutlined, WarningOutlined, InfoCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import { format } from 'date-fns';
import { usePaperTrading } from '../../../context/PaperTradingContext';
import { WebSocketTopic } from '../../../services/WebSocketService';
import { ErrorSeverity, ErrorCategory, PaperTradingError } from '../../../types/paperTrading';

const { Title, Text } = Typography;
const { Panel } = Collapse;

/**
 * Component for monitoring and displaying errors in the paper trading system
 */
const PaperTradingErrorMonitor: React.FC = () => {
  const [errors, setErrors] = useState<PaperTradingError[]>([]);
  const [circuitStatus, setCircuitStatus] = useState<Record<string, string>>({
    data_provider: 'CLOSED',
    strategy: 'CLOSED',
    execution: 'CLOSED'
  });
  const { webSocketService } = usePaperTrading();

  useEffect(() => {
    // Subscribe to error events
    if (webSocketService) {
      const errorHandler = (data: any) => {
        const error = data as PaperTradingError;
        setErrors(prev => {
          // Add new error to the beginning of the array
          const newErrors = [error, ...prev];
          // Keep only the last 50 errors
          return newErrors.slice(0, 50);
        });

        // Update circuit status if included in the error details
        if (error.details && typeof error.details === 'object' && 'circuit_state' in error.details) {
          const circuitState = error.details.circuit_state;
          if (typeof circuitState === 'string') {
            setCircuitStatus(prev => ({
              ...prev,
              [error.error_category]: circuitState
            }));
          }
        }
      };

      // Use WebSocketTopic.STATUS for errors since there's no specific errors topic
      webSocketService.on(WebSocketTopic.STATUS, errorHandler);

      return () => {
        webSocketService.off(WebSocketTopic.STATUS, errorHandler);
      };
    }
  }, [webSocketService]);

  // Clear all errors
  const clearErrors = () => {
    setErrors([]);
  };

  // Get severity icon and color
  const getSeverityIcon = (severity: ErrorSeverity) => {
    switch (severity) {
      case ErrorSeverity.INFO:
        return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
      case ErrorSeverity.WARNING:
        return <WarningOutlined style={{ color: '#faad14' }} />;
      case ErrorSeverity.ERROR:
        return <CloseCircleOutlined style={{ color: '#f5222d' }} />;
      case ErrorSeverity.CRITICAL:
        return <CloseCircleOutlined style={{ color: '#a8071a' }} />;
      default:
        return <InfoCircleOutlined style={{ color: '#1890ff' }} />;
    }
  };

  // Get circuit status badge
  const getCircuitStatusBadge = (status: string) => {
    switch (status) {
      case 'CLOSED':
        return <Badge status="success" text="Healthy" />;
      case 'HALF_OPEN':
        return <Badge status="warning" text="Recovering" />;
      case 'OPEN':
        return <Badge status="error" text="Circuit Open" />;
      default:
        return <Badge status="default" text="Unknown" />;
    }
  };

  // Get category display name
  const getCategoryName = (category: ErrorCategory) => {
    switch (category) {
      case ErrorCategory.DATA_PROVIDER:
        return 'Data Provider';
      case ErrorCategory.STRATEGY:
        return 'Strategy';
      case ErrorCategory.EXECUTION:
        return 'Execution';
      case ErrorCategory.PORTFOLIO:
        return 'Portfolio';
      case ErrorCategory.CONFIGURATION:
        return 'Configuration';
      case ErrorCategory.AUTHENTICATION:
        return 'Authentication';
      case ErrorCategory.NETWORK:
        return 'Network';
      case ErrorCategory.DATABASE:
        return 'Database';
      case ErrorCategory.SYSTEM:
        return 'System';
      default:
        return 'Unknown';
    }
  };

  // Format timestamp
  const formatTimestamp = (timestamp: number) => {
    try {
      return format(new Date(timestamp * 1000), 'yyyy-MM-dd HH:mm:ss');
    } catch (e) {
      return 'Invalid Date';
    }
  };

  const columns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: number) => formatTimestamp(timestamp),
      width: 180
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: ErrorSeverity) => (
        <Space>
          {getSeverityIcon(severity)}
          <Text>{severity.toUpperCase()}</Text>
        </Space>
      ),
      width: 120
    },
    {
      title: 'Category',
      dataIndex: 'error_category',
      key: 'error_category',
      render: (category: ErrorCategory) => getCategoryName(category),
      width: 150
    },
    {
      title: 'Message',
      dataIndex: 'message',
      key: 'message',
      render: (message: string, record: PaperTradingError) => (
        <Collapse ghost>
          <Panel header={message} key="1">
            {record.troubleshooting && record.troubleshooting.length > 0 && (
              <div>
                <Text strong>Troubleshooting Steps:</Text>
                <ul>
                  {record.troubleshooting.map((step: string, index: number) => (
                    <li key={index}>{step}</li>
                  ))}
                </ul>
              </div>
            )}
            {record.details && Object.keys(record.details).length > 0 && (
              <div>
                <Text strong>Details:</Text>
                <pre>{JSON.stringify(record.details, null, 2)}</pre>
              </div>
            )}
          </Panel>
        </Collapse>
      )
    }
  ];

  return (
    <Card 
      title={<Title level={4}>System Health Monitor</Title>}
      extra={
        <Button 
          icon={<ReloadOutlined />} 
          onClick={clearErrors}
          type="text"
        >
          Clear
        </Button>
      }
      className="paper-trading-error-monitor"
    >
      <div className="circuit-status-container">
        <Title level={5}>Circuit Breaker Status</Title>
        <div className="circuit-status-grid">
          <div className="circuit-status-item">
            <Text strong>Data Provider:</Text>
            {getCircuitStatusBadge(circuitStatus.data_provider)}
          </div>
          <div className="circuit-status-item">
            <Text strong>Strategy Engine:</Text>
            {getCircuitStatusBadge(circuitStatus.strategy)}
          </div>
          <div className="circuit-status-item">
            <Text strong>Execution Handler:</Text>
            {getCircuitStatusBadge(circuitStatus.execution)}
          </div>
        </div>
      </div>

      <Title level={5} style={{ marginTop: 20 }}>Error Log</Title>
      {errors.length === 0 ? (
        <Alert message="No errors to display" type="info" showIcon />
      ) : (
        <Table 
          dataSource={errors} 
          columns={columns} 
          rowKey={(record: PaperTradingError) => `${record.timestamp}-${record.error_code.code}`}
          pagination={{ pageSize: 10 }}
          size="small"
        />
      )}
    </Card>
  );
};

export default PaperTradingErrorMonitor;
