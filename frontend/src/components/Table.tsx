import React from 'react';

interface TableProps {
  children: React.ReactNode;
  className?: string;
}

export const Table: React.FC<TableProps> = ({ children, className = '' }) => {
  return (
    <div className="overflow-x-auto">
      <table className={`min-w-full divide-y divide-gray-200 ${className}`}>
        {children}
      </table>
    </div>
  );
};

interface TableHeaderProps {
  children: React.ReactNode;
  className?: string;
}

export const TableHeader: React.FC<TableHeaderProps> = ({ children, className = '' }) => {
  return (
    <thead className={`bg-gray-50 ${className}`}>
      {children}
    </thead>
  );
};

interface TableRowProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

export const TableRow: React.FC<TableRowProps> = ({ children, className = '', onClick }) => {
  return (
    <tr 
      className={`${className} ${onClick ? 'cursor-pointer hover:bg-gray-50' : ''}`}
      onClick={onClick}
    >
      {children}
    </tr>
  );
};

interface TableCellProps {
  children?: React.ReactNode;
  className?: string;
  as?: 'td' | 'th';
  colSpan?: number;
  rowSpan?: number;
}

export const TableCell: React.FC<TableCellProps> = ({ 
  children, 
  className = '', 
  as = 'td',
  colSpan,
  rowSpan
}) => {
  const baseClasses = 'px-6 py-4';
  const cellClasses = as === 'th' 
    ? 'text-left text-xs font-medium text-gray-500 uppercase tracking-wider'
    : 'text-sm text-gray-500';
  
  const props = {
    className: `${baseClasses} ${cellClasses} ${className}`,
    colSpan,
    rowSpan
  };
  
  return as === 'th' 
    ? <th {...props}>{children}</th> 
    : <td {...props}>{children}</td>;
};

export default { Table, TableHeader, TableRow, TableCell };
