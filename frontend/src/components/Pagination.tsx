import React from 'react';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  showPageNumbers?: boolean;
  className?: string;
}

/**
 * Pagination Component
 * 
 * Provides navigation controls for paginated content
 */
export const Pagination: React.FC<PaginationProps> = ({
  currentPage,
  totalPages,
  onPageChange,
  showPageNumbers = true,
  className = '',
}) => {
  // Generate page numbers to display
  const getPageNumbers = (): number[] => {
    const pageNumbers: number[] = [];
    
    // Always show first page
    pageNumbers.push(1);
    
    // Calculate range around current page
    const rangeStart = Math.max(2, currentPage - 1);
    const rangeEnd = Math.min(totalPages - 1, currentPage + 1);
    
    // Add ellipsis after first page if needed
    if (rangeStart > 2) {
      pageNumbers.push(-1); // -1 represents ellipsis
    }
    
    // Add pages in range
    for (let i = rangeStart; i <= rangeEnd; i++) {
      pageNumbers.push(i);
    }
    
    // Add ellipsis before last page if needed
    if (rangeEnd < totalPages - 1) {
      pageNumbers.push(-2); // -2 represents ellipsis
    }
    
    // Always show last page if there is more than one page
    if (totalPages > 1) {
      pageNumbers.push(totalPages);
    }
    
    return pageNumbers;
  };
  
  // Handle page change
  const handlePageChange = (page: number) => {
    if (page >= 1 && page <= totalPages && page !== currentPage) {
      onPageChange(page);
    }
  };
  
  // Render page button
  const renderPageButton = (page: number, index: number) => {
    // Ellipsis
    if (page < 0) {
      return (
        <span key={`ellipsis-${index}`} className="px-3 py-1 text-gray-500">
          ...
        </span>
      );
    }
    
    // Page number
    const isActive = page === currentPage;
    return (
      <button
        key={page}
        onClick={() => handlePageChange(page)}
        className={`px-3 py-1 rounded ${
          isActive
            ? 'bg-blue-600 text-white'
            : 'text-gray-700 hover:bg-gray-100'
        }`}
        aria-current={isActive ? 'page' : undefined}
      >
        {page}
      </button>
    );
  };
  
  return (
    <nav className={`flex items-center justify-center space-x-1 ${className}`}>
      {/* Previous button */}
      <button
        onClick={() => handlePageChange(currentPage - 1)}
        disabled={currentPage === 1}
        className={`px-3 py-1 rounded ${
          currentPage === 1
            ? 'text-gray-400 cursor-not-allowed'
            : 'text-gray-700 hover:bg-gray-100'
        }`}
        aria-label="Previous page"
      >
        &laquo;
      </button>
      
      {/* Page numbers */}
      {showPageNumbers && getPageNumbers().map(renderPageButton)}
      
      {/* Next button */}
      <button
        onClick={() => handlePageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
        className={`px-3 py-1 rounded ${
          currentPage === totalPages
            ? 'text-gray-400 cursor-not-allowed'
            : 'text-gray-700 hover:bg-gray-100'
        }`}
        aria-label="Next page"
      >
        &raquo;
      </button>
    </nav>
  );
};

export default Pagination;
