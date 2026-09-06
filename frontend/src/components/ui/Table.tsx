import { HTMLAttributes } from 'react';
import { clsx } from 'clsx';
import { SkeletonTable } from './Skeleton';

interface Column<T> {
  key: string;
  header: string;
  render?: (row: T, index: number, cellValue: unknown) => React.ReactNode;
  className?: string;
  headerClassName?: string;
}

interface TableProps<T> extends HTMLAttributes<HTMLTableElement> {
  columns: Column<T>[];
  data: T[];
  keyExtractor: (row: T, index: number) => string;
  striped?: boolean;
  hoverable?: boolean;
  emptyMessage?: string;
  loading?: boolean;
  loadingRows?: number;
}

export function Table<T>({
  className,
  columns,
  data,
  keyExtractor,
  striped = true,
  hoverable = true,
  emptyMessage = 'No data available',
  loading,
  loadingRows = 5,
  ...props
}: TableProps<T>) {
  return (
    <div className={clsx('table-container', className)} {...props}>
      {loading ? (
        <div className="p-8">
          <SkeletonTable rows={loadingRows} columns={columns.length} />
        </div>
      ) : data.length === 0 ? (
        <div className="p-8 text-center">
          <p className="text-medical-neutral-500 dark:text-medical-neutral-400">{emptyMessage}</p>
        </div>
      ) : (
        <table className="table-base" role="table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th
                  key={column.key}
                  scope="col"
                  className={clsx(column.headerClassName)}
                >
                  {column.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, rowIndex) => (
              <tr
                key={keyExtractor(row, rowIndex)}
                className={clsx(
                  hoverable && 'hover:bg-medical-neutral-50 dark:hover:bg-medical-neutral-800/50',
                  striped && rowIndex % 2 === 1 && 'bg-medical-neutral-50/50 dark:bg-medical-neutral-800/30'
                )}
              >
                {columns.map((column) => (
                  <td key={column.key} className={clsx(column.className)}>
                    {column.render
                      ? column.render(row, rowIndex, (row as Record<string, unknown>)[column.key])
                      : String((row as Record<string, unknown>)[column.key] ?? '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

interface SortableColumn<T> extends Column<T> {
  sortable?: boolean;
  sortDirection?: 'asc' | 'desc' | null;
  onSort?: (key: string, direction: 'asc' | 'desc') => void;
}

interface SortableTableProps<T> extends Omit<TableProps<T>, 'columns'> {
  columns: SortableColumn<T>[];
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
  onSortChange?: (key: string, direction: 'asc' | 'desc') => void;
}

export function SortableTable<T>({
  columns,
  sortBy,
  sortDirection,
  onSortChange,
  ...props
}: SortableTableProps<T>) {
  const handleSort = (key: string) => {
    if (!onSortChange) return;
    const column = columns.find((c) => c.key === key);
    if (!column?.sortable) return;

    let newDirection: 'asc' | 'desc' = 'asc';
    if (sortBy === key && sortDirection === 'asc') {
      newDirection = 'desc';
    }
    onSortChange(key, newDirection);
  };

  const columnsWithSort: Column<T>[] = columns.map((column) => ({
    ...column,
    render: column.sortable
      ? (row, index, cellValue) => (
          <button
            type="button"
            onClick={() => handleSort(column.key)}
            className="flex items-center gap-1 hover:text-medical-primary-600 dark:hover:text-medical-primary-400"
            aria-sort={sortBy === column.key ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'}
          >
            {column.render ? column.render(row, index, cellValue) : String(cellValue ?? '')}
            {sortBy === column.key && (
              <svg
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d={sortDirection === 'asc' ? 'M5 15l7-7 7 7' : 'M19 9l-7 7-7-7'}
                />
              </svg>
            )}
          </button>
        )
      : column.render,
  }));

  return <Table<T> columns={columnsWithSort} {...props} />;
}