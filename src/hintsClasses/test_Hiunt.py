from hintsClasses.Hint import Hint, TuningValue

hints = [
        Hint('work_mem', TuningValue(0, 8, 'mb'), []),
        Hint('work_mem', TuningValue(0, 36, 'mb'), []),
        Hint('work_mem', TuningValue(0, 64, 'mb'), []),
        Hint('work_mem', TuningValue(1, 0.01, 'mem'), []),
        Hint('work_mem', TuningValue(1, 0.03, 'mem'), []),
        Hint('work_mem', TuningValue(1, 0.05, 'mem'), []),
        Hint('work_mem', TuningValue(2, 0), ['sort', 'memory available']),
        Hint('work_mem', TuningValue(2, 0), ['hash']),
        Hint('work_mem', TuningValue(2, 0), ['complex query']),
        Hint('work_mem', TuningValue(1, 0.25, 'mem'), ['analytics']),
        Hint('work_mem', TuningValue(2, 0), ['write heavy', 'dirty data in kernel page']),

        Hint('wal_buffers', TuningValue(0, 1, 'mb'), []),
        Hint('wal_buffers', TuningValue(0, 8, 'mb'), []),
        Hint('wal_buffers', TuningValue(0, 16, 'mb'), ['wal operation']),  # pg_stat_wal;
        Hint('wal_buffers', TuningValue(2, 0), ['write heavy', 'memory available']),
        Hint('wal_buffers', TuningValue(2, 1), ['not write heavy']),
        Hint('wal_buffers', TuningValue(2, 0), ['wal operation']),
        Hint('wal_buffers', TuningValue(2, 0), ['delay in wal data to disk']),  # pg_stat_wal;
        # Hint('wal_buffers', TuningValue(1, 0.03, 'shared_buffers'), ['wal operation']),
        Hint('wal_buffers', TuningValue(2, 0), ['many connections']),
        Hint('wal_buffers', TuningValue(2, 0), ['many concurrent transactions']),

        Hint('temp_buffers', TuningValue(2, 0), ['many connections']),
        Hint('temp_buffers', TuningValue(2, 0), ['frequent access temporary table']),
        # SELECT * FROM pg_stat_user_tables WHERE schemaname = 'pg_temp';

        Hint('shared_buffers', TuningValue(0, 2, 'gb'), []),
        Hint('shared_buffers', TuningValue(0, 2.5, 'gb'), []),
        Hint('shared_buffers', TuningValue(0, 4, 'gb'), []),
        Hint('shared_buffers', TuningValue(1, 0.2, 'mem'), []),
        Hint('shared_buffers', TuningValue(1, 0.4, 'mem'), []),
        Hint('shared_buffers', TuningValue(1, 0.25, 'mem'), []),
        Hint('shared_buffers', TuningValue(1, 0.15, 'mem'), ['not memory available']),
        Hint('shared_buffers', TuningValue(2, 0), ['more than 1GB mem']),
        Hint('shared_buffers', TuningValue(2, 1), ['less than 1GB mem']),
        Hint('shared_buffers', TuningValue(2, 0), ['issues with caching data']),  # 如果缓冲区命中率较低，表示数据大部分时间都没有在缓冲区中
        Hint('shared_buffers', TuningValue(2, 0), ['read heavy']),

        Hint('effective_cache_size', TuningValue(1, 0.5, 'mem'), []),
        Hint('effective_cache_size', TuningValue(1, 0.7, 'mem'), ['memory available']),
        Hint('effective_cache_size', TuningValue(1, 0.75, 'mem'), ['memory available']),
        Hint('effective_cache_size', TuningValue(1, 0.8, 'mem'), ['memory available']),
        Hint('effective_cache_size', TuningValue(2, 0), ['read heavy']),
        Hint('effective_cache_size', TuningValue(2, 0), ['read heavy', 'no index']),
        Hint('effective_cache_size', TuningValue(2, 0), ['index']),

        Hint('maintenance_work_mem', TuningValue(0, 256, 'mb'), []),
        Hint('maintenance_work_mem', TuningValue(0, 512, 'mb'), []),
        Hint('maintenance_work_mem', TuningValue(0, 1, 'gb'), ['maintenance operations']),
        Hint('maintenance_work_mem', TuningValue(1, 0.1, 'mem'), ['maintenance operations']),
        Hint('maintenance_work_mem', TuningValue(1, 0.15, 'mem'), ['maintenance operations']),
        Hint('maintenance_work_mem', TuningValue(1, 0.2, 'mem'), ['maintenance operations']),
        Hint('maintenance_work_mem', TuningValue(1, 0.05, 'mem'), []),
        Hint('maintenance_work_mem', TuningValue(2, 0), ['maintenance operations']),

        Hint('max_connections', TuningValue(2, 0), ['many connections']),
        Hint('max_connections', TuningValue(1, 4, 'cpu'), []),
        Hint('max_connections', TuningValue(1, 8, 'cpu'), []),
        Hint('max_connections', TuningValue(0, 500), []),
        Hint('max_connections', TuningValue(2, 1), ['not memory available']),
        Hint('max_connections', TuningValue(0, 25), ['not many cpus']),
        Hint('max_connections', TuningValue(0, 500), ['many cpus']),

        Hint('bgwriter_lru_multiplier', TuningValue(0, 1.5), ['write heavy']),
        Hint('bgwriter_lru_multiplier', TuningValue(0, 1), ['write heavy']),
        Hint('bgwriter_lru_multiplier', TuningValue(0, 2.5), ['write heavy']),
        Hint('bgwriter_lru_multiplier', TuningValue(0, 1), ['not write heavy']),
        Hint('bgwriter_lru_multiplier', TuningValue(0, 0.5), ['not write heavy']),
        Hint('bgwriter_lru_multiplier', TuningValue(0, 0.7), ['not write heavy']),
        Hint('bgwriter_lru_multiplier', TuningValue(0, 0.9), ['not write heavy']),
        Hint('bgwriter_lru_multiplier', TuningValue(2, 0), ['IO available']),
        Hint('bgwriter_lru_multiplier', TuningValue(2, 1), ['not IO available']),

        Hint('backend_flush_after', TuningValue(2, 0), ['write heavy', 'dirty data in kernel page']),
        Hint('backend_flush_after', TuningValue(2, 1), ['not write heavy', 'not dirty data in kernel page']),
        Hint('backend_flush_after', TuningValue(0, 20), []),
        Hint('backend_flush_after', TuningValue(0, 22), []),
        Hint('backend_flush_after', TuningValue(0, 25), []),

        Hint('bgwriter_delay', TuningValue(2, 1), ['write heavy']),
        Hint('bgwriter_delay', TuningValue(2, 0), ['write heavy', 'frequent dirty buffers']),
        Hint('bgwriter_delay', TuningValue(2, 1), ['not write heavy', 'not frequent dirty buffers']),

        Hint('max_parallel_workers', TuningValue(2, 0),
             ['parallel operations', 'large dataset', 'many cpus', 'IO available']),
        Hint('max_parallel_workers', TuningValue(2, 1), ['parallel operations', 'not many cpus']),
        Hint('max_parallel_workers', TuningValue(2, 1), ['parallel operations', 'not IO available']),
        Hint('max_parallel_workers', TuningValue(1, 4, 'cpu'), []),
        Hint('max_parallel_workers', TuningValue(1, 8, 'cpu'), []),
        Hint('max_parallel_workers', TuningValue(2, 0), ['large queries']),
        Hint('max_parallel_workers', TuningValue(1, 7, 'cpu'), ['not Dynamic Shared Memory available']),

        Hint('hash_mem_multiplier', TuningValue(2, 0), ['hash', 'memory available']),
        Hint('hash_mem_multiplier', TuningValue(2, 0), ['large queries']),

        Hint('checkpoint_flush_after', TuningValue(2, 1), ['not memory available']),  # , 'small shared_buffers']),
        Hint('checkpoint_flush_after', TuningValue(2, 0), ['memory available']),  # 'large shared_buffers']),

        Hint('max_wal_size', TuningValue(0, 8, 'gb'), []),
        Hint('max_wal_size', TuningValue(2, 0), ['large dataset']),
        Hint('max_wal_size', TuningValue(2, 0), ['write heavy']),
        Hint('max_wal_size', TuningValue(2, 1), ['not write heavy']),
        Hint('max_wal_size', TuningValue(2, 0), ['frequent checkpoint']),
        Hint('max_wal_size', TuningValue(2, 0), ['large queries']),
        Hint('max_wal_size', TuningValue(2, 1), ['not large queries']),

        Hint('vacuum_cost_page_dirty', TuningValue(2, 1), ['write heavy']),
        Hint('vacuum_cost_page_dirty', TuningValue(2, 1), ['insert or update']),
        Hint('vacuum_cost_page_dirty', TuningValue(2, 0), ['not write heavy']),

        Hint('min_parallel_table_scan_size', TuningValue(0, 8, 'mb'), []),
        Hint('min_parallel_table_scan_size', TuningValue(2, 1), ['small table']),
        Hint('min_parallel_table_scan_size', TuningValue(2, 0), ['large table']),
        Hint('min_parallel_table_scan_size', TuningValue(2, 1), ['read heavy']),
        Hint('min_parallel_table_scan_size', TuningValue(2, 0), ['not read heavy']),

        Hint('min_parallel_index_scan_size', TuningValue(0, 512, 'kb'), []),
        Hint('min_parallel_index_scan_size', TuningValue(2, 1), ['no index']),
        Hint('min_parallel_index_scan_size', TuningValue(2, 1), ['read heavy']),
        Hint('min_parallel_index_scan_size', TuningValue(2, 0), ['index']),
        Hint('min_parallel_index_scan_size', TuningValue(2, 0), ['not read heavy']),
    ]
for hint_id, hint in enumerate(hints, start=1):
    hint.set_hint_id(hint_id)

hint_ids = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 23, 24, 25, 26, 27, 28, 34, 40, 41, 42, 47, 50, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62, 66, 67, 68, 69, 70, 72, 76, 77, 78, 81, 84, 85, 86, 87, 89, 91, 92, 93, 94, 98, 99, 102, 103]
for hint in hints:

    print(str(hint.get_hint_id()) + ' : ' + hint.to_string())